"""테스트 공통 fixture.

- 실제 데이터(data/raw, AI Hub 원본)가 없으면 그 데이터를 쓰는 테스트는 이유를 적고 건너뛴다.
- 학습 데이터 전체를 만드는 실행(make_train_dataset)은 오래 걸려서 세션에서 한 번만 하고, 끝나면 결과 폴더를 지운다.
- 이미 만든 AI Hub 결과 폴더를 검사만 하려면 PILL_TEST_AIHUB_EXPORT에 그 폴더(train_annotations, train_images가 있는 곳)를 넣는다.
    예: PILL_TEST_AIHUB_EXPORT=data/processed/ai_hub uv run pytest tests/test_train_dataset.py -k aihub
"""

import os
import shutil
from pathlib import Path

import pytest

from helpers import (
    AIHUB_ANN_DIR,
    AIHUB_IMG_DIR,
    CHANGE_CSV_V1,
    KAGGLE_ANN_DIR,
    KAGGLE_IMG_DIR,
    LABEL_FIX_CSV,
    PROJECT_ROOT,
    TEST_OUT_DIR,
    tree_hash,
)


def skip_without(*folders):
    """폴더가 하나라도 없으면 테스트를 건너뛴다. 건너뛴 이유에 받는 방법을 적는다."""
    missing = [Path(f).relative_to(PROJECT_ROOT).as_posix() for f in folders if not Path(f).is_dir()]
    if missing:
        pytest.skip(f"데이터 폴더가 없어 건너뜀: {', '.join(missing)} (README의 학습 데이터 준비를 보고 받으세요)")


@pytest.fixture(scope="session")
def train_dataset(tmp_path_factory):
    """make_train_dataset을 data/processed/_pytest/에 한 번 실행한 결과.

    변경 기록은 임시 폴더에 main의 v1(apply_label_fixes.csv)을 복사해 두고 그 옆에 쓴다.
    그래서 data/label_fix는 건드리지 않고, v2로 저장되는지 볼 수 있다.

    Returns:
        dict: result(make_train_dataset 반환값), kwargs(넣은 인자), raw_before(실행 전 원본 해시)
    """
    skip_without(KAGGLE_ANN_DIR, KAGGLE_IMG_DIR, AIHUB_ANN_DIR, AIHUB_IMG_DIR)
    from pill_detection.data import make_train_dataset

    change_dir = tmp_path_factory.mktemp("label_fix")
    shutil.copy(CHANGE_CSV_V1, change_dir / CHANGE_CSV_V1.name)
    kwargs = {
        "fix_csv": LABEL_FIX_CSV,
        "kaggle_ann_dir": KAGGLE_ANN_DIR,
        "kaggle_img_dir": KAGGLE_IMG_DIR,
        "aihub_ann_dir": AIHUB_ANN_DIR,
        "aihub_img_dir": AIHUB_IMG_DIR,
        "kaggle_out_ann_dir": TEST_OUT_DIR / "train_annotations",
        "aihub_out_dir": TEST_OUT_DIR / "ai_hub",
        "change_csv": change_dir / CHANGE_CSV_V1.name,
    }
    raw_before = {"kaggle": tree_hash(KAGGLE_ANN_DIR), "aihub": tree_hash(AIHUB_ANN_DIR)}
    shutil.rmtree(TEST_OUT_DIR, ignore_errors=True)
    result = make_train_dataset(**kwargs)
    yield {"result": result, "kwargs": kwargs, "raw_before": raw_before}
    shutil.rmtree(TEST_OUT_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def aihub_export(request):
    """검사할 AI Hub 결과 폴더. (annotation 폴더, 이미지 폴더)

    PILL_TEST_AIHUB_EXPORT가 있으면 그 폴더를 그대로 검사하고, 없으면 train_dataset이 만든 폴더를 검사한다.
    """
    given = os.environ.get("PILL_TEST_AIHUB_EXPORT")
    if given:
        root = Path(given) if Path(given).is_absolute() else PROJECT_ROOT / given
        skip_without(root / "train_annotations", root / "train_images")
        return root / "train_annotations", root / "train_images"
    run = request.getfixturevalue("train_dataset")
    out = Path(run["kwargs"]["aihub_out_dir"])
    return out / "train_annotations", out / "train_images"
