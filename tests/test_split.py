"""split 파일 번호 매기기와 조합 단위 분할 검사. 틀려도 학습은 돌아가서 눈치채기 어렵다.

    uv run pytest
"""

import json
from pathlib import Path

import pytest

from pill_detection.data.convert import check_split, latest_split, make_group_split, next_split_path, split_versions

SPLIT_V1 = Path(__file__).parents[1] / "configs" / "splits" / "split_v1.json"


def test_versions_sorted_by_number(tmp_path):
    for name in ("split_v1.json", "split_v10.json", "split_v2.json", "split_v2_old.json", "notes.json"):
        (tmp_path / name).write_text("{}")
    assert [p.name for p in split_versions(tmp_path)] == ["split_v1.json", "split_v2.json", "split_v10.json"]
    assert latest_split(tmp_path).name == "split_v10.json"
    assert next_split_path(tmp_path).name == "split_v11.json"


def test_empty_splits_dir(tmp_path):
    assert latest_split(tmp_path) is None
    assert next_split_path(tmp_path).name == "split_v1.json"


def test_image_in_both_raises():
    with pytest.raises(ValueError, match="모두 있는 이미지"):
        check_split({"train": ["K-001-002_0.png"], "val": ["K-001-002_0.png"]})


def test_combo_in_both_raises():
    # 같은 조합의 각도만 다른 사진이 train과 val로 나뉜 경우
    with pytest.raises(ValueError, match="K-001-002"):
        check_split({"train": ["K-001-002_0.png"], "val": ["K-001-002_1.png"]})


def test_repo_split_v1_passes_check():
    check_split(json.loads(SPLIT_V1.read_text(encoding="utf-8")))


def test_make_group_split_keeps_combos_and_classes():
    # 조합 4개 x 각도 3장. 클래스 1은 모든 조합에, 클래스 2~4는 한 조합에만 있다
    images, annotations = [], []
    for combo, cats in (("K-001-002", [1, 2]), ("K-001-003", [1, 3]), ("K-001-004", [1, 4]), ("K-001", [1])):
        for angle in range(3):
            image_id = len(images) + 1
            images.append({"id": image_id, "file_name": f"{combo}_{angle}.png", "width": 976, "height": 1280})
            for cat in cats:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat,
                                    "bbox": [0, 0, 10, 10]})
    split = make_group_split({"images": images, "annotations": annotations, "categories": []}, val_ratio=0.25)
    check_split(split)
    assert split["val"] == ["K-001_0.png", "K-001_1.png", "K-001_2.png"]  # 이 조합만 빼도 train에 모든 클래스가 남는다
