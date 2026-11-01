"""합친 정제 목록(data/label_fix/label_fixes.csv)을 읽는 규칙을 본다.

- Kaggle 줄(21줄)은 main의 옛 정제 목록과 값이 같아야 한다.
- AI Hub 줄은 01 노트북의 정제 목록(1,136줄)에서 제외 이미지의 다른 줄을 빼고, 한 알약의 여러 줄을 최종 값 한 줄로 합친 것이다.
"""

import collections

import pytest

from conftest import skip_without
from helpers import KAGGLE_ANN_DIR, KAGGLE_FIXED_HASH, KAGGLE_IMG_DIR, LABEL_FIX_CSV, LEGACY_FIX_CSV, tree_hash
from pill_detection.data import FIX_COLUMNS, make_fixed_annotations, read_label_fixes

HEADER = ",".join(FIX_COLUMNS) + "\n"
FN = "K-003351-016262-018357_0_2_0_2_75_000_200.png"


def test_columns():
    """열 이름과 순서. 고친 뒤의 값은 모두 _after로 끝난다."""
    assert FIX_COLUMNS == ["source", "file_name", "category_id", "action",
                           "bbox_after", "area_after", "category_id_after", "image_fix"]
    assert LABEL_FIX_CSV.read_text(encoding="utf-8").splitlines()[0] == ",".join(FIX_COLUMNS)


def test_rows_per_source():
    """source별 줄 수와 AI Hub 줄의 종류."""
    fixes = read_label_fixes(LABEL_FIX_CSV)
    assert collections.Counter(fix["source"] for fix in fixes) == {"kaggle": 21, "aihub": 1093}
    aihub = [fix for fix in fixes if fix["source"] == "aihub"]
    assert collections.Counter(fix["action"] for fix in aihub) == {"modify": 987, "add": 70, "exclude": 36}
    assert len({fix["file_name"] for fix in aihub if fix["action"] == "exclude"}) == 35
    assert sum(fix["category_id_after"] is not None for fix in aihub) == 38
    assert [fix["image_fix"] for fix in aihub if fix["image_fix"]] == [{"back_color": "연회색 배경"}] * 4


def test_returned_dict_shape():
    """한 줄이 돌려주는 dict. 빈칸은 None, bbox는 리스트, image_fix는 {필드: 값}이다."""
    fixes = read_label_fixes(LABEL_FIX_CSV)
    assert next(fix for fix in fixes if (fix["file_name"], fix["category_id"]) == (FN, 18357)) == {
        "source": "kaggle", "file_name": FN, "category_id": 18357, "action": "modify",
        "bbox_after": [567, 625, 311, 315], "area_after": 97965, "category_id_after": None, "image_fix": None,
    }
    exclude = next(fix for fix in fixes if fix["action"] == "exclude")
    assert (exclude["bbox_after"], exclude["area_after"], exclude["category_id_after"]) == (None, None, None)


def test_kaggle_rows_equal_legacy_csv():
    """Kaggle 줄은 옛 5열 정제 목록을 읽은 결과와 같다. 옛 목록은 source=kaggle로 읽는다."""
    legacy = read_label_fixes(LEGACY_FIX_CSV)
    assert {fix["source"] for fix in legacy} == {"kaggle"}
    assert read_label_fixes(LABEL_FIX_CSV, source="kaggle") == legacy


def test_source_filter():
    """source를 주면 그 줄만 돌려준다."""
    aihub = read_label_fixes(LABEL_FIX_CSV, source="aihub")
    assert len(aihub) == 1093 and {fix["source"] for fix in aihub} == {"aihub"}


@pytest.mark.parametrize("row, message", [
    (f'web,{FN},18357,modify,"[567, 625, 311, 315]",97965,,\n', "source"),
    (f'aihub,{FN},18357,exclude,"[567, 625, 311, 315]",97965,,\n', "exclude"),
    (f"aihub,{FN},18357,modify,,,,\n", "bbox_after"),
    (f'aihub,{FN},18357,modify,"[567, 625, 311, 315]",1,,\n', "area_after"),
    (f'aihub,{FN},18357,modify,"[567, 625, 311, 315]",97965,abc,\n', "category_id_after"),
    (f'aihub,{FN},18357,modify,"[567, 625, 311, 315]",97965,,color=red\n', "image_fix"),
    (f'aihub,{FN},18357,modify,"[567, 625, 311, 315]",97965,,back_color\n', "image_fix"),
    (f'aihub,{FN},18357,fix,"[567, 625, 311, 315]",97965,,\n', "action"),
])
def test_rejects_bad_rows(tmp_path, row, message):
    """형식이 틀린 줄은 ValueError로 멈추고, 메시지에 틀린 열을 적는다."""
    path = tmp_path / "label_fixes.csv"
    path.write_text(HEADER + row, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_label_fixes(path)


def test_rejects_duplicate_key(tmp_path):
    """같은 (source, file_name, category_id)는 한 줄만 쓴다. source가 다르면 따로 본다."""
    row = f'{{}},{FN},18357,modify,"[567, 625, 311, 315]",97965,,\n'
    path = tmp_path / "label_fixes.csv"
    path.write_text(HEADER + row.format("kaggle") + row.format("aihub"), encoding="utf-8")
    assert len(read_label_fixes(path)) == 2
    path.write_text(HEADER + row.format("aihub") + row.format("aihub"), encoding="utf-8")
    with pytest.raises(ValueError, match="같습니다"):
        read_label_fixes(path)


def test_make_fixed_annotations_uses_only_kaggle_rows(tmp_path):
    """합친 정제 목록을 make_fixed_annotations에 넣으면 Kaggle 줄만 써서, 옛 목록과 같은 사본을 만든다."""
    skip_without(KAGGLE_ANN_DIR, KAGGLE_IMG_DIR)
    out_dir = tmp_path / "train_annotations"
    result = make_fixed_annotations(LABEL_FIX_CSV, KAGGLE_ANN_DIR, KAGGLE_IMG_DIR, out_dir, tmp_path / "apply_label_fixes.csv")
    assert (result["n_base"], result["n_out"]) == (763, 771)
    assert tree_hash(out_dir) == KAGGLE_FIXED_HASH
