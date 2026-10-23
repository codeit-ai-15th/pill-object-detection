"""data_fixes.yaml 반영(apply_fixes) 검사. 잘못 적어도 학습은 돌아가서 눈치채기 어렵다.

    uv run pytest
"""

from pathlib import Path

import pytest
import yaml

from pill_detection.data.convert import FIX_KEYS, apply_fixes

IMG_W, IMG_H = 976, 1280  # 이 데이터셋의 이미지 크기
DATA_FIXES = Path(__file__).parents[1] / "configs" / "data_fixes.yaml"


def make_coco():
    """이미지 2장, 라벨 3개짜리 작은 원본"""
    return {
        "images": [{"id": 1, "file_name": "a.png", "width": IMG_W, "height": IMG_H},
                   {"id": 2, "file_name": "b.png", "width": IMG_W, "height": IMG_H}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 3351, "bbox": [100, 100, 200, 200]},
                        {"id": 11, "image_id": 1, "category_id": 1900, "bbox": [400, 400, 200, 200]},
                        {"id": 20, "image_id": 2, "category_id": 3351, "bbox": [100, 100, 200, 200]}],
        "categories": [{"id": 1900, "name": "보령부스파정 5mg"}, {"id": 3351, "name": "일양하이트린정 2mg"}],
    }


def ann_ids(coco):
    return [a["id"] for a in coco["annotations"]]


def test_data_fixes_yaml_uses_known_keys():
    # 원본 데이터가 없어도 레포의 data_fixes.yaml에서 키 오타를 잡는다
    fixes = yaml.safe_load(DATA_FIXES.read_text(encoding="utf-8")) or {}
    assert set(fixes) <= set(FIX_KEYS)


@pytest.mark.parametrize("key", ["exclude_image", "add_annotations"])  # 오타, 처리 코드가 아직 없는 키
def test_unknown_key_raises(key):
    with pytest.raises(ValueError, match=key):
        apply_fixes(make_coco(), {key: []})


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="no_such.png"):
        apply_fixes(make_coco(), {"exclude_images": ["no_such.png"]})


@pytest.mark.parametrize("fixes", [{}, {"exclude_images": [], "drop_annotations": [], "fix_bboxes": {}},
                                   {"exclude_images": None}])  # 빈 파일, 지금 파일, 값을 비워 둔 키
def test_empty_fixes_change_nothing(fixes):
    fixed, empty = apply_fixes(make_coco(), fixes)
    assert fixed == make_coco() and empty == []


def test_exclude_image_removes_its_labels():
    fixed, _ = apply_fixes(make_coco(), {"exclude_images": ["b.png"]})
    assert [im["file_name"] for im in fixed["images"]] == ["a.png"]
    assert ann_ids(fixed) == [10, 11]


def test_drop_annotation():
    fixed, _ = apply_fixes(make_coco(), {"drop_annotations": [11]})
    assert ann_ids(fixed) == [10, 20]


def test_fix_bbox():
    fixed, _ = apply_fixes(make_coco(), {"fix_bboxes": {11: [500, 500, 100, 100]}})
    assert fixed["annotations"][1]["bbox"] == [500, 500, 100, 100]


def test_bbox_outside_image_is_dropped():
    # 원본에 실제로 있는 좌표 오타 (annotation_id 903). 잘라내면 크기가 없어져서 빠진다
    coco = make_coco()
    coco["annotations"][1]["bbox"] = [6567, 625, 311, 315]
    fixed, empty = apply_fixes(coco, {})
    assert ann_ids(fixed) == [10, 20] and empty == [11]
