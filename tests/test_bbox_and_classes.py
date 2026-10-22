"""bbox 형식 변환과 클래스 매핑 검사. 둘 다 틀려도 학습은 돌아가서 눈치채기 어렵다.

    uv run pytest
"""

import pytest

from pill_detection.data import classes
from pill_detection.utils import clip_xywh, xywh_to_yolo, xyxy_to_xywh, yolo_to_xywh

IMG_W, IMG_H = 976, 1280  # 이 데이터셋의 이미지 크기


def test_xywh_yolo_round_trip():
    bbox = [600, 867, 260, 230]
    yolo = xywh_to_yolo(bbox, IMG_W, IMG_H)
    assert all(0 <= v <= 1 for v in yolo)
    assert yolo_to_xywh(yolo, IMG_W, IMG_H) == pytest.approx(bbox)


def test_xywh_to_yolo_uses_center():
    # 이미지 정가운데에 있는 bbox는 중심 좌표가 (0.5, 0.5)
    assert xywh_to_yolo([388, 540, 200, 200], IMG_W, IMG_H)[:2] == pytest.approx([0.5, 0.5])


def test_xyxy_to_xywh():
    assert xyxy_to_xywh([10, 20, 110, 70]) == [10, 20, 100, 50]


def test_clip_partly_outside():
    assert clip_xywh([-10, 1200, 100, 100], IMG_W, IMG_H) == [0, 1200, 90, 80]


def test_clip_fully_outside_becomes_empty():
    # 원본에 실제로 있는 좌표 오타 (annotation_id 903)
    assert clip_xywh([6567, 625, 311, 315], IMG_W, IMG_H)[2] == 0


def test_class_mapping_round_trip():
    for idx in range(classes.NUM_CLASSES):
        assert classes.category_to_yolo(classes.yolo_to_category(idx)) == idx


def test_category_ids_unique():
    ids = [category_id for category_id, _ in classes.CLASSES]
    assert len(ids) == len(set(ids))
