"""이미지 → 검출 결과. 데모(demo/app.py)와 제출(scripts/make_submission.py)이 함께 쓴다. (EXPERIMENTS.md 규칙 3)

    predict(image, weights=None) -> list[dict]

    리스트의 원소 하나 = 검출된 알약 하나
    {"category_id": 3483, "name": "약 이름", "bbox": [x, y, w, h], "score": 0.93}
    bbox는 좌상단 x, y와 너비, 높이. 원본 이미지 픽셀 단위
"""

from functools import lru_cache

from ultralytics import YOLO

from pill_detection.data.classes import CLASSES, yolo_to_category, yolo_to_name
from pill_detection.utils import xyxy_to_xywh


@lru_cache(maxsize=2)
def _load_model(weights):
    """가중치를 불러온다. 같은 경로로 다시 부르면 이미 불러 둔 모델을 쓴다.

    Args:
        weights (str): 가중치 파일 경로. lru_cache의 키가 되므로 문자열로 넘긴다.

    Returns:
        ultralytics.YOLO: 불러온 모델.
    """
    return YOLO(weights)


def predict(image, weights=None, conf=0.25):
    """이미지 한 장에서 알약을 검출한다. weights를 비우면 가짜 결과(mock)를 돌려준다.

    Args:
        image (str | Path | PIL.Image.Image | numpy.ndarray): 파일 경로, PIL 이미지, numpy 배열 중 하나.
        weights (str | Path | None): 가중치 파일 경로. None이면 모델 없이 mock 결과를 돌려준다.
        conf (float): 이 점수보다 낮은 검출은 버린다. 데모는 기본값, 제출은 mAP를 위해 낮은 값(0.001)을 쓴다.

    Returns:
        list[dict]: 검출된 알약 하나당 dict 하나. 키는 category_id, name, bbox, score이고
            bbox는 [좌상단 x, 좌상단 y, 너비, 높이], 원본 픽셀이다. (모듈 docstring 참고)
    """
    if weights is None:
        return _mock_result()

    boxes = _load_model(str(weights)).predict(image, conf=conf, verbose=False)[0].boxes
    return [
        {"category_id": yolo_to_category(int(cls)), "name": yolo_to_name(int(cls)),
         "bbox": xyxy_to_xywh(xyxy), "score": score}
        for xyxy, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
    ]


def _mock_result():
    """모델 없이 데모 화면을 만들 때 쓰는 고정된 가짜 결과.

    Returns:
        list[dict]: predict()와 같은 형식의 검출 결과 2개.
    """
    return [
        {"category_id": CLASSES[0][0], "name": CLASSES[0][1], "bbox": [150.0, 250.0, 210.0, 230.0], "score": 0.91},
        {"category_id": CLASSES[1][0], "name": CLASSES[1][1], "bbox": [520.0, 700.0, 260.0, 240.0], "score": 0.78},
    ]
