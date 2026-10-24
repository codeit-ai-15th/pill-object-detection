"""bbox 형식 변환과 시각화.

이 프로젝트에서 쓰는 bbox 형식은 세 가지다. 섞어 쓰면 학습은 되는데 점수가 0에 가깝게 나온다.
- xywh: [좌상단 x, 좌상단 y, 너비, 높이], 원본 픽셀. 원본 라벨·predict()·제출 파일
- xyxy: [좌상단 x, 좌상단 y, 우하단 x, 우하단 y], 원본 픽셀. Ultralytics 추론 결과
- yolo: [중심 x, 중심 y, 너비, 높이], 이미지 크기로 나눈 0~1 값. YOLO 라벨 파일
"""

from PIL import Image, ImageDraw, ImageFont


def xywh_to_yolo(bbox, img_w, img_h):
    """xywh bbox를 YOLO 라벨 형식으로 바꾼다.

    Args:
        bbox (list[float]): [좌상단 x, 좌상단 y, 너비, 높이]. 원본 픽셀.
        img_w (int): 이미지 너비.
        img_h (int): 이미지 높이.

    Returns:
        list[float]: [중심 x, 중심 y, 너비, 높이]. 이미지 크기로 나눈 0~1 값.
    """
    x, y, w, h = bbox
    return [(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h]


def yolo_to_xywh(bbox, img_w, img_h):
    """YOLO 라벨 형식 bbox를 xywh로 바꾼다. xywh_to_yolo()의 반대.

    Args:
        bbox (list[float]): [중심 x, 중심 y, 너비, 높이]. 이미지 크기로 나눈 0~1 값.
        img_w (int): 이미지 너비.
        img_h (int): 이미지 높이.

    Returns:
        list[float]: [좌상단 x, 좌상단 y, 너비, 높이]. 원본 픽셀.
    """
    cx, cy, w, h = bbox
    w, h = w * img_w, h * img_h
    return [cx * img_w - w / 2, cy * img_h - h / 2, w, h]


def xyxy_to_xywh(bbox):
    """Ultralytics 추론 결과의 xyxy bbox를 xywh로 바꾼다.

    Args:
        bbox (list[float]): [좌상단 x, 좌상단 y, 우하단 x, 우하단 y]. 원본 픽셀.

    Returns:
        list[float]: [좌상단 x, 좌상단 y, 너비, 높이]. 원본 픽셀.
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def clip_xywh(bbox, img_w, img_h):
    """이미지 밖으로 나간 부분을 잘라낸다.

    완전히 밖에 있으면 너비나 높이가 0이 된다.

    Args:
        bbox (list[float]): [좌상단 x, 좌상단 y, 너비, 높이]. 원본 픽셀.
        img_w (int): 이미지 너비.
        img_h (int): 이미지 높이.

    Returns:
        list[float]: 이미지 안으로 잘라낸 [좌상단 x, 좌상단 y, 너비, 높이].
    """
    x, y, w, h = bbox
    x1, y1 = min(max(0, x), img_w), min(max(0, y), img_h)
    x2, y2 = min(max(0, x + w), img_w), min(max(0, y + h), img_h)
    return [x1, y1, x2 - x1, y2 - y1]


def draw_boxes(image, boxes, labels=None, color="red", width=4):
    """이미지에 xywh bbox를 그려 새 PIL 이미지로 돌려준다.

    기본 글꼴은 한글을 그리지 못하므로 labels에는 category_id 같은 숫자·영문을 넣는다.

    Args:
        image (str | Path | PIL.Image.Image): 이미지 경로나 PIL 이미지.
        boxes (list[list[float]]): 그릴 bbox 목록. 각각 [좌상단 x, 좌상단 y, 너비, 높이], 원본 픽셀.
        labels (list | None): bbox마다 위에 적을 글자. boxes와 순서가 같다. None이면 적지 않는다.
        color (str): 선과 글자의 색.
        width (int): 선 두께(픽셀).

    Returns:
        PIL.Image.Image: bbox를 그린 RGB 이미지.
    """
    image = Image.open(image) if not isinstance(image, Image.Image) else image
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    for i, (x, y, w, h) in enumerate(boxes):
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        if labels:
            draw.text((x, max(0, y - 32)), str(labels[i]), fill=color, font=font)
    return image
