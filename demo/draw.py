"""검출 결과를 이미지에 그린다. 한글 제품명을 쓰므로 동봉한 글꼴(fonts/NanumGothic-Bold.ttf, OFL)을 쓴다.

pill_detection.utils.draw_boxes는 기본 글꼴이라 한글이 안 나오고 색이 하나라서 여기 따로 둔다.
"""

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pill_detection.datasets import OTHERS_CATEGORY_ID

FONT_PATH = Path(__file__).resolve().parent / "fonts" / "NanumGothic-Bold.ttf"
SYSTEM_FONTS = [  # 동봉 글꼴이 없을 때 시도하는 순서
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "C:/Windows/Fonts/malgun.ttf",  # Windows
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",  # Linux
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
PALETTE = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#42D4F4",
           "#F032E6", "#BFEF45", "#469990", "#9A6324", "#800000", "#000075"]
OTHERS_COLOR = "#808080"
BASE_SHORT_SIDE = 976  # 대회 이미지(976×1280)의 짧은 변. 이 크기에서 선 4px, 글자 28px


def open_image(image_bytes):
    """업로드한 파일 바이트를 RGB 이미지로 연다. 휴대폰 사진의 EXIF 회전 정보를 반영해 바로 세운다.

    Args:
        image_bytes (bytes): PNG·JPEG 파일 내용.

    Returns:
        PIL.Image.Image: RGB 이미지.
    """
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def load_font(size):
    """한글 글꼴을 size 크기로 연다. 동봉 글꼴 → 시스템 글꼴 → PIL 기본 글꼴(한글 불가) 순."""
    for path in [FONT_PATH, *SYSTEM_FONTS]:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def color_for(category_id):
    """클래스마다 고정된 색. others는 회색."""
    if category_id == OTHERS_CATEGORY_ID:
        return OTHERS_COLOR
    return PALETTE[int(category_id) % len(PALETTE)]


def draw_detections(image, detections, label_of):
    """bbox와 라벨(이름, confidence)을 그린 새 RGB 이미지를 돌려준다.

    Args:
        image (PIL.Image.Image): 원본 이미지.
        detections (list[dict]): {"category_id", "bbox"([x, y, w, h] 원본 픽셀), "score"} 목록.
        label_of (Callable[[int], str]): category_id → 박스 위에 적을 이름.

    Returns:
        PIL.Image.Image: 그린 결과. 원본은 바꾸지 않는다.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    scale = min(out.size) / BASE_SHORT_SIDE
    line = max(2, round(4 * scale))
    font = load_font(max(14, round(28 * scale)))
    pad = max(2, round(4 * scale))
    for det in sorted(detections, key=lambda d: d["score"]):  # 점수 높은 것을 마지막에 그려 위로 오게
        x, y, w, h = det["bbox"]
        color = color_for(det["category_id"])
        draw.rectangle([x, y, x + w, y + h], outline=color, width=line)
        text = f"{label_of(det['category_id'])} {det['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        box_top = y - text_h - 2 * pad
        if box_top < 0:  # 위로 넘치면 박스 안쪽
            box_top = y
        box_left = min(max(0, x), max(0, out.width - text_w - 2 * pad))
        draw.rectangle([box_left, box_top, box_left + text_w + 2 * pad, box_top + text_h + 2 * pad], fill=color)
        draw.text((box_left + pad - left, box_top + pad - top), text, fill="white", font=font)
    return out
