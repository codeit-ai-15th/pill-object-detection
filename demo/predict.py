"""데모용 모델 선택·로드·예측. Ultralytics(YOLO, RT-DETR)와 torchvision Faster R-CNN을 같은 형식으로 쓴다.

어느 가중치를 쓰나
- 환경 변수 3개가 계열마다 .pt 경로 하나를 정한다(.env에 적는다. 이름은 .env.example).
    PILL_DEMO_YOLO, PILL_DEMO_RTDETR, PILL_DEMO_FRCNN
- 상대 경로는 레포 루트 기준이다. 비어 있는 변수의 계열은 데모에 안 보인다.
- 로드할 때 파일 내용으로 계열을 판별하고(detect_family), 변수와 다르면 오류를 낸다.

예측 형식 (pill_detection.evaluate의 예측 dict와 같다. file_name만 없다)
- {"category_id": K-코드 숫자(others는 pill_detection.datasets.OTHERS_CATEGORY_ID), "bbox": [x, y, w, h] 원본 픽셀, "score": float}

체크포인트 형식
- Ultralytics best.pt: {"model": DetectionModel | RTDETRDetectionModel, ...}. 클래스 이름(model.names)이 "3351" 같은 K-코드 문자열이거나
  1차 실험 가중치처럼 한글 제품명이다. 둘 다 category_from_model_name으로 K-코드로 바꾼다.
- Faster R-CNN: {"model": state_dict, ...}. 클래스 목록이 없다. 학습 label = 학습 폴더 categories의 idx + 1 (evaluate.py 모듈 docstring)이고,
  classes.csv의 클래스 수가 방법마다 달라(v1=v2 118, v3 96, v4 75) 분류 층 크기로 방법을 알아낸 뒤 idx_vN 열로 되돌린다.
  2차 run의 체크포인트는 config(image_size 640, nms_iou, max_detections)와 signature 키가 있고 BatchNorm을 eps 0의 FrozenBatchNorm2d로
  바꿔 학습했다. 스모크 노트북 체크포인트는 config가 없고 torchvision 기본 크기(800/1333), eps 1e-5다. 둘 다 그대로 맞춘다.

사용 예
    >>> slots = configured_weights()                 # {"yolo": WeightSlot, "rtdetr": ..., "frcnn": ...}
    >>> detector = load_model(slots["yolo"].path, "yolo")
    >>> detector.predict(Image.open("x.png"), conf=0.5)
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from drug_info import ROOT, load_classes, normalize_name
from pill_detection.datasets import OTHERS_CATEGORY_ID, OTHERS_NAME
from pill_detection.utils import xyxy_to_xywh

FAMILIES = {"yolo": "YOLO", "rtdetr": "RT-DETR", "frcnn": "Faster R-CNN"}  # 표시 순서
ENV_VARS = {"yolo": "PILL_DEMO_YOLO", "rtdetr": "PILL_DEMO_RTDETR", "frcnn": "PILL_DEMO_FRCNN"}
VARIANT_BY_N_CLASSES = {118: "v1", 96: "v3", 75: "v4"}  # v1과 v2는 클래스 수와 순서가 같다
RAW_CONF = 0.05  # 모델에서 받아 두는 최저 점수. 화면의 기준값은 이 위에서 거른다
ENV_EXAMPLE = "예) PILL_DEMO_YOLO=runs/models/exp2_v3_yolo26s_e10_s0_best.pt"


class DemoError(Exception):
    """사용자에게 그대로 보여 주는 오류. 메시지에 원인과 해결 방법을 함께 적는다."""


@dataclass
class WeightSlot:
    """계열 하나의 가중치 설정."""

    family: str  # FAMILIES의 키
    env_var: str
    path: Path | None  # 변수가 비어 있으면 None
    error: str | None = None  # 변수는 있는데 쓸 수 없는 이유

    @property
    def label(self):
        return FAMILIES[self.family]

    @property
    def usable(self):
        return self.path is not None and self.error is None


def configured_weights(env=None):
    """환경 변수 3개를 읽어 계열별 가중치 설정을 돌려준다.

    Args:
        env (Mapping | None): 읽을 환경. None이면 os.environ.

    Returns:
        dict[str, WeightSlot]: FAMILIES 순서. 변수가 비면 path None, 파일이 없으면 error에 이유.

    Raises:
        DemoError: 세 변수가 모두 비어 있을 때.
    """
    env = os.environ if env is None else env
    slots = {}
    for family, var in ENV_VARS.items():
        value = (env.get(var) or "").strip()
        if not value:
            slots[family] = WeightSlot(family, var, None)
            continue
        path = Path(os.path.expanduser(value))
        if not path.is_absolute():
            path = ROOT / path
        error = None if path.is_file() else f"{var}가 가리키는 파일이 없습니다: {path}\n[해결] .env의 {var}를 실제 .pt 경로로 고치세요."
        slots[family] = WeightSlot(family, var, path, error)
    if all(slot.path is None for slot in slots.values()):
        raise DemoError("가중치 환경 변수가 모두 비어 있습니다.\n"
                        f"[해결] .env에 {', '.join(ENV_VARS.values())} 중 하나 이상을 적으세요. {ENV_EXAMPLE}")
    return slots


def _load_checkpoint(path):
    """torch.load. 가중치를 메모리에 다 올리지 않도록 mmap을 먼저 시도한다."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError, OSError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _is_state_dict(value):
    return isinstance(value, dict) and len(value) > 0 and all(isinstance(v, torch.Tensor) for v in value.values())


def detect_family(path):
    """체크포인트 내용으로 계열을 판별한다.

    Args:
        path (str | Path): .pt 경로.

    Returns:
        str: "yolo", "rtdetr", "frcnn" 중 하나.

    Raises:
        DemoError: 두 계열 어느 쪽도 아닐 때.
    """
    path = Path(path)
    checkpoint = _load_checkpoint(path)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if _is_state_dict(model):
        if "roi_heads.box_predictor.cls_score.weight" not in model:
            raise DemoError(f"{path.name}은 Faster R-CNN(fasterrcnn_resnet50_fpn) state_dict가 아닙니다.\n"
                            "[해결] 2차·3차 실험의 Faster R-CNN best.pt/last.pt를 넣으세요.")
        return "frcnn"
    if isinstance(model, torch.nn.Module):
        return "rtdetr" if "RTDETR" in type(model).__name__.upper() else "yolo"
    raise DemoError(f"{path.name}은 지원하지 않는 체크포인트입니다 (Ultralytics best.pt 또는 Faster R-CNN state_dict만 됩니다).\n"
                    f"[해결] 학습이 저장한 .pt 파일인지 확인하세요. {ENV_EXAMPLE}")


def category_from_model_name(name, classes=None):
    """모델 클래스 이름을 category_id로 바꾼다. "3351" → 3351, "others" → OTHERS_CATEGORY_ID, "보령부스파정 5mg" → 1900.

    Args:
        name (str): Ultralytics model.names의 값.
        classes (dict | None): load_classes()의 결과. None이면 읽는다.

    Returns:
        int: category_id.

    Raises:
        DemoError: 숫자도, others도, classes.csv의 제품명도 아닐 때.
    """
    text = str(name).strip()
    if text == OTHERS_NAME:
        return OTHERS_CATEGORY_ID
    if text.isdigit():
        return int(text)
    classes = load_classes() if classes is None else classes
    key = normalize_name(text)
    for category_id, row in classes.items():
        if row["name"] == key:
            return category_id
    raise DemoError(f"모델 클래스 이름 '{name}'을 K-코드로 바꿀 수 없습니다. data/datasets/classes.csv에 같은 제품명이 없습니다.\n"
                    "[해결] 2차 실험 이후 학습 폴더(클래스 이름이 K-코드 숫자)로 학습한 가중치를 쓰세요.")


def variant_from_n_classes(n_classes):
    """Faster R-CNN 분류 층의 클래스 수(배경 제외)로 학습 방법(v1~v4)을 알아낸다."""
    if n_classes not in VARIANT_BY_N_CLASSES:
        raise DemoError(f"Faster R-CNN 클래스 수가 {n_classes}개라 어느 학습 방법인지 알 수 없습니다 "
                        f"(v1·v2 118, v3 96, v4 75만 지원).\n[해결] 2차·3차 실험 학습 폴더로 학습한 가중치를 쓰세요. 1차 가중치는 지원하지 않습니다.")
    return VARIANT_BY_N_CLASSES[n_classes]


def frcnn_category_table(variant, classes=None):
    """Faster R-CNN 예측 label → category_id 표. label = classes.csv의 idx_<variant> + 1.

    Args:
        variant (str): "v1"~"v4".
        classes (dict | None): load_classes()의 결과.

    Returns:
        dict[int, int]: label → category_id.
    """
    classes = load_classes() if classes is None else classes
    return {row["idx"][variant] + 1: category_id for category_id, row in classes.items() if row["idx"][variant] is not None}


def _freeze_batchnorm(module, eps):
    """BatchNorm2d를 FrozenBatchNorm2d로 바꾼다. 학습 때와 같은 층이어야 state_dict가 strict로 들어간다."""
    from torchvision.ops import FrozenBatchNorm2d

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=eps))
        elif isinstance(child, FrozenBatchNorm2d):
            child.eps = eps
        else:
            _freeze_batchnorm(child, eps)


class UltralyticsDetector:
    """YOLO·RT-DETR 가중치. kind는 로드 뒤 실제 클래스로 정한다."""

    def __init__(self, path):
        from ultralytics import YOLO

        self.path = Path(path)
        self.model = YOLO(str(self.path))
        self.kind = "rtdetr" if type(self.model).__name__ == "RTDETR" else "yolo"
        classes = load_classes()
        self.category_of = {int(i): category_from_model_name(n, classes) for i, n in self.model.names.items()}
        self.n_classes = len(self.category_of)
        self.variant = None

    def predict(self, image, conf=0.5):
        """이미지 한 장을 예측한다. cuda가 있으면 거기서, 없으면 CPU에서 돈다.

        Args:
            image (PIL.Image.Image): RGB 이미지.
            conf (float): 이 점수 미만은 버린다.

        Returns:
            list[dict]: 모듈 docstring의 예측 형식. 점수 내림차순.
        """
        device = 0 if torch.cuda.is_available() else "cpu"
        result = self.model.predict(image, conf=max(conf, RAW_CONF), device=device, verbose=False)[0]
        boxes = result.boxes
        detections = [{"category_id": self.category_of[int(cls)], "bbox": xyxy_to_xywh(xyxy), "score": float(score)}
                      for xyxy, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist())
                      if score >= conf]
        return sorted(detections, key=lambda d: -d["score"])


class FrcnnDetector:
    """torchvision Faster R-CNN 가중치."""

    kind = "frcnn"

    def __init__(self, path):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.path = Path(path)
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        n_outputs = state_dict["roi_heads.box_predictor.cls_score.weight"].shape[0]
        self.n_classes = n_outputs - 1
        self.variant = variant_from_n_classes(self.n_classes)
        self.category_of = frcnn_category_table(self.variant)
        config = checkpoint.get("config") or {}
        kwargs = {"weights": None, "weights_backbone": None, "box_score_thresh": RAW_CONF}
        if config.get("image_size"):
            kwargs["min_size"] = kwargs["max_size"] = int(config["image_size"])
        if config.get("nms_iou"):
            kwargs["box_nms_thresh"] = float(config["nms_iou"])
        if config.get("max_detections"):
            kwargs["box_detections_per_img"] = int(config["max_detections"])
        self.image_size = kwargs.get("min_size")
        model = fasterrcnn_resnet50_fpn(**kwargs)
        _freeze_batchnorm(model, eps=0.0 if "signature" in checkpoint else 1e-5)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, n_outputs)
        model.load_state_dict(state_dict, strict=True)
        self.model = model.eval()  # 장치는 예측할 때 정한다 (ZeroGPU는 @spaces.GPU 함수 안에서만 cuda가 보인다)

    def predict(self, image, conf=0.5):
        """이미지 한 장을 예측한다. UltralyticsDetector.predict와 같은 형식. cuda가 있으면 거기서, 없으면 CPU에서 돈다."""
        from torchvision.transforms.functional import to_tensor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        tensor = to_tensor(image.convert("RGB")).to(device)
        with torch.inference_mode():
            out = model([tensor])[0]
        detections = [{"category_id": self.category_of[int(label)], "bbox": xyxy_to_xywh(xyxy), "score": float(score)}
                      for xyxy, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist())
                      if score >= conf]
        return sorted(detections, key=lambda d: -d["score"])


def load_model(path, expected_family=None):
    """가중치를 불러온다. 계열은 파일 내용으로 정한다.

    Args:
        path (str | Path): .pt 경로.
        expected_family (str | None): 환경 변수가 말하는 계열. 다르면 오류.

    Returns:
        UltralyticsDetector | FrcnnDetector: predict(image, conf)와 kind, n_classes, variant를 가진 객체.

    Raises:
        DemoError: 파일이 없거나, 지원하지 않거나, 계열이 다를 때.
    """
    path = Path(path)
    if not path.is_file():
        raise DemoError(f"가중치 파일이 없습니다: {path}\n[해결] .env의 경로를 확인하세요. {ENV_EXAMPLE}")
    family = detect_family(path)
    if expected_family is not None and family != expected_family:
        raise DemoError(f"{ENV_VARS[expected_family]}가 가리키는 {path.name}은 {FAMILIES[family]} 가중치입니다.\n"
                        f"[해결] 이 경로를 .env의 {ENV_VARS[family]}에 적으세요.")
    detector = FrcnnDetector(path) if family == "frcnn" else UltralyticsDetector(path)
    if detector.kind != family:  # 파일 판별과 실제 로드 결과가 어긋나면 코드 문제다
        raise DemoError(f"{path.name}의 계열 판별이 일치하지 않습니다 ({family} ≠ {detector.kind}).\n[해결] 이슈로 알려 주세요.")
    return detector
