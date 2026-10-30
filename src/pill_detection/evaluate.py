"""2차 실험 val 채점과 Kaggle 제출 CSV 변환.

채점 규칙 (evaluate_val)
- 지표: mAP@[0.75:0.95]. IoU 기준 0.75, 0.80, 0.85, 0.90, 0.95의 AP를 클래스와 기준에 대해 평균한다. 대회 지표와 같다. (RULE.md)
- 정답은 scripts/build_dataset.py가 만드는 <out>/eval/val_gt.json이다. val의 모든 객체가 원래 K-코드로 들어 있고, 4가지 방법이 같다.
- 대회 클래스(val_gt.json categories의 competition = 1)만 채점한다.
- 아래 예측은 채점 전에 버린다. (drop_ignored)
    - 대회 클래스가 아닌 클래스(others 포함)로 낸 예측
    - 대회 클래스 밖 정답 box와 IoU 0.5 이상 겹친 예측 (클래스와 상관없이)
    - test에는 대회 클래스 밖 알약이 없다. 그 알약에 낸 예측을 오답으로 세면, 그 알약을 학습한 v1~v3만 감점되고 others로 묶은 v4는 감점되지 않는다.
- 점수 3개
    - val: val 전체 (1차 val Kaggle 47장 + AI Hub)
    - kaggle: 1차 val Kaggle 47장만. 1차 실험 점수와 비교할 때 쓴다. 1차 best.pt도 같은 함수로 다시 잰다.
    - test_only: test 전용 클래스(categories의 test_only = 1)만

예측 형식
- 예측 하나는 dict이다: {"file_name": "K-..._70_000_200.png", "category_id": 3351, "bbox": [x, y, w, h], "score": 0.93}
    - bbox는 xywh 원본 픽셀이다. (pill_detection.utils 모듈 docstring)
    - category_id는 K-코드 숫자이고, others는 pill_detection.datasets.OTHERS_CATEGORY_ID이다.
- Ultralytics 결과는 predictions_from_ultralytics로 바꾼다. 모델 클래스 이름("3351", "others")을 category_from_name으로 K-코드로 바꾼다.

사용 예
    >>> from ultralytics import YOLO
    >>> from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics
    >>> results = YOLO("best.pt").predict("/content/exp2/v2/images/val", conf=0.001, stream=True)
    >>> scores = evaluate_val(predictions_from_ultralytics(results), "/content/exp2/v2/eval/val_gt.json")
    >>> scores["val"], scores["kaggle"], scores["test_only"]
"""

import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np

from pill_detection.datasets import OTHERS_CATEGORY_ID, OTHERS_NAME
from pill_detection.utils import xyxy_to_xywh

IOU_THRESHOLDS = np.linspace(0.75, 0.95, 5)  # 대회 지표의 IoU 기준
IGNORE_IOU = 0.5  # 대회 클래스 밖 정답 box와 이만큼 겹친 예측은 버린다
SUBMISSION_COLUMNS = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]


def box_iou(box_a, box_b):
    """두 xywh box의 IoU(겹친 넓이 ÷ 합친 넓이, 1이면 완전히 같음)."""
    inter_w = max(0, min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0, min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def load_gt(val_gt):
    """채점 정답을 읽는다.

    Args:
        val_gt (str | Path | dict): eval/val_gt.json 경로, 또는 이미 읽은 dict.

    Returns:
        dict: COCO 형식 dict (images, annotations, categories).
    """
    if isinstance(val_gt, dict):
        return val_gt
    return json.loads(Path(val_gt).read_text(encoding="utf-8"))


def category_from_name(name):
    """모델 클래스 이름을 category_id로 바꾼다. ("3351" → 3351, "others" → OTHERS_CATEGORY_ID)

    Args:
        name (str): data.yaml의 names 값.

    Returns:
        int: category_id.
    """
    return OTHERS_CATEGORY_ID if name == OTHERS_NAME else int(name)


def predictions_from_ultralytics(results):
    """Ultralytics predict 결과를 예측 dict 목록으로 바꾼다.

    Args:
        results (Iterable[ultralytics.engine.results.Results]): model.predict()의 결과. stream=True로 받은 generator도 된다.

    Returns:
        list[dict]: {"file_name", "category_id", "bbox"(xywh 원본 픽셀), "score"} 목록.
    """
    predictions = []
    for result in results:
        name = Path(result.path).name
        boxes = result.boxes
        for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            predictions.append({"file_name": name, "category_id": category_from_name(result.names[int(cls)]),
                                "bbox": xyxy_to_xywh(xyxy), "score": float(conf)})
    return predictions


def drop_ignored(predictions, gt):
    """채점하지 않을 예측을 버린다. (모듈 docstring의 채점 규칙)

    Args:
        predictions (list[dict]): 예측 목록. file_name은 gt의 images에 있어야 한다.
        gt (dict): load_gt()의 결과.

    Returns:
        list[dict]: 남긴 예측. 순서는 그대로다.
    """
    competition = {c["id"] for c in gt["categories"] if c["competition"]}
    image_of = {image["file_name"]: image["id"] for image in gt["images"]}
    ignore_boxes = {}  # image id → 대회 클래스 밖 정답 box
    for ann in gt["annotations"]:
        if ann["category_id"] not in competition:
            ignore_boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
    kept = []
    for pred in predictions:
        if pred["category_id"] not in competition:
            continue
        if any(box_iou(pred["bbox"], box) >= IGNORE_IOU for box in ignore_boxes.get(image_of[pred["file_name"]], [])):
            continue
        kept.append(pred)
    return kept


def coco_map(gt, predictions, image_ids=None, category_ids=None):
    """mAP@[0.75:0.95]와 클래스별 AP를 계산한다. (pycocotools, 01 노트북 6-1과 같은 설정)

    Args:
        gt (dict): COCO 형식 정답. (load_gt)
        predictions (list[dict]): 예측 목록. file_name이나 image_id 중 하나가 있어야 한다.
        image_ids (Iterable[int] | None): 이 이미지만 채점한다. None이면 전부.
        category_ids (Iterable[int] | None): 이 클래스만 채점한다. None이면 정답에 있는 클래스 전부.

    Returns:
        tuple[float, dict[int, float]]: (mAP, 클래스별 AP). 채점 범위에 정답이 없는 클래스는 클래스별 AP에 없다.
            예측이 하나도 없으면 (0.0, {}).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_of = {image["file_name"]: image["id"] for image in gt["images"]}
    results = [{"image_id": p.get("image_id", image_of.get(p.get("file_name"))), "category_id": p["category_id"],
                "bbox": list(p["bbox"]), "score": p["score"]} for p in predictions]
    if not results:
        return 0.0, {}
    gt = {**gt, "categories": [{"id": c["id"]} for c in gt["categories"]]
          + [{"id": c} for c in sorted({r["category_id"] for r in results} - {c["id"] for c in gt["categories"]})]}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools의 진행 메시지 숨김
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = IOU_THRESHOLDS
        image_ids = None if image_ids is None else set(image_ids)
        if image_ids is not None:
            evaluator.params.imgIds = sorted(image_ids)
        present = {a["category_id"] for a in gt["annotations"] if image_ids is None or a["image_id"] in image_ids}
        evaluator.params.catIds = sorted(present if category_ids is None else present & set(category_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]  # [IoU 기준, recall, 클래스], 면적 전체, 최대 검출 수 100
    per_class = {}
    for k, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, k]
        per_class[category_id] = float(values[values > -1].mean()) if (values > -1).any() else 0.0
    return float(evaluator.stats[0]), per_class


def evaluate_val(predictions, val_gt):
    """2차 val 점수 3개와 클래스별 AP를 계산한다. (모듈 docstring의 채점 규칙)

    Args:
        predictions (list[dict]): val 이미지의 예측 목록. predictions_from_ultralytics의 결과 형식이다.
        val_gt (str | Path | dict): eval/val_gt.json 경로나 dict.

    Returns:
        dict: {"val": float, "kaggle": float, "test_only": float, "per_class": {category_id: AP}, "n_predictions": 채점한 예측 수}

    Raises:
        ValueError: 예측의 file_name이 val 정답에 없을 때. (다른 방법의 val이나 test 이미지를 넣은 경우)
    """
    gt = load_gt(val_gt)
    names = {image["file_name"] for image in gt["images"]}
    unknown = sorted({p["file_name"] for p in predictions} - names)
    if unknown:
        raise ValueError(f"val 정답에 없는 이미지의 예측이 {len(unknown)}장 있습니다. 예: {unknown[0]}\n"
                         "[해결] <out>/images/val의 이미지로 예측하고, 같은 <out>/eval/val_gt.json으로 채점하세요.")
    kept = drop_ignored(predictions, gt)
    competition = {c["id"] for c in gt["categories"] if c["competition"]}
    test_only = {c["id"] for c in gt["categories"] if c.get("test_only")}
    kaggle_images = {image["id"] for image in gt["images"] if image.get("source") == "kaggle"}
    val_map, per_class = coco_map(gt, kept, category_ids=competition)
    kaggle_map, _ = coco_map(gt, kept, image_ids=kaggle_images, category_ids=competition)
    test_only_map, _ = coco_map(gt, kept, category_ids=test_only)
    return {"val": val_map, "kaggle": kaggle_map, "test_only": test_only_map, "per_class": per_class,
            "n_predictions": len(kept)}


def competition_from_classes(classes_csv):
    """classes.csv에서 대회 클래스를 읽는다.

    Args:
        classes_csv (str | Path): data/datasets/classes.csv (또는 학습 폴더의 classes.csv 사본).

    Returns:
        set[int]: 대회 클래스의 category_id.
    """
    with open(classes_csv, newline="", encoding="utf-8-sig") as f:
        return {int(row["category_id"]) for row in csv.DictReader(f) if row["competition"] == "1"}


def write_submission(predictions, path, competition):
    """test 예측을 Kaggle 제출 CSV로 쓴다. 대회 클래스 예측만 남긴다.

    Args:
        predictions (list[dict]): test 이미지의 예측 목록. file_name은 test 파일 이름(예: 1.png)이다.
        path (str | Path): 저장할 CSV 경로. 상위 폴더가 없으면 만든다.
        competition (set[int]): 대회 클래스. (competition_from_classes)

    Returns:
        int: 쓴 줄 수. image_id는 파일 이름의 숫자, annotation_id는 1부터 매긴다.
    """
    rows = [p for p in predictions if p["category_id"] in competition]
    rows.sort(key=lambda p: (int(Path(p["file_name"]).stem), -p["score"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(SUBMISSION_COLUMNS)
        for i, p in enumerate(rows, start=1):
            x, y, w, h = p["bbox"]
            writer.writerow([i, int(Path(p["file_name"]).stem), p["category_id"],
                             round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(p["score"], 4)])
    return len(rows)
