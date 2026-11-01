"""2차 val 채점과 제출 CSV 변환(pill_detection.evaluate)을 장난감 정답으로 본다."""

import csv
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pycocotools")

from pill_detection import evaluate as ev  # noqa: E402
from pill_detection.datasets import OTHERS_CATEGORY_ID  # noqa: E402

# 사진 2장: kaggle 1장, aihub 1장. 클래스 1(대회, Kaggle), 5(대회, test 전용), 9(대회 클래스 밖)
GT = {
    "images": [{"id": 1, "file_name": "k.png", "width": 100, "height": 100, "source": "kaggle"},
               {"id": 2, "file_name": "a.png", "width": 100, "height": 100, "source": "aihub"}],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 5, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 30, 20], "area": 600, "iscrowd": 0},
        {"id": 4, "image_id": 2, "category_id": 9, "bbox": [60, 10, 20, 30], "area": 600, "iscrowd": 0},
    ],
    "categories": [{"id": 1, "competition": 1, "test_only": 0}, {"id": 5, "competition": 1, "test_only": 1},
                   {"id": 9, "competition": 0, "test_only": 0}],
}
NAME = {1: "k.png", 2: "a.png"}


def perfect():
    """정답 box를 그대로 예측으로 넣는다."""
    return [{"file_name": NAME[a["image_id"]], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
            for a in GT["annotations"]]


def test_perfect_and_wrong_predictions():
    """정답을 넣으면 점수 3개가 모두 1.0, 대회 클래스를 모두 틀리면 0.0이다."""
    scores = ev.evaluate_val(perfect(), GT)
    assert (scores["val"], scores["kaggle"], scores["test_only"]) == pytest.approx((1.0, 1.0, 1.0))
    assert scores["per_class"] == pytest.approx({1: 1.0, 5: 1.0})
    swap = {1: 5, 5: 1}
    wrong = [{**p, "category_id": swap.get(p["category_id"], p["category_id"])} for p in perfect()]
    scores = ev.evaluate_val(wrong, GT)
    assert (scores["val"], scores["kaggle"], scores["test_only"]) == pytest.approx((0.0, 0.0, 0.0))


def test_predictions_on_noncompetition_pills_are_ignored():
    """대회 클래스 밖 정답 box에 낸 예측(어느 클래스든)과 others 예측은 채점에서 빠진다."""
    on_pill = {"file_name": "a.png", "category_id": 1, "bbox": [61, 11, 20, 30], "score": 0.99}
    others = {"file_name": "k.png", "category_id": OTHERS_CATEGORY_ID, "bbox": [30, 30, 10, 10], "score": 0.99}
    predictions = perfect() + [on_pill, others]
    assert ev.drop_ignored(predictions, GT) == [p for p in perfect() if p["category_id"] != 9]
    assert ev.evaluate_val(predictions, GT)["val"] == pytest.approx(1.0)
    assert ev.coco_map(GT, [p for p in predictions if p is not others])[0] < 0.99  # 무시하지 않으면 감점된다


def test_scores_use_their_own_scope():
    """kaggle 점수는 Kaggle 사진만, test_only 점수는 test 전용 클래스만 본다."""
    aihub_wrong = [p for p in perfect() if p["file_name"] == "k.png"]
    scores = ev.evaluate_val(aihub_wrong, GT)
    assert scores["kaggle"] == pytest.approx(1.0) and scores["val"] < 0.99
    only_5 = [p for p in perfect() if p["category_id"] == 5]
    scores = ev.evaluate_val(only_5, GT)
    assert scores["test_only"] == pytest.approx(1.0) and scores["val"] < 0.99


def test_scores_can_be_saved_as_json():
    """클래스별 AP의 key는 Python int라서 점수를 그대로 json.dumps로 저장할 수 있다."""
    scores = ev.evaluate_val(perfect(), GT)
    assert all(type(c) is int for c in scores["per_class"])
    assert json.loads(json.dumps(scores))["per_class"] == {"1": pytest.approx(1.0), "5": pytest.approx(1.0)}


def test_empty_scope_score_is_none():
    """채점 범위에 정답이 없으면 점수는 -1이 아니라 None이다. (test 전용 클래스나 Kaggle 사진이 없는 정답)"""
    no_test_only = {**GT, "categories": [{**c, "test_only": 0} for c in GT["categories"]]}
    assert ev.evaluate_val(perfect(), no_test_only)["test_only"] is None
    no_kaggle = {**GT, "images": [{**image, "source": "aihub"} for image in GT["images"]]}
    scores = ev.evaluate_val(perfect(), no_kaggle)
    assert scores["kaggle"] is None and scores["val"] == pytest.approx(1.0)


def test_unknown_image_is_refused():
    """val 정답에 없는 사진의 예측은 다른 방법의 val이나 test를 넣은 것이므로 거부한다."""
    with pytest.raises(ValueError, match="val 정답에 없는"):
        ev.evaluate_val(perfect() + [{"file_name": "x.png", "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1}], GT)


def test_predictions_from_ultralytics():
    """Ultralytics 결과의 xyxy, 클래스 이름("3351", "others")을 xywh, category_id로 바꾼다."""
    boxes = SimpleNamespace(xyxy=SimpleNamespace(tolist=lambda: [[10, 20, 40, 60], [0, 0, 5, 5]]),
                            cls=SimpleNamespace(tolist=lambda: [0, 1]), conf=SimpleNamespace(tolist=lambda: [0.9, 0.5]))
    result = SimpleNamespace(path="/content/exp2/v4/images/val/k.png", boxes=boxes, names={0: "3351", 1: "others"})
    assert ev.predictions_from_ultralytics([result]) == [
        {"file_name": "k.png", "category_id": 3351, "bbox": [10, 20, 30, 40], "score": 0.9},
        {"file_name": "k.png", "category_id": OTHERS_CATEGORY_ID, "bbox": [0, 0, 5, 5], "score": 0.5},
    ]


def test_write_submission_keeps_competition_classes(tmp_path):
    """제출 CSV는 대회 클래스 예측만 남기고, image_id는 test 파일 번호, annotation_id는 1부터 매긴다."""
    predictions = [{"file_name": "12.png", "category_id": 3351, "bbox": [1.234, 2, 3, 4], "score": 0.91234},
                   {"file_name": "3.png", "category_id": OTHERS_CATEGORY_ID, "bbox": [0, 0, 1, 1], "score": 0.99},
                   {"file_name": "3.png", "category_id": 20753, "bbox": [0, 0, 1, 1], "score": 0.99}]
    path = tmp_path / "submission.csv"
    assert ev.write_submission(predictions, path, competition={3351}) == 1
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [ev.SUBMISSION_COLUMNS, ["1", "12", "3351", "1.23", "2", "3", "4", "0.9123"]]
