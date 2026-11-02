"""3차 실험 클래스 불균형 대응(pill_detection.balance)을 장난감 학습 폴더로 본다."""

import collections
import csv
import json
import math
from types import SimpleNamespace

import pytest
import yaml

from pill_detection import balance as bl

# 이미지 7장. 클래스 10은 흔하고(5장), 20은 중간(3장), 30은 희소하다(1장)
TOY_IMAGES = {
    "k1.png": ("kaggle", [10, 20]),
    "a1.png": ("aihub", [10]),
    "a2.png": ("aihub", [10]),
    "a3.png": ("aihub", [10, 20]),
    "a4.png": ("aihub", [20]),
    "a5.png": ("aihub", [30]),
    "a6.png": ("aihub", [10]),
}
# 모델 클래스 번호(idx)가 category_id 순서와 다르게 적힌 경우도 보려고 categories를 섞어 둔다
TOY_CATEGORIES = [{"id": 30, "idx": 2}, {"id": 10, "idx": 0}, {"id": 20, "idx": 1}]


def make_dataset(root, data_yaml=True, categories=TOY_CATEGORIES):
    """build_dataset 결과처럼 coco/train.json, images/train/, data.yaml을 둔 학습 폴더를 만든다."""
    coco = {"images": [], "annotations": [], "categories": categories}
    (root / "images" / "train").mkdir(parents=True)
    for image_id, (name, (source, classes)) in enumerate(TOY_IMAGES.items(), start=1):
        (root / "images" / "train" / name).touch()
        coco["images"].append({"id": image_id, "file_name": name, "source": source})
        for c in classes:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id, "category_id": c})
    (root / "coco").mkdir()
    (root / "coco" / "train.json").write_text(json.dumps(coco), encoding="utf-8")
    if data_yaml:
        data = {"path": str(root), "train": "images/train", "val": "images/val", "nc": 3,
                "names": {0: "10", 1: "20", 2: "30"}}
        (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def test_load_train_objects(tmp_path):
    """이미지마다 출처와 객체 클래스를 객체 순서대로 읽는다."""
    objects = bl.load_train_objects(make_dataset(tmp_path))
    assert objects == {name: {"source": source, "classes": classes} for name, (source, classes) in TOY_IMAGES.items()}


def test_missing_coco_or_idx_is_refused(tmp_path):
    """coco/train.json이 없거나(--format yolo) categories에 idx가 없으면(옛 학습 폴더) 다시 만들라고 알린다."""
    with pytest.raises(FileNotFoundError, match="build_dataset.py"):
        bl.load_train_objects(tmp_path)
    old = make_dataset(tmp_path / "old", categories=[{"id": c} for c in (10, 20, 30)])
    with pytest.raises(ValueError, match="idx"):
        bl.load_train_objects(old)


def test_repeat_list_repeats_only_rare_images(tmp_path):
    """기본 threshold(f_c의 중앙값 3/7)에서는 클래스 30(1/7)만 r = √3 ≈ 1.73이라 a5만 1~2번이고, 나머지는 한 번씩 들어간다."""
    objects = bl.load_train_objects(make_dataset(tmp_path))
    for seed in range(20):
        files = bl.repeat_list(objects, seed=seed)
        count = collections.Counter(files)
        assert count["a5.png"] in (1, 2) and all(count[name] == 1 for name in TOY_IMAGES if name != "a5.png")
        assert files == sorted(files)
    assert bl.repeat_list(objects, seed=3) == bl.repeat_list(objects, seed=3)


def test_repeat_list_rounds_by_probability(tmp_path):
    """threshold 1.0이면 r_30 = √7 ≈ 2.65, r_20 = √(7/3) ≈ 1.53, r_10 = √1.4 ≈ 1.18이고, 이미지는 든 클래스의 최댓값을 쓴다.
    반복 횟수는 r의 정수 부분이거나 그보다 1 많고, seed를 바꿔 가며 평균하면 r에 가깝다."""
    objects = bl.load_train_objects(make_dataset(tmp_path))
    r20, r10 = math.sqrt(7 / 3), math.sqrt(1.4)
    expected = {"k1.png": r20, "a1.png": r10, "a2.png": r10, "a3.png": r20, "a4.png": r20, "a5.png": math.sqrt(7),
                "a6.png": r10}
    counts = [collections.Counter(bl.repeat_list(objects, threshold=1.0, seed=seed)) for seed in range(2000)]
    for name, r in expected.items():
        seen = [count[name] for count in counts]
        assert set(seen) == {math.floor(r), math.floor(r) + 1}
        assert abs(sum(seen) / len(seen) - r) < 0.05


def test_undersample_keeps_kaggle_and_rare_classes(tmp_path):
    """target 2: Kaggle은 모두 넣고, 희소한 클래스부터 채운다. 30은 a5, 20은 a3(10도 함께 채움)으로 끝나서 a1, a2, a4, a6은 빠진다."""
    files = bl.undersample_list(bl.load_train_objects(make_dataset(tmp_path)), target=2)
    assert files == ["a3.png", "a5.png", "k1.png"]


def test_write_train_list(tmp_path):
    """train.txt는 절대경로를 반복 횟수만큼 적고, data.yaml은 train만 바꾼다. 개수는 반복한 줄까지 센다."""
    ds = make_dataset(tmp_path)
    info = bl.write_train_list(ds, "rfs", sorted([*TOY_IMAGES, "a5.png"]))
    out = ds / "sampling" / "rfs"
    lines = (out / "train.txt").read_text(encoding="utf-8").splitlines()
    assert lines.count(str(ds.resolve() / "images" / "train" / "a5.png")) == 2 and len(lines) == 8
    assert all(line.startswith(str(ds.resolve())) for line in lines)
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))
    assert data == {**base, "train": str(out.resolve() / "train.txt")}
    assert info == {"n_base": 7, "n_list": 8, "n_unique": 7,
                    "class_counts": {"base": {10: 5, 20: 3, 30: 1}, "list": {10: 5, 20: 3, 30: 2}}}


def test_write_train_list_without_data_yaml_and_unknown_files(tmp_path):
    """--format coco로 만든 폴더에는 train.txt만 둔다. 다른 폴더의 이미지가 든 목록은 거부한다."""
    ds = make_dataset(tmp_path, data_yaml=False)
    bl.write_train_list(ds, "under", ["k1.png"])
    assert sorted(p.name for p in (ds / "sampling" / "under").iterdir()) == ["train.txt"]
    with pytest.raises(ValueError, match="train에 없습니다"):
        bl.write_train_list(ds, "bad", ["k1.png", "x.png"])


def test_class_weights_match_ultralytics(tmp_path):
    """객체 수 [5, 3, 1](idx 순서)에서 Ultralytics compute_class_weights를 평균 1로 나눈 값과 같다."""
    ds = make_dataset(tmp_path)
    weights = bl.class_weights(ds, 0.5)
    expected = [(1 / n) ** 0.5 for n in (5, 3, 1)]
    assert weights == pytest.approx([w / (sum(expected) / 3) for w in expected])
    assert sum(weights) / 3 == pytest.approx(1.0)
    assert bl.class_weights(ds, 0) == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="0~1"):
        bl.class_weights(ds, 1.5)

    np = pytest.importorskip("numpy")
    train = pytest.importorskip("ultralytics.models.yolo.detect.train")
    trainer = SimpleNamespace(args=SimpleNamespace(cls_pw=0.5))
    ultralytics = train.DetectionTrainer.compute_class_weights(trainer, np.array([5, 3, 1], dtype=np.float32))
    assert weights == pytest.approx((ultralytics / ultralytics.mean()).tolist())


def test_class_weights_count_the_training_list(tmp_path):
    """files를 주면 목록의 반복한 줄까지 센다(샘플링 후 분포). a5를 두 번 넣으면 객체 수가 [5, 3, 2]다.
    목록에 없는 클래스가 생기거나, 학습 폴더에 없는 이미지가 든 목록은 거부한다."""
    ds = make_dataset(tmp_path)
    expected = [(1 / n) ** 0.5 for n in (5, 3, 2)]
    weights = bl.class_weights(ds, 0.5, sorted([*TOY_IMAGES, "a5.png"]))
    assert weights == pytest.approx([w / (sum(expected) / 3) for w in expected])
    assert bl.class_weights(ds, 0.5, list(TOY_IMAGES)) == pytest.approx(bl.class_weights(ds, 0.5))
    with pytest.raises(ValueError, match="객체가 없는"):
        bl.class_weights(ds, 0.5, ["a1.png"])
    with pytest.raises(ValueError, match="train에 없습니다"):
        bl.class_weights(ds, 0.5, ["x.png"])


def test_fastrcnn_class_weights(tmp_path):
    """class_weights 앞에 배경 가중치를 붙인 weighted CE다. box 손실은 그대로다. power 0이면 원래대로 돌린다.
    배경은 객체 수 [5, 3, 1]로 가중한 클래스 가중치 평균이라, 알약 RoI가 객체 수 비율대로 있으면 알약 전체와 배경의 비중이
    가중치가 없을 때와 같다."""
    torch = pytest.importorskip("torch")
    roi_heads = pytest.importorskip("torchvision.models.detection.roi_heads")
    import torch.nn.functional as F

    ds = make_dataset(tmp_path)
    original = roi_heads.fastrcnn_loss
    torch.manual_seed(0)
    logits, box_regression = torch.randn(6, 4), torch.randn(6, 16)
    labels, targets = [torch.tensor([0, 1, 2]), torch.tensor([3, 0, 1])], [torch.randn(3, 4), torch.randn(3, 4)]
    cls_loss, box_loss = original(logits, box_regression, labels, targets)
    weights = bl.class_weights(ds, 0.5)
    background = (5 * weights[0] + 3 * weights[1] + 1 * weights[2]) / 9
    roi_labels = [1] * 5 + [2] * 3 + [3] + [0] * 27   # 알약 RoI 9개(객체 수 비율), 배경 27개
    roi_weights = [([background, *weights])[label] for label in roi_labels]
    assert sum(roi_weights[:9]) / sum(roi_weights) == pytest.approx(9 / 36)
    try:
        bl.use_fastrcnn_class_weights(ds, 0.5)
        bl.use_fastrcnn_class_weights(ds, 0.5)  # 다시 불러도 원래 함수를 감싼다
        weighted, weighted_box = roi_heads.fastrcnn_loss(logits, box_regression, labels, targets)
        expected = F.cross_entropy(logits, torch.cat(labels), weight=torch.tensor([background, *weights]))
        assert torch.allclose(weighted, expected) and torch.allclose(weighted_box, box_loss)
        assert not torch.allclose(weighted, cls_loss)
        assert roi_heads.fastrcnn_loss.original is original
        assert torch.allclose(roi_heads.fastrcnn_loss.weight, torch.tensor([background, *weights]))
        with pytest.raises(ValueError, match="클래스 수"):
            roi_heads.fastrcnn_loss(torch.randn(6, 5), torch.randn(6, 20), labels, targets)
        bl.use_fastrcnn_class_weights(ds, 0)
        assert roi_heads.fastrcnn_loss is original
    finally:
        roi_heads.fastrcnn_loss = original


def write_classes_csv(path, n_train):
    """class_ap_rows가 읽는 열만 둔 classes.csv. n_train은 {category_id: v3 train 객체 수}다."""
    columns = ["category_id", "k_code", "name", "group", "n_test", "n_val", "n_train_v3"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for c, n in n_train.items():
            writer.writerow([c, f"K-{c:06d}", f"약{c}", "kaggle", 10, 5, n])
    return path


def test_class_ap_rows_sort_by_train_count(tmp_path):
    """scores.json에서 다시 읽은 글자 key도 받고, train 객체가 적은 클래스부터 놓는다."""
    classes_csv = write_classes_csv(tmp_path / "classes.csv", {1: 300, 2: 27, 3: 150})
    rows = bl.class_ap_rows({"1": 0.9, "2": 0.5, "3": 0.7}, classes_csv, "v3")
    assert [(r["category_id"], r["n_train"], r["ap"]) for r in rows] == [(2, 27, 0.5), (3, 150, 0.7), (1, 300, 0.9)]
    assert rows[0] == {"category_id": 2, "k_code": "K-000002", "name": "약2", "group": "kaggle",
                       "n_train": 27, "n_val": 5, "n_test": 10, "ap": 0.5}
    with pytest.raises(ValueError, match="방법"):
        bl.class_ap_rows({1: 0.9}, classes_csv, "v5")


def test_imbalance_summary_splits_in_half(tmp_path):
    """train 객체 수 순서로 반으로 나눈다. 홀수면 가운데 클래스는 많은 쪽에 넣는다."""
    classes_csv = write_classes_csv(tmp_path / "classes.csv", {1: 300, 2: 27, 3: 150, 4: 500})
    summary = bl.imbalance_summary({1: 0.9, 2: 0.4, 3: 0.6, 4: 1.0}, classes_csv, "v3")
    assert summary == pytest.approx({"rare_ap": 0.5, "frequent_ap": 0.95, "gap": 0.45, "n_classes": 4})
    odd = bl.imbalance_summary({1: 0.9, 2: 0.4, 3: 0.6}, classes_csv, "v3")
    assert odd == pytest.approx({"rare_ap": 0.4, "frequent_ap": 0.75, "gap": 0.35, "n_classes": 3})
    with pytest.raises(ValueError, match="둘로 나눌 수 없습니다"):
        bl.imbalance_summary({1: 0.9}, classes_csv, "v3")
