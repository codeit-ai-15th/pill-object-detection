"""3차 실험: 클래스 불균형 대응 (오버샘플링, 언더샘플링, 클래스별 손실 가중치).

방법
- 오버샘플링 (repeat_list): 모든 train 이미지를 한 번씩 넣고, 희소 클래스가 든 이미지를 더 반복한다. LVIS의 Repeat Factor Sampling이다.
- 언더샘플링 (undersample_list): 객체가 많은 클래스의 이미지를 덜 넣는다. 2차 v2를 고른 규칙(datasets.plan_uniform)과 같다.
- 클래스별 손실 가중치 (class_weights): 객체 수가 적은 클래스의 분류 손실을 키운다. Ultralytics의 cls_pw와 같은 식이다.

불균형 지표 (class_ap_rows, imbalance_summary)
- 클래스별 AP(evaluate.evaluate_val의 per_class)에 classes.csv의 train 객체 수를 붙이고, train 객체가 적은 절반과 많은 절반의 평균 AP를 비교한다.
- 2차 val 전체 점수가 1에 가까우면 개선이 드러나지 않으므로, 3차는 이 차이(gap)와 Kaggle public으로 비교한다.
- evaluate.py가 아니라 여기에 둔 이유: 2차 Faster R-CNN 노트북의 재개 서명(code_identity)에 utils.py, data.py, datasets.py,
  evaluate.py의 해시가 들어간다. 그 파일을 고치면 2차 run을 이어서 학습하거나 best로 다시 예측할 수 없다.

비교 기준: 총 학습량 맞춤
- 샘플링하면 train 목록 길이가 원래 장수 N과 달라진다. epoch 수를 round(E × N ÷ 목록 길이)로 정해서 총 step 수를 2차 기준 run과 같게 한다.
  E는 기준 run의 epoch 수다. epoch 수를 그대로 두면 오버샘플링은 더 오래, 언더샘플링은 덜 학습해서 균형 효과와 학습량 효과가 섞인다.
- write_train_list가 돌려주는 n_base(N)와 n_list(목록 길이)로 계산한다.

학습 폴더 안에 생기는 것 (write_train_list)
    <out>/sampling/<이름>/
    ├── train.txt   # train 이미지 절대경로, 한 줄에 하나. 반복하는 이미지는 그 횟수만큼 여러 줄
    └── data.yaml   # <out>/data.yaml 사본에서 train만 train.txt로 바꾼다. <out>/data.yaml이 없으면(--format coco) 만들지 않는다

모델별 연결
- YOLO26, RT-DETR (Ultralytics): data에 sampling/<이름>/data.yaml을 준다. 코드는 더 필요 없다.
    - YOLO26의 손실 가중치는 model.train(..., cls_pw=0.5)처럼 공식 인자로 준다. class_weights는 쓰지 않는다.
    - RT-DETR는 cls_pw를 넘기면 "Class weights" 로그만 찍히고 손실(RTDETRDetectionLoss)에는 반영되지 않는다. 그래서 넘기지 않는다.
    - train.txt를 쓸 때 fraction을 주지 않는다. Ultralytics는 목록을 파일명 순서로 정렬해 앞부분만 남기므로 일부 알약 조합만 학습한다.
- Faster R-CNN (torchvision): train.txt 각 줄의 파일 이름(Path(line).name)을 Dataset의 이미지 목록으로 쓴다. 중복된 줄도 그대로 둔다.
    - 손실 가중치는 학습 전에 use_fastrcnn_class_weights(class_weights(DS, 0.5))를 한 번 부른다.
    - 2차 노트북을 복사해 쓸 때는 code_identity가 해시하는 공용 파일 목록에 balance.py를 넣는다. 그래야 이 파일이 바뀐 run을 이어서 학습하지 않는다.

사용 예
    >>> from pill_detection.balance import load_train_objects, repeat_list, write_train_list
    >>> DS = "/content/exp2/v3"
    >>> info = write_train_list(DS, "rfs", repeat_list(load_train_objects(DS)))
    >>> epochs = round(20 * info["n_base"] / info["n_list"])   # 기준 run이 20 epoch일 때
    >>> YOLO("yolo26s.pt").train(data=f"{DS}/sampling/rfs/data.yaml", epochs=epochs, ...)
"""

import collections
import json
import math
import random
import statistics
from pathlib import Path

import yaml

from pill_detection.datasets import VARIANTS, plan_uniform, read_rows


def _read_train_coco(dataset_dir):
    """학습 폴더의 coco/train.json을 읽는다. categories에 모델 클래스 번호(idx)가 있는지 확인한다."""
    path = Path(dataset_dir) / "coco" / "train.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}가 없습니다. 샘플링과 클래스 가중치는 COCO 라벨에서 객체를 셉니다.\n"
                                f"[해결] 학습 폴더를 지우고 uv run scripts/build_dataset.py --variant <방법> --out {dataset_dir} "
                                "로 다시 만드세요. --format은 기본값(both)이나 coco로 둡니다.")
    coco = json.loads(path.read_text(encoding="utf-8"))
    if any("idx" not in c for c in coco["categories"]):
        raise ValueError(f"{path}의 categories에 idx(모델 클래스 번호)가 없습니다. 옛 코드로 만든 학습 폴더입니다.\n"
                         "[해결] git pull로 최신 main을 받고 학습 폴더를 다시 만드세요.")
    return coco


def load_train_objects(dataset_dir):
    """학습 폴더의 train 이미지별 출처와 객체 클래스를 읽는다.

    Args:
        dataset_dir (str | Path): build_dataset이 만든 학습 폴더. coco/train.json이 있어야 한다.

    Returns:
        dict[str, dict]: 파일명 → {"source": "kaggle" 또는 "aihub", "classes": [category_id, ...]}.
            classes는 객체 순서 그대로이고 중복을 포함한다. others는 datasets.OTHERS_CATEGORY_ID다.
    """
    coco = _read_train_coco(dataset_dir)
    name_of = {image["id"]: image["file_name"] for image in coco["images"]}
    objects = {image["file_name"]: {"source": image["source"], "classes": []} for image in coco["images"]}
    for ann in coco["annotations"]:
        objects[name_of[ann["image_id"]]]["classes"].append(ann["category_id"])
    return objects


def repeat_list(objects, threshold=None, seed=0):
    """오버샘플링: 희소 클래스가 든 이미지를 더 반복한 train 목록. (LVIS의 Repeat Factor Sampling)

    - 클래스 c의 이미지 비율 f_c = c가 든 이미지 수 ÷ train 이미지 수
    - 클래스 반복 계수 r_c = max(1, √(threshold ÷ f_c)). f_c가 threshold보다 작은 클래스만 1보다 커진다.
    - 이미지 반복 계수 r = 그 이미지에 든 클래스의 r_c 중 최댓값.
    - 이미지 반복 횟수 = r의 정수 부분 + (소수 부분의 확률로 1). 예) r = 1.3이면 30% 확률로 2번, 아니면 1번.
      평균이 r이 되므로 r이 1.5보다 작은 이미지도 반복 효과가 남는다. 모든 이미지가 한 번 이상 들어간다.
    - LVIS는 epoch마다 새로 뽑는다. 여기서는 목록 파일 하나를 모든 모델이 같이 쓰도록 seed로 한 번만 뽑는다.

    Args:
        objects (dict[str, dict]): load_train_objects()의 결과.
        threshold (float | None): 반복을 시작하는 이미지 비율. LVIS는 0.001을 썼다.
            None이면 모델 클래스 f_c의 중앙값이다. 절반의 클래스가 반복 대상이 되도록 정한 기본값이며, 값을 주면 바꿀 수 있다.
        seed (int): 소수 부분을 뽑는 난수 seed. 입력과 seed가 같으면 목록이 늘 같다.

    Returns:
        list[str]: 파일명 목록. 파일명 순서이고, 반복하는 이미지는 연달아 여러 번 나온다.
    """
    images_with = collections.Counter(c for obj in objects.values() for c in set(obj["classes"]))
    freq = {c: k / len(objects) for c, k in images_with.items()}
    if threshold is None:
        threshold = statistics.median(freq.values())
    repeat = {c: max(1.0, math.sqrt(threshold / f)) for c, f in freq.items()}
    rng = random.Random(seed)
    files = []
    for name in sorted(objects):
        r = max((repeat[c] for c in objects[name]["classes"]), default=1.0)
        files += [name] * (math.floor(r) + (rng.random() < r - math.floor(r)))
    return files


def undersample_list(objects, target):
    """언더샘플링: 클래스별 객체 수를 target개 근처까지만 채우도록 고른 train 목록.

    - 2차의 모든 방법처럼 Kaggle 이미지는 모두 넣는다.
    - AI Hub 이미지는 datasets.plan_uniform으로 고른다. 객체가 적은 클래스부터, 그 클래스가 target개가 될 때까지 그 클래스가 든 이미지를 넣는다.
    - 한 이미지에 여러 클래스가 함께 찍혀서 target을 넘는 클래스도 있다. 객체가 target보다 적은 클래스는 전부 들어간다.
    - 2차 v2는 v1 이미지에 target 200으로 이 규칙을 쓴 결과다. v2 폴더에 쓰려면 200보다 작은 target을 준다.

    Args:
        objects (dict[str, dict]): load_train_objects()의 결과.
        target (int): 클래스별 목표 객체 수.

    Returns:
        list[str]: 고른 파일명 목록. 파일명 순서이고 중복은 없다.
    """
    def as_image(name):  # plan_uniform이 읽는 이미지 정보 형식
        return {"file_name": name, "objects": [(c, None, None) for c in objects[name]["classes"]]}

    base = [as_image(name) for name in sorted(objects) if objects[name]["source"] == "kaggle"]
    pool = [as_image(name) for name in sorted(objects) if objects[name]["source"] != "kaggle"]
    chosen, _ = plan_uniform(base, pool, target)
    return sorted([image["file_name"] for image in base] + chosen)


def write_train_list(dataset_dir, name, files):
    """train 목록을 학습 폴더의 sampling/<name>/에 쓴다. (모듈 docstring의 학습 폴더 안에 생기는 것 참고)

    같은 이름으로 다시 부르면 덮어쓴다. 목록은 입력(repeat_list는 seed까지)이 같으면 늘 같아서, Colab 세션마다 다시 만들어도 된다.

    Args:
        dataset_dir (str | Path): build_dataset이 만든 학습 폴더.
        name (str): 목록 이름. 폴더 이름이 된다. (예: "rfs", "under150")
        files (list[str]): repeat_list()나 undersample_list()의 결과.

    Returns:
        dict: {"n_base": 원래 train 장수, "n_list": 목록 줄 수, "n_unique": 목록의 고유 이미지 수,
            "class_counts": {"base": {category_id: 객체 수}, "list": {category_id: 객체 수}}}.
            list는 반복한 줄까지 센 객체 수다.

    Raises:
        ValueError: 목록의 이미지가 이 학습 폴더의 train에 없을 때.
    """
    dataset_dir = Path(dataset_dir).resolve()
    objects = load_train_objects(dataset_dir)
    unknown = sorted(set(files) - set(objects))
    if unknown:
        raise ValueError(f"목록의 이미지 {len(unknown):,}장이 {dataset_dir}의 train에 없습니다. 예: {unknown[0]}\n"
                         "[해결] 목록을 만들 때 쓴 학습 폴더(같은 방법)를 주세요.")
    out = dataset_dir / "sampling" / name
    out.mkdir(parents=True, exist_ok=True)
    image_dir = dataset_dir / "images" / "train"
    (out / "train.txt").write_text("".join(f"{image_dir / f}\n" for f in files), encoding="utf-8")
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.is_file():
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        data["train"] = str(out / "train.txt")
        (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    base = collections.Counter(c for obj in objects.values() for c in obj["classes"])
    listed = collections.Counter(c for f in files for c in objects[f]["classes"])
    return {"n_base": len(objects), "n_list": len(files), "n_unique": len(set(files)),
            "class_counts": {"base": dict(sorted(base.items())), "list": dict(sorted(listed.items()))}}


def class_weights(dataset_dir, power):
    """클래스별 손실 가중치. Ultralytics cls_pw와 같은 식이다. (ultralytics/models/yolo/detect/train.py의 compute_class_weights)

    - 가중치 = (1 ÷ train 객체 수)^power를 평균이 1이 되게 나눈 값. power가 0이면 모두 1이고, 1이면 객체 수에 반비례한다.
    - 객체 수는 원래 train(coco/train.json)에서 센다. 모델 클래스는 train에 객체가 있는 클래스라서 0개인 클래스는 없다.
    - Ultralytics는 학습에 넣은 목록에서 센다. train.txt를 주면 반복한 줄까지 세므로, 같은 cls_pw여도 원래 train으로 학습할 때보다 가중치 차이가 작다.
      (v3, 0.5 기준: 원래 train 0.48~3.30, rfs 목록(seed 0) 0.51~1.91)

    Args:
        dataset_dir (str | Path): build_dataset이 만든 학습 폴더.
        power (float): 0~1. Ultralytics의 cls_pw 값과 같은 뜻이다.

    Returns:
        list[float]: 모델 클래스 번호(categories의 idx) 순서의 가중치.
    """
    if not 0 <= power <= 1:
        raise ValueError(f"power는 0~1이어야 합니다: {power}\n[해결] Ultralytics의 cls_pw와 같은 범위로 주세요.")
    coco = _read_train_coco(dataset_dir)
    count = collections.Counter(ann["category_id"] for ann in coco["annotations"])
    categories = sorted(coco["categories"], key=lambda c: c["idx"])
    empty = [c["id"] for c in categories if count[c["id"]] == 0]
    if empty:
        raise ValueError(f"train에 객체가 없는 모델 클래스가 있습니다: {empty[:5]}\n"
                         "[해결] build_dataset으로 만든 학습 폴더를 고치지 말고 그대로 쓰세요.")
    weights = [(1 / count[c["id"]]) ** power for c in categories]
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]


def class_ap_rows(per_class, classes_csv, variant):
    """클래스별 AP에 그 클래스의 객체 수를 붙인다. train 객체가 적은 클래스부터 놓는다.

    Args:
        per_class (dict): evaluate.evaluate_val()의 "per_class". scores.json에서 다시 읽어 key가 글자("3351")여도 된다.
        classes_csv (str | Path): data/datasets/classes.csv.
        variant (str): 그 run의 방법. (v1~v4) train 객체 수를 n_train_<방법> 열에서 읽는다.

    Returns:
        list[dict]: {"category_id", "k_code", "name", "group", "n_train", "n_val", "n_test", "ap"} 목록.
            n_train 순서이고, 같으면 category_id 순서다.
    """
    if variant not in VARIANTS:
        raise ValueError(f"방법이 맞지 않습니다: {variant}\n[해결] {', '.join(VARIANTS)} 중 그 run을 학습한 방법을 주세요.")
    table = {int(row["category_id"]): row for row in read_rows(classes_csv)}
    rows = []
    for key, ap in per_class.items():
        row = table[int(key)]
        rows.append({"category_id": int(key), "k_code": row["k_code"], "name": row["name"], "group": row["group"],
                     "n_train": int(row[f"n_train_{variant}"]), "n_val": int(row["n_val"]), "n_test": int(row["n_test"]),
                     "ap": ap})
    return sorted(rows, key=lambda r: (r["n_train"], r["category_id"]))


def imbalance_summary(per_class, classes_csv, variant):
    """train 객체가 적은 클래스와 많은 클래스의 평균 AP. 3차 실험의 비교 지표이자 모델 선정 근거다.

    클래스별 AP가 있는 클래스(val에 정답이 있는 대회 클래스)를 train 객체 수 순서로 놓고 반으로 나눈다.
    홀수면 가운데 클래스는 많은 쪽에 넣는다.

    Args:
        per_class (dict): evaluate.evaluate_val()의 "per_class". (class_ap_rows와 같다)
        classes_csv (str | Path): data/datasets/classes.csv.
        variant (str): 그 run의 방법. (v1~v4)

    Returns:
        dict: {"rare_ap": 적은 쪽 평균 AP, "frequent_ap": 많은 쪽 평균 AP, "gap": frequent_ap - rare_ap,
            "n_classes": 나눈 클래스 수}. W&B Summary에 "exp3/<key>"로 그대로 붙일 수 있다.
    """
    rows = class_ap_rows(per_class, classes_csv, variant)
    if len(rows) < 2:
        raise ValueError(f"클래스별 AP가 {len(rows)}개라 둘로 나눌 수 없습니다.\n"
                         "[해결] evaluate_val()이 2차 val 전체로 낸 per_class를 주세요.")
    half = len(rows) // 2
    rare = sum(r["ap"] for r in rows[:half]) / half
    frequent = sum(r["ap"] for r in rows[half:]) / (len(rows) - half)
    return {"rare_ap": rare, "frequent_ap": frequent, "gap": frequent - rare, "n_classes": len(rows)}


def use_fastrcnn_class_weights(weights):
    """torchvision Faster R-CNN의 RoI 분류 손실에 클래스 가중치를 건다. 학습 전에 한 번 부른다.

    - torchvision.models.detection.roi_heads.fastrcnn_loss를 바꿔 끼운다. 같은 커널의 모든 Faster R-CNN에 적용되므로,
      가중치 없이 학습하거나 예측만 할 때는 None을 넣어 원래대로 돌린다.
    - 분류 손실만 F.cross_entropy(weight=[1.0, *weights])로 바뀐다. 0번은 배경이라 1.0이다.
      box 손실과 RPN 손실(클래스를 구분하지 않음)은 그대로다.
    - PyTorch는 가중치를 곱한 손실을 가중치 합으로 나눠 평균을 낸다. 그래서 가중치는 RoI끼리의 상대적인 비중만 바꾼다.

    Args:
        weights (list[float] | None): class_weights()의 결과(모델 클래스 번호 순서). None이면 원래 손실로 돌린다.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.models.detection import roi_heads

    original = getattr(roi_heads.fastrcnn_loss, "original", roi_heads.fastrcnn_loss)
    if weights is None:
        roi_heads.fastrcnn_loss = original
        return
    weight = torch.tensor([1.0, *weights])

    def weighted_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        if class_logits.shape[1] != len(weight):
            raise ValueError(f"가중치 수({len(weight) - 1})와 모델 클래스 수({class_logits.shape[1] - 1})가 다릅니다.\n"
                             "[해결] 모델을 만든 학습 폴더로 class_weights를 다시 계산하세요.")
        _, box_loss = original(class_logits, box_regression, labels, regression_targets)
        loss = F.cross_entropy(class_logits, torch.cat(labels, dim=0), weight=weight.to(class_logits.device))
        return loss, box_loss

    weighted_fastrcnn_loss.original = original
    roi_heads.fastrcnn_loss = weighted_fastrcnn_loss
