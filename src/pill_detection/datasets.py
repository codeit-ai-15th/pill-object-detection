"""2차 실험 데이터: 이미지 선택 방법 4가지의 train 목록, 고정 val, 학습 폴더 만들기.

용어
- 대회 클래스: Kaggle train 클래스 + test에 나오는 클래스. 채점과 제출은 이 클래스에 대해서만 한다.
- 2차 val: 이미지 선택 4가지 방법(v1, v2, v3, v4. 바로 밑 방법(VARIANTS)에서 자세히 기술되어있음)이 같이 쓰는 검증 이미지. 1차 실험의 val에서 사용한 Kaggle 47장 + 등장 클래스 빈도가 test 클래스 빈도와 비슷하게 고른 AI Hub 출처 데이터의 조합이다.
  train에는 넣지 않는다. 
- 2차 train: val을 뺀 나머지에서 방법 규칙으로 고른 이미지. Kaggle 1차 train 185장은 4가지 방법에 모두 들어간다.

방법 (VARIANTS)
- v1: val을 뺀 이미지를 전부 사용함 (Kaggle 1차 train + AI Hub)
- v2: 객체가 적은 클래스부터 클래스별 200개를 채우도록 고른 AI Hub 출처 이미지 (plan_uniform)
- v3: 대회 클래스가 하나라도 든 이미지
- v4: v1과 같은 이미지를 사용함. 단, 대회 클래스가 아닌 알약은 others 한 클래스로 묶는다 (test에만 나오는 클래스는 묶지 않는다)

흐름
- train/val 구성 결정 (scripts/make_dataset_lists.py, 1회): load_images → select_val → select_variants → make_image_rows, build_class_table
  → data/datasets/images.csv, classes.csv 생성. (git으로 관리한다. 열과 줄 설명: data/datasets/README.md)
- train/val 만들기 (scripts/build_dataset.py, 학습할 때마다): images.csv, classes.csv 두 CSV와 정제한 학습 데이터로 학습 폴더를 만든다.

학습 폴더 (build_dataset)
    <out>/
    ├── images/{train,val}/      # 이미지 하드링크 (안 되면 복사). --format과 상관없이 늘 둔다
    ├── labels/{train,val}/      # YOLO 라벨. 모델 클래스 번호와 [중심 x, 중심 y, 너비, 높이] (0~1)
    ├── data.yaml                # Ultralytics 설정. names는 {번호: "3351", ..., 번호: "others"}
    ├── coco/{train,val}.json    # COCO 형식. file_name은 images/{train,val}/ 기준. category id는 K-코드 숫자,
    │                            #   others는 OTHERS_CATEGORY_ID. categories의 idx는 0부터의 모델 클래스 번호(YOLO와 같다)
    ├── eval/val_gt.json         # 채점용 val 정답 (pill_detection.evaluate). 모든 방법에서 같다
    ├── classes.csv              # data/datasets/classes.csv 사본
    └── dataset_info.json        # 방법, 개수, 목록 CSV의 sha256, git commit hash
- YOLO 라벨과 coco/*.json에는 그 방법의 모델 클래스만 들어간다. 모델 클래스는 그 방법(v1, v2, v3, v4)의 train에 객체가 있는 클래스다.
  v3의 val에만 있는 클래스처럼 모델 클래스가 아닌 객체는 빼고, 뺀 수를 dataset_info.json에 적는다.
- eval/val_gt.json은 val의 모든 객체를 원래 K-코드로 담는다. 채점은 대회 클래스만 하고, 나머지 객체는 무시 영역으로 쓴다.
"""

import collections
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from pill_detection.data import combo_name, load_annotations
from pill_detection.utils import clip_xywh, xywh_to_yolo

VARIANTS = ("v1", "v2", "v3", "v4")
OTHERS_VARIANTS = ("v4",)  # 대회 클래스 밖 알약을 others로 묶는 방법
OTHERS_CATEGORY_ID = 1_000_000  # others의 category id. K-코드와 겹치지 않는다. 학습 label로 쓰는 번호는 COCO categories의 idx다
OTHERS_NAME = "others"

V2_TARGET = 200  # v2의 클래스별 목표 객체 수 (01 노트북 4-5)
VAL_AIHUB_SHARE = 0.10  # val의 AI Hub 몫: AI Hub 대회 클래스 객체의 이 비율만큼을 클래스별 목표로 나눈다
VAL_CLASS_CAP = 0.20  # 클래스마다 AI Hub 객체의 이 비율까지만 val에 넣는다. train에 쓸 객체가 모자라지 않게 한다
VAL_CLASS_FLOOR = 20  # 대회 클래스마다 val 목표 객체 수의 최솟값 (상한이 더 작으면 상한)
VAL_SEED = 410  # 같은 조건의 조합 중 무엇을 먼저 고를지 정하는 seed. 1차 split.csv와 같은 값

# 00 5-3 기록표의 test 전용 알약 18종. 01 노트북 5-2에서 같은 알약으로 확인한 AI Hub 클래스다
# 노트북 02(test 알약 crop 검수)의 결과가 나오기 전까지 대회 클래스를 정할 때 쓴다
TEST_ONLY_CLASSES = (4378, 5094, 5886, 6192, 6563, 10221, 10224, 12420, 18110,
                     21026, 22627, 23203, 23223, 27653, 29871, 31705, 33878, 44199)

# data 폴더 기준 학습 데이터 경로 (scripts/prepare_train_data.py가 만든다)
ANN_DIRS = {"kaggle": "processed/train_annotations", "aihub": "processed/ai_hub/train_annotations"}
IMAGE_DIRS = {"kaggle": "raw/train_images", "aihub": "processed/ai_hub/train_images"}
SOURCES = ("kaggle", "aihub")

IMAGE_COLUMNS = ["source", "file_name", "combo", "split", "exp1_split", *VARIANTS,
                 "n_objects", "n_competition", "n_noncompetition"]
CLASS_COLUMNS = ["category_id", "k_code", "name", "group", "competition", *(f"idx_{v}" for v in VARIANTS),
                 "n_test", "n_val", *(f"n_train_{v}" for v in VARIANTS)]
GROUPS = ("kaggle", "test_only", "aihub_only", OTHERS_NAME)


def load_images(data_dir):
    """정제한 학습 데이터(Kaggle 사본 + AI Hub 사본)를 이미지 단위로 읽는다.

    Args:
        data_dir (str | Path): data 폴더. 아래에 ANN_DIRS의 annotation 폴더가 있어야 한다.

    Returns:
        tuple[dict[str, dict], dict[int, str]]: (이미지 파일명 → 이미지 정보, category_id → 제품명).
            이미지 정보는 아래와 같다. 파일명 순서로 들어 있다.

            {
                "source": "aihub",                 # kaggle 또는 aihub
                "file_name": "K-000250-000573-002483-006192_0_2_0_2_70_000_200.png",  # AI Hub는 사본 이름
                "combo": "K-000250-000573-002483-006192",
                "image_id": 1501, "width": 976, "height": 1280,
                "objects": [(250, [x, y, w, h], ann_id), ...],  # (category_id, bbox 원본 픽셀, annotation id)
            }

    Raises:
        FileNotFoundError: annotation 폴더가 없을 때.
        ValueError: Kaggle과 AI Hub에 같은 파일명이 있을 때.
    """
    data_dir = Path(data_dir)
    images, names = {}, {}
    for source in SOURCES:
        ann_dir = data_dir / ANN_DIRS[source]
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"학습 데이터 annotation 폴더가 없습니다: {ann_dir}\n"
                                    "[해결] uv run scripts/prepare_train_data.py로 만들거나, Drive의 zip을 레포 최상위에 푸세요.")
        for record in load_annotations(ann_dir).values():
            name = record["file_name"]
            image = images.get(name)
            if image is None:
                image = images[name] = {"source": source, "file_name": name, "combo": combo_name(name),
                                        "image_id": record["image_id"], "width": record["width"],
                                        "height": record["height"], "objects": []}
            elif image["source"] != source:
                raise ValueError(f"Kaggle과 AI Hub에 같은 파일명이 있습니다: {name}\n"
                                 "[해결] 학습 데이터를 다시 만드세요: uv run scripts/prepare_train_data.py")
            image["objects"].append((record["category_id"], record["bbox"], record["ann_id"]))
            names.setdefault(record["category_id"], record["category_name"].strip())
    return dict(sorted(images.items())), names


def object_classes(image):
    """이미지 안 알약의 category_id 목록 (객체 순서 그대로, 중복 포함)."""
    return [c for c, _, _ in image["objects"]]


def kaggle_classes(images):
    """Kaggle train에 있는 클래스."""
    return {c for image in images.values() if image["source"] == "kaggle" for c in object_classes(image)}


def competition_classes(images, test_classes):
    """대회 클래스를 정한다: Kaggle train 클래스 + test에 나오는 클래스.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        test_classes (Iterable[int]): test에 나오는 클래스. 노트북 02 검수표(read_test_counts)나 TEST_ONLY_CLASSES.

    Returns:
        set[int]: 대회 클래스의 category_id.
    """
    return kaggle_classes(images) | set(test_classes)


def class_group(category_id, kaggle, competition):
    """클래스 그룹: kaggle(Kaggle train에 있음), test_only(Kaggle train에 없고 test에 나옴), aihub_only(나머지), others."""
    if category_id == OTHERS_CATEGORY_ID:
        return OTHERS_NAME
    if category_id in kaggle:
        return "kaggle"
    return "test_only" if category_id in competition else "aihub_only"


def variant_category(category_id, variant, competition):
    """방법에 따른 학습 라벨의 category_id. others로 묶는 방법이면 대회 클래스 밖 알약을 OTHERS_CATEGORY_ID로 바꾼다."""
    if variant in OTHERS_VARIANTS and category_id not in competition:
        return OTHERS_CATEGORY_ID
    return category_id


def read_exp1_split(csv_path, images=None):
    """1차 실험의 Kaggle train/val 분할(Drive의 split.csv)을 읽는다.

    Args:
        csv_path (str | Path): file_name, split 열이 있는 CSV. split은 train 또는 val이다.
        images (dict[str, dict] | None): load_images()의 이미지 정보. 주면 Kaggle 이미지와 같은 목록인지 확인한다.

    Returns:
        dict[str, str]: Kaggle 이미지 파일명 → "train" 또는 "val".

    Raises:
        ValueError: 열이나 값이 맞지 않거나, Kaggle 이미지와 목록이 다르거나, 한 조합이 train과 val에 나뉘어 있을 때.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    hint = "[해결] 1차 실험에 쓴 Drive의 split.csv(file_name, combo, split 열)를 그대로 받아 경로를 주세요."
    if not rows or not {"file_name", "split"} <= set(rows[0]):
        raise ValueError(f"{csv_path}: file_name, split 열이 없습니다.\n{hint}")
    split = {row["file_name"]: row["split"] for row in rows}
    bad = sorted({s for s in split.values() if s not in ("train", "val")})
    if bad or len(split) != len(rows):
        raise ValueError(f"{csv_path}: split 값이 train, val이 아니거나({bad}) 파일명이 겹칩니다.\n{hint}")
    by_combo = collections.defaultdict(set)
    for name, s in split.items():
        by_combo[combo_name(name)].add(s)
    mixed = sorted(c for c, s in by_combo.items() if len(s) > 1)
    if mixed:
        raise ValueError(f"{csv_path}: 한 조합이 train과 val에 나뉘어 있습니다: {', '.join(mixed[:3])}\n{hint}")
    if images is not None:
        kaggle = {name for name, image in images.items() if image["source"] == "kaggle"}
        if set(split) != kaggle:
            raise ValueError(f"{csv_path}: Kaggle 이미지 목록과 다릅니다. "
                             f"CSV에만 있음 {len(set(split) - kaggle)}장, Kaggle에만 있음 {len(kaggle - set(split))}장\n{hint}")
    return split


def read_test_counts(csv_path):
    """노트북 02의 test 알약 검수표에서 클래스별 test 객체 수를 센다.

    Args:
        csv_path (str | Path): notebooks/eda/02_raewoo_test_crops.csv. final_category_id 열을 읽는다.
            빈칸(학습 데이터 어디에도 없는 알약)은 세지 않는다.

    Returns:
        collections.Counter: category_id → test 객체 수.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return collections.Counter(int(row["final_category_id"]) for row in csv.DictReader(f)
                                   if row["final_category_id"].strip())


def plan_uniform(base, pool, target=V2_TARGET, sort_key=None):
    """객체가 적은 클래스부터 목표 객체 수를 채우도록 pool에서 이미지를 고른다. (01 노트북 4-5)

    - base 이미지는 모두 넣은 것으로 보고 센다.
    - 전체(base + pool) 객체가 적은 클래스부터, 그 클래스가 목표를 채울 때까지 그 클래스가 있는 이미지를 넣는다.
    - 한 클래스 안에서는 이미 목표를 채운 클래스가 적게 함께 찍힌 이미지부터 넣는다. 같으면 sort_key 순서다.

    Args:
        base (list[dict]): 늘 넣는 이미지. (예: Kaggle train) load_images()의 이미지 정보.
        pool (list[dict]): 고를 후보 이미지. (예: AI Hub)
        target (int): 클래스별 목표 객체 수.
        sort_key (Callable[[str], str] | None): 같은 조건인 이미지의 순서를 정하는 파일명 → 정렬 값.
            None이면 파일명이다. 01 노트북 결과를 그대로 재현할 때는 사본 이름 → 원래 이름을 넣는다.

    Returns:
        tuple[list[str], collections.Counter]: (고른 pool 이미지 파일명 (정렬), base + 고른 이미지의 클래스별 객체 수).
    """
    sort_key = sort_key or (lambda name: name)
    classes_of = {image["file_name"]: object_classes(image) for image in pool}
    count = collections.Counter(c for image in base for c in object_classes(image))
    total = count + collections.Counter(c for classes in classes_of.values() for c in classes)
    by_class = collections.defaultdict(list)  # 클래스 → 그 클래스가 있는 pool 이미지
    for name in sorted(classes_of, key=sort_key):
        for c in set(classes_of[name]):
            by_class[c].append(name)
    chosen = set()
    for c in sorted(total, key=lambda c: (total[c], c)):
        candidates = sorted((name for name in by_class[c] if name not in chosen),
                            key=lambda name: (sum(count[o] >= target for o in classes_of[name]), sort_key(name)))
        for name in candidates:
            if count[c] >= target:
                break
            chosen.add(name)
            count.update(classes_of[name])
    return sorted(chosen), count


def _tiebreak(seed, combo):
    """같은 조건의 조합 순서. seed와 조합 이름만으로 정해서 입력 순서와 상관없다."""
    return hashlib.sha256(f"{seed}:{combo}".encode()).hexdigest()


def select_val(images, competition, test_counts, exp1_split, share=VAL_AIHUB_SHARE, cap=VAL_CLASS_CAP,
               floor=VAL_CLASS_FLOOR, seed=VAL_SEED):
    """2차 val을 고른다. 1차 val Kaggle 이미지 + test 클래스 빈도를 닮게 고른 AI Hub 조합.

    규칙
    1. 1차 val Kaggle 이미지를 모두 넣는다. 1차 train은 넣지 않는다.
    2. AI Hub 몫의 크기는 AI Hub 대회 클래스 객체 수 × share이다.
    3. AI Hub에 있는 대회 클래스마다 목표 val 객체 수를 정한다. AI Hub 몫을 test 빈도 비율로 나누고,
       floor 이상, 그 클래스 AI Hub 객체 수 × cap(상한) 이하로 맞춘다. 상한이 floor보다 작으면 상한이다.
    4. AI Hub 객체가 적은 대회 클래스부터, 목표에 닿을 때까지 그 클래스가 든 AI Hub 조합을 통째로 넣는다.
       - 대회 클래스 밖 알약이 적게 섞인 조합부터, 같으면 이미 목표를 채운 대회 클래스 객체가 적은 조합부터,
         같으면 _tiebreak(seed, 조합) 순서다.
       - 조합을 넣으면 어떤 클래스든 val의 AI Hub 객체가 상한을 넘는 조합은 건너뛴다. (대회 클래스 밖 클래스도 상한을 지킨다)
    5. AI Hub에 없는 대회 클래스(K-003351, K-003483)는 1차 val Kaggle 이미지로만 채운다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        competition (set[int]): 대회 클래스.
        test_counts (dict[int, int]): 클래스별 test 객체 수. (read_test_counts) 없는 클래스는 0으로 본다
        exp1_split (dict[str, str]): read_exp1_split()의 결과.
        share (float): AI Hub 몫의 비율.
        cap (float): 클래스별 상한 비율.
        floor (int): 대회 클래스의 목표 최솟값.
        seed (int): 같은 조건인 조합의 순서를 정하는 값.

    Returns:
        dict: {
            "files": set[str] (val 이미지 파일명),
            "wanted": {category_id: 상한을 적용하기 전 목표 val 객체 수} (AI Hub에 있는 대회 클래스),
            "target": {category_id: 목표 val 객체 수} (wanted에 상한을 적용한 값),
            "cap": {category_id: val에 넣을 수 있는 AI Hub 객체 수} (AI Hub에 있는 모든 클래스),
            "count": collections.Counter (val 객체 수, Kaggle + AI Hub),
            "aihub_count": collections.Counter (val의 AI Hub 객체 수),
            "capped": set[int] (상한 때문에 조합을 건너뛴 적이 있는 클래스),
            "budget": float (AI Hub 몫의 객체 수),
        }

    Raises:
        ValueError: AI Hub에 있는 대회 클래스의 test 객체 수 합이 0일 때.
    """
    test_counts = collections.Counter(test_counts)
    files = {name for name, s in exp1_split.items() if s == "val"}
    count = collections.Counter(c for name in files for c in object_classes(images[name]))

    combos = collections.defaultdict(list)  # AI Hub 조합 → 이미지 파일명
    for name, image in images.items():
        if image["source"] == "aihub":
            combos[image["combo"]].append(name)
    combo_objects = {combo: collections.Counter(c for name in names for c in object_classes(images[name]))
                     for combo, names in combos.items()}
    aihub_objects = sum(combo_objects.values(), collections.Counter())
    in_aihub = sorted(c for c in competition if aihub_objects[c])
    budget = share * sum(aihub_objects[c] for c in in_aihub)
    test_total = sum(test_counts[c] for c in in_aihub)
    if not test_total:
        raise ValueError("AI Hub에 있는 대회 클래스의 test 객체 수가 모두 0입니다.\n"
                         "[해결] 노트북 02의 검수표(final_category_id)를 확인하세요.")
    caps = {c: math.floor(cap * n) for c, n in aihub_objects.items()}
    wanted = {c: max(round(budget * test_counts[c] / test_total), floor) for c in in_aihub}  # 상한을 적용하기 전 목표
    target = {c: min(wanted[c], caps[c]) for c in in_aihub}
    noncompetition = {combo: sum(n for c, n in objects.items() if c not in competition)
                      for combo, objects in combo_objects.items()}

    aihub_count, chosen, capped = collections.Counter(), set(), set()
    for c in sorted(in_aihub, key=lambda c: (aihub_objects[c], c)):
        candidates = sorted(
            (combo for combo, objects in combo_objects.items() if objects[c] and combo not in chosen),
            key=lambda combo: (noncompetition[combo],
                               sum(n for o, n in combo_objects[combo].items() if o in target and count[o] >= target[o]),
                               _tiebreak(seed, combo)))
        for combo in candidates:
            if count[c] >= target[c]:
                break
            over = [o for o, n in combo_objects[combo].items() if aihub_count[o] + n > caps[o]]
            if over:
                capped.update(over)
                continue
            chosen.add(combo)
            files.update(combos[combo])
            count.update(combo_objects[combo])
            aihub_count.update(combo_objects[combo])
    return {"files": files, "wanted": wanted, "target": target, "cap": caps, "count": count, "aihub_count": aihub_count,
            "capped": capped, "budget": budget}


def select_variants(images, val_files, competition, exp1_split, target=V2_TARGET):
    """val을 뺀 나머지에서 방법별 2차 train을 고른다.

    - Kaggle 1차 train은 4가지 방법에 모두 넣는다. (Kaggle 이미지는 모두 대회 클래스라 v3에도 들어간다)
    - v2는 Kaggle 1차 train을 base로, val 밖 AI Hub 이미지에서 plan_uniform으로 고른다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        val_files (set[str]): select_val()의 val 이미지.
        competition (set[int]): 대회 클래스.
        exp1_split (dict[str, str]): read_exp1_split()의 결과.
        target (int): v2의 클래스별 목표 객체 수.

    Returns:
        dict[str, set[str]]: 방법 → 2차 train 이미지 파일명.

    Raises:
        ValueError: Kaggle 1차 train 이미지가 val에 들어 있을 때.
    """
    kaggle_train = {name for name, s in exp1_split.items() if s == "train"}
    if kaggle_train & val_files:
        raise ValueError("Kaggle 1차 train 이미지가 2차 val에 들어 있습니다. 1차 비교가 깨집니다.\n"
                         "[해결] select_val에 넣은 exp1_split을 확인하세요.")
    train = [image for name, image in images.items() if name not in val_files]
    aihub = [image for image in train if image["source"] == "aihub"]
    v2, _ = plan_uniform([images[name] for name in sorted(kaggle_train)], aihub, target)
    return {
        "v1": {image["file_name"] for image in train},
        "v2": kaggle_train | set(v2),
        "v3": {image["file_name"] for image in train if set(object_classes(image)) & competition},
        "v4": {image["file_name"] for image in train},
    }


def object_counts(images, files, variant=None, competition=None):
    """고른 이미지의 클래스별 객체 수. variant를 주면 그 방법의 학습 라벨(variant_category)로 센다."""
    count = collections.Counter()
    for name in files:
        for c in object_classes(images[name]):
            count[variant_category(c, variant, competition) if variant else c] += 1
    return count


def model_classes(images, train_files, variant, competition):
    """방법의 모델 클래스: 그 방법의 2차 train에 객체가 있는 클래스(학습 라벨 기준). 번호 순서는 category_id 순서다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        train_files (set[str]): 그 방법의 2차 train 이미지.
        variant (str): 방법. (v1~v4)
        competition (set[int]): 대회 클래스.

    Returns:
        list[int]: 모델 클래스 번호 순서의 category_id. others(OTHERS_CATEGORY_ID)는 가장 크므로 맨 뒤다.
    """
    return sorted(object_counts(images, train_files, variant, competition))


def make_image_rows(images, val_files, variants, exp1_split, competition):
    """images.csv의 줄을 만든다. Kaggle 다음 AI Hub, 같은 출처 안에서는 파일명 순서다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        val_files (set[str]): 2차 val 이미지.
        variants (dict[str, set[str]]): select_variants()의 결과.
        exp1_split (dict[str, str]): read_exp1_split()의 결과.
        competition (set[int]): 대회 클래스.

    Returns:
        list[dict]: IMAGE_COLUMNS 열의 줄.
    """
    rows = []
    for name, image in sorted(images.items(), key=lambda kv: (SOURCES.index(kv[1]["source"]), kv[0])):
        classes = object_classes(image)
        n_competition = sum(c in competition for c in classes)
        val = name in val_files
        rows.append({
            "source": image["source"], "file_name": name, "combo": image["combo"],
            "split": "val" if val else "train", "exp1_split": exp1_split.get(name, ""),
            **{v: int(val or name in variants[v]) for v in VARIANTS},
            "n_objects": len(classes), "n_competition": n_competition, "n_noncompetition": len(classes) - n_competition,
        })
    return rows


def build_class_table(images, names, val_files, variants, competition, test_counts):
    """classes.csv의 줄을 만든다. 학습 데이터나 test에 나오는 클래스 + others 한 줄, category_id 순서다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        names (dict[int, str]): category_id → 제품명. (load_images)
        val_files (set[str]): 2차 val 이미지.
        variants (dict[str, set[str]]): select_variants()의 결과.
        competition (set[int]): 대회 클래스.
        test_counts (dict[int, int]): 클래스별 test 객체 수. 없는 클래스는 0으로 본다

    Returns:
        list[dict]: CLASS_COLUMNS 열의 줄. idx_vN은 그 방법의 모델 클래스 번호이고, 쓰지 않는 클래스는 빈칸이다.
            n_train_vN은 그 방법의 학습 라벨 기준이다. (v4에서 대회 클래스 밖 클래스는 0이고 others 줄에 합쳐진다)
    """
    test_counts = collections.Counter(test_counts)
    kaggle = kaggle_classes(images)
    n_val = object_counts(images, val_files)
    n_val[OTHERS_CATEGORY_ID] = sum(n for c, n in n_val.items() if c not in competition)
    idx = {v: {c: i for i, c in enumerate(model_classes(images, variants[v], v, competition))} for v in VARIANTS}
    n_train = {v: object_counts(images, variants[v], v, competition) for v in VARIANTS}
    rows = []
    for c in sorted(set(names) | set(test_counts) | {OTHERS_CATEGORY_ID}):
        others = c == OTHERS_CATEGORY_ID
        rows.append({
            "category_id": c, "k_code": OTHERS_NAME if others else f"K-{c:06d}",
            "name": OTHERS_NAME if others else names.get(c, ""),
            "group": class_group(c, kaggle, competition), "competition": int(c in competition),
            **{f"idx_{v}": idx[v].get(c, "") for v in VARIANTS},
            "n_test": "" if others else test_counts[c], "n_val": n_val[c],
            **{f"n_train_{v}": "" if others and v not in OTHERS_VARIANTS else n_train[v][c] for v in VARIANTS},
        })
    return rows


def make_lists(images, names, exp1_split, test_counts):
    """결정 단계 전체: 대회 클래스, 2차 val, 방법별 train, images.csv와 classes.csv의 줄을 만든다.

    Args:
        images (dict[str, dict]): load_images()의 이미지 정보.
        names (dict[int, str]): category_id → 제품명. (load_images)
        exp1_split (dict[str, str]): read_exp1_split()의 결과.
        test_counts (dict[int, int]): 클래스별 test 객체 수. (read_test_counts)

    Returns:
        dict: {"competition": set[int], "val": select_val()의 결과, "variants": select_variants()의 결과,
               "image_rows": images.csv 줄, "class_rows": classes.csv 줄}
    """
    competition = competition_classes(images, (c for c, n in test_counts.items() if n))
    val = select_val(images, competition, test_counts, exp1_split)
    variants = select_variants(images, val["files"], competition, exp1_split)
    return {"competition": competition, "val": val, "variants": variants,
            "image_rows": make_image_rows(images, val["files"], variants, exp1_split, competition),
            "class_rows": build_class_table(images, names, val["files"], variants, competition, test_counts)}


def exp1_split_from_rows(image_rows):
    """images.csv의 exp1_split 열에서 1차 분할을 되살린다. (Kaggle 줄만)"""
    return {row["file_name"]: row["exp1_split"] for row in image_rows if row["exp1_split"]}


def rows_to_text(columns, rows):
    """CSV 글자를 만든다. 줄바꿈은 \\n이다."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def diff_rows(path, columns, rows):
    """CSV 파일과 새 줄을 비교한다. 파일은 쓰지 않는다. (make_dataset_lists.py --check)

    Args:
        path (str | Path): 비교할 CSV 경로. 없으면 빈 파일로 본다.
        columns (list[str]): 열 이름.
        rows (list[dict]): 새 줄.

    Returns:
        dict | None: 같으면 None. 다르면 {"added": 새 내용에만 있는 줄 수, "removed": 파일에만 있는 줄 수}.
            열 이름 줄도 센다.
    """
    path = Path(path)
    new = rows_to_text(columns, rows)
    old = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.is_file() else ""
    if old == new:
        return None
    old_lines, new_lines = collections.Counter(old.splitlines()), collections.Counter(new.splitlines())
    return {"added": sum((new_lines - old_lines).values()), "removed": sum((old_lines - new_lines).values())}


def write_rows(path, columns, rows):
    """줄을 CSV로 쓴다. 이미 같은 내용이면 쓰지 않는다.

    Args:
        path (str | Path): CSV 경로. 상위 폴더가 없으면 만든다.
        columns (list[str]): 열 이름.
        rows (list[dict]): 줄.

    Returns:
        bool: 새로 쓰거나 내용이 바뀌었으면 True, 같아서 그대로 두었으면 False.
    """
    path = Path(path)
    text = rows_to_text(columns, rows)
    if diff_rows(path, columns, rows) is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return True


def read_rows(path):
    """CSV를 dict 줄 목록으로 읽는다. 값은 모두 글자다."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_image_list(images_csv, variant):
    """images.csv에서 방법의 train과 val 이미지를 고른다.

    Args:
        images_csv (str | Path): data/datasets/images.csv.
        variant (str): 방법. (v1~v4)

    Returns:
        dict[str, list[str]]: {"train": 파일명 목록, "val": 파일명 목록}. CSV 순서다.
    """
    lists = {"train": [], "val": []}
    for row in read_rows(images_csv):
        if row[variant] == "1":
            lists[row["split"]].append(row["file_name"])
    return lists


def read_class_table(classes_csv, variant):
    """classes.csv에서 방법의 모델 클래스와 대회 클래스를 읽는다.

    Args:
        classes_csv (str | Path): data/datasets/classes.csv.
        variant (str): 방법. (v1~v4)

    Returns:
        dict: {"classes": [모델 클래스 번호 순서의 category_id], "competition": set[int], "rows": {category_id: 줄}}
    """
    rows = {int(row["category_id"]): row for row in read_rows(classes_csv)}
    numbered = sorted((int(row[f"idx_{variant}"]), c) for c, row in rows.items() if row[f"idx_{variant}"] != "")
    classes = [c for _, c in numbered]
    if [i for i, _ in numbered] != list(range(len(numbered))):
        raise ValueError(f"{classes_csv}: idx_{variant}가 0부터 빈틈없이 매겨져 있지 않습니다.\n"
                         "[해결] scripts/make_dataset_lists.py가 만든 classes.csv를 고치지 말고 그대로 쓰세요.")
    competition = {c for c, row in rows.items() if row["competition"] == "1"}
    return {"classes": classes, "competition": competition, "rows": rows}


def category_label(category_id):
    """모델 클래스 이름. data.yaml의 names와 COCO가 아닌 곳에서 쓴다. (3351 → "3351", others → "others")"""
    return OTHERS_NAME if category_id == OTHERS_CATEGORY_ID else str(category_id)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    """지금 레포의 commit. git이 없거나 레포가 아니면 None."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_by_git(path):
    """path가 이 레포 안이고 .gitignore에 걸리지 않으면 True. 레포 밖이거나 git이 없으면 False."""
    repo_root = Path(__file__).resolve().parents[2]  # src/pill_detection/datasets.py → 레포 최상위
    if not Path(path).resolve().is_relative_to(repo_root):
        return False
    try:  # check-ignore: 0이면 무시됨, 1이면 무시되지 않음, 128이면 git 레포가 아님
        result = subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True, cwd=repo_root)
    except OSError:
        return False
    return result.returncode == 1


def _place(src, dst, link):
    """이미지를 dst에 둔다. hardlink가 안 되면(다른 디스크) 복사한다. 실제로 쓴 방법을 돌려준다."""
    if link == "symlink":
        os.symlink(Path(src).resolve(), dst)
        return "symlink"
    if link == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _coco_category(category_id, row):
    """COCO categories 항목. classes.csv의 줄에서 만든다."""
    return {"id": category_id, "name": category_label(category_id), "supercategory": "pill",
            "k_code": row["k_code"], "product_name": row["name"], "group": row["group"],
            "competition": int(row["competition"]), "test_only": int(row["group"] == "test_only")}


def _coco(images, files, categories, label_of):
    """COCO dict를 만든다. label_of(category_id)가 None인 객체는 뺀다. bbox는 이미지 안으로 자르고 너비나 높이가 0이면 뺀다."""
    coco = {"images": [], "annotations": [], "categories": categories}
    dropped = clipped_out = 0
    for name in files:
        image = images[name]
        coco["images"].append({"id": image["image_id"], "file_name": name, "width": image["width"],
                               "height": image["height"], "source": image["source"]})
        for c, bbox, ann_id in image["objects"]:
            label = label_of(c)
            if label is None:
                dropped += 1
                continue
            box = clip_xywh(bbox, image["width"], image["height"])
            if box[2] <= 0 or box[3] <= 0:
                clipped_out += 1
                continue
            coco["annotations"].append({"id": ann_id, "image_id": image["image_id"], "category_id": label,
                                        "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
    return coco, dropped, clipped_out


def build_dataset(variant, out_dir, data_dir, images_csv, classes_csv, fmt="both", link="hardlink"):
    """images.csv와 classes.csv대로 한 방법의 학습 폴더를 만든다. (모듈 docstring의 학습 폴더 참고)

    Args:
        variant (str): 방법. (v1~v4)
        out_dir (str | Path): 만들 폴더. 없거나 비어 있어야 한다. 학습 데이터 원본 폴더 안이나,
            레포 안에서 .gitignore에 걸리지 않는 곳(예: data/datasets/)이면 안 된다.
        data_dir (str | Path): data 폴더. 정제한 학습 데이터(ANN_DIRS, IMAGE_DIRS)가 있어야 한다.
        images_csv (str | Path): data/datasets/images.csv.
        classes_csv (str | Path): data/datasets/classes.csv.
        fmt (str): "yolo", "coco", "both". 라벨 형식만 고른다. 이미지와 eval/val_gt.json은 형식과 상관없이 늘 둔다.
        link (str): 이미지를 두는 방법. "hardlink"(안 되면 복사), "symlink", "copy".

    Returns:
        dict: dataset_info.json에 쓴 내용. (개수, 뺀 객체 수, 이미지를 둔 방법별 수 등)

    Raises:
        ValueError: 인자가 틀렸거나, 출력 폴더가 위 조건에 맞지 않거나, 목록의 이미지가 학습 데이터(annotation, PNG)에 없을 때.
            폴더를 만들기 전에 확인한다.
    """
    if variant not in VARIANTS or fmt not in ("yolo", "coco", "both") or link not in ("hardlink", "symlink", "copy"):
        raise ValueError(f"인자가 맞지 않습니다: variant={variant}, fmt={fmt}, link={link}\n"
                         f"[해결] variant는 {', '.join(VARIANTS)}, fmt는 yolo/coco/both, link는 hardlink/symlink/copy 중 하나입니다.")
    out_dir, data_dir = Path(out_dir).resolve(), Path(data_dir).resolve()
    protected = [data_dir / d for d in (*ANN_DIRS.values(), *IMAGE_DIRS.values())] + [data_dir / "raw", data_dir / "external"]
    if any(out_dir == p or out_dir.is_relative_to(p) for p in protected):
        raise ValueError(f"출력 폴더가 학습 데이터 원본 폴더 안입니다: {out_dir}\n"
                         "[해결] --out에 다른 폴더를 주세요. 예: --out data/processed/exp2/v1 또는 /content/exp2/v1")
    if _tracked_by_git(out_dir):
        raise ValueError(f"출력 폴더가 레포 안인데 .gitignore에 걸리지 않습니다: {out_dir}\n"
                         "이미지와 라벨 수만 개가 git add에 함께 올라갈 수 있습니다.\n"
                         "[해결] --out data/processed/exp2/v1처럼 data/processed/ 아래나, /content/exp2/v1처럼 레포 밖 폴더를 주세요.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"출력 폴더가 비어 있지 않습니다: {out_dir}\n"
                         "[해결] 다시 만들려면 그 폴더를 지우거나, 다른 --out을 주세요.")

    lists = read_image_list(images_csv, variant)
    table = read_class_table(classes_csv, variant)
    images, _ = load_images(data_dir)
    missing = [name for names in lists.values() for name in names if name not in images]
    if missing:
        raise ValueError(f"목록의 이미지 {len(missing):,}장이 학습 데이터에 없습니다. 예: {missing[0]}\n"
                         "[해결] 학습 데이터를 다시 받거나(Drive zip) 만들고(uv run scripts/prepare_train_data.py), "
                         "레포를 최신으로 맞추세요(git pull).")
    sources = {name: data_dir / IMAGE_DIRS[images[name]["source"]] / name for names in lists.values() for name in names}
    no_png = [path for path in sources.values() if not path.is_file()]
    if no_png:
        raise ValueError(f"목록의 이미지 {len(no_png):,}장의 PNG 파일이 없습니다. 예: {no_png[0]}\n"
                         "[해결] Drive의 이미지 zip을 레포 최상위에 다시 풀거나, uv run scripts/prepare_train_data.py로 다시 만드세요.")

    classes, competition, rows = table["classes"], table["competition"], table["rows"]
    index = {c: i for i, c in enumerate(classes)}

    def label_of(c):
        label = variant_category(c, variant, competition)
        return label if label in index else None

    info = {"variant": variant, "format": fmt, "nc": len(classes),
            "n_images": {split: len(names) for split, names in lists.items()},
            "n_objects": {}, "dropped_objects": {}, "clipped_out_objects": {}, "placed": collections.Counter(),
            "images_csv_sha256": _sha256(images_csv), "classes_csv_sha256": _sha256(classes_csv),
            "git_commit": _git_commit()}
    out_dir.mkdir(parents=True, exist_ok=True)

    # 채점용 val 정답: 모든 객체를 원래 K-코드로 둔다
    gt_categories = [_coco_category(c, rows[c]) for c in sorted({c for name in lists["val"] for c in object_classes(images[name])})]
    gt, _, _ = _coco(images, lists["val"], gt_categories, lambda c: c)
    (out_dir / "eval").mkdir()
    (out_dir / "eval" / "val_gt.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")

    categories = [{**_coco_category(c, rows[c]), "idx": i} for i, c in enumerate(classes)]
    for split, names in lists.items():
        coco, dropped, clipped_out = _coco(images, names, categories, label_of)
        info["n_objects"][split] = len(coco["annotations"])
        info["dropped_objects"][split] = dropped
        info["clipped_out_objects"][split] = clipped_out
        image_dir = out_dir / "images" / split
        image_dir.mkdir(parents=True)
        for name in names:
            info["placed"][_place(sources[name], image_dir / name, link)] += 1
        if fmt in ("coco", "both"):
            (out_dir / "coco").mkdir(exist_ok=True)
            (out_dir / "coco" / f"{split}.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        if fmt in ("yolo", "both"):
            label_dir = out_dir / "labels" / split
            label_dir.mkdir(parents=True)
            by_image = collections.defaultdict(list)
            for ann in coco["annotations"]:
                by_image[ann["image_id"]].append(ann)
            for name in names:
                image = images[name]
                lines = []
                for ann in by_image[image["image_id"]]:
                    cx, cy, w, h = xywh_to_yolo(ann["bbox"], image["width"], image["height"])
                    lines.append(f"{index[ann['category_id']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                (label_dir / f"{Path(name).stem}.txt").write_text("".join(lines), encoding="utf-8")
    if fmt in ("yolo", "both"):
        data_yaml = {"path": str(out_dir), "train": "images/train", "val": "images/val", "nc": len(classes),
                     "names": {i: category_label(c) for i, c in enumerate(classes)}}
        (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copy2(classes_csv, out_dir / "classes.csv")
    info["placed"] = dict(info["placed"])
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info
