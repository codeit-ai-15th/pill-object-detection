"""원본 어노테이션(data/raw) → YOLO 형식.

원본 구조: train_annotations/<알약 조합>_json/<알약 코드>/<이미지 이름>.json
JSON 1개에는 이미지 1장의 알약 1개만 들어 있다. 이미지 기준으로 합쳐 COCO 형식 dict 하나로 다룬다.
    {"images": [{id, file_name, width, height}], "annotations": [{id, image_id, category_id, bbox}],
     "categories": [{id, name}]}
"""

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from pill_detection.data.classes import category_to_yolo
from pill_detection.utils import clip_xywh, xywh_to_yolo


def load_raw_annotations(raw_dir):
    """train_annotations 아래 JSON을 모두 읽어 COCO 형식 dict 하나로 합친다.

    Args:
        raw_dir (str | Path): train_annotations/ 폴더가 있는 원본 폴더. 예) data/raw

    Returns:
        dict: 모듈 docstring 구조의 COCO 형식 dict. images는 파일 이름 순, annotations는 annotation_id 순,
            categories는 category_id 순이다.
    """
    images, annotations, categories = {}, [], {}
    for path in sorted((Path(raw_dir) / "train_annotations").rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for image in data["images"]:
            images[image["id"]] = {k: image[k] for k in ("id", "file_name", "width", "height")}
        for ann in data["annotations"]:
            annotations.append({k: ann[k] for k in ("id", "image_id", "category_id", "bbox")})
        for category in data["categories"]:
            categories[category["id"]] = category["name"]
    return {
        "images": sorted(images.values(), key=lambda im: im["file_name"]),
        "annotations": sorted(annotations, key=lambda a: a["id"]),
        "categories": [{"id": cid, "name": name} for cid, name in sorted(categories.items())],
    }


def combo_of(file_name):
    """이미지 이름에서 알약 조합을 꺼낸다.

    Args:
        file_name (str): 이미지 파일 이름. 예) 'K-003351-016232-033880_0_2_..png'

    Returns:
        str: 알약 조합. 예) 'K-003351-016232-033880'
    """
    return file_name.split("_")[0]


def _pill_codes(file_name):
    """이미지 이름에 적힌 알약 코드를 꺼낸다. 사진에 있어야 하는 category_id와 같다.

    Args:
        file_name (str): 이미지 파일 이름. 예) 'K-003351-016232_..png'

    Returns:
        set[int]: 알약 코드. 예) {3351, 16232}
    """
    return {int(code) for code in combo_of(file_name).split("-")[1:]}


def _annotations_by_image(coco):
    """라벨을 이미지별로 묶는다.

    Args:
        coco (dict): COCO 형식 dict.

    Returns:
        defaultdict[int, list[dict]]: image_id → 그 이미지의 라벨 목록. 라벨이 없는 이미지는 빈 목록.
    """
    grouped = defaultdict(list)
    for ann in coco["annotations"]:
        grouped[ann["image_id"]].append(ann)
    return grouped


def inspect(coco):
    """학습 전에 알아야 할 라벨 문제를 찾는다. 고칠지는 사람이 정해서 configs/data_fixes.yaml에 적는다.

    원본에도, apply_fixes() 결과에도 쓴다. prepare_data.py는 둘 다 검사해 고친 뒤 남은 문제를 보여 준다.

    Args:
        coco (dict): COCO 형식 dict.

    Returns:
        dict[str, list[dict]]: 문제 종류별 목록.
            - missing_labels: 이미지 이름에는 있는 알약 코드인데 라벨이 없다 (사진 속 알약이 배경으로 학습됨).
              항목은 {file_name, category_ids}.
            - same_box: 같은 이미지에서 bbox 좌표가 완전히 같은 라벨이 2개 이상 있다.
            - out_of_image: bbox가 이미지 밖으로 나간다 (apply_fixes가 잘라낸다).
            same_box와 out_of_image의 항목은 {file_name, annotation_id, category_id, bbox}.
    """
    anns_by_image = _annotations_by_image(coco)
    issues = {"missing_labels": [], "same_box": [], "out_of_image": []}
    for image in coco["images"]:
        anns = anns_by_image[image["id"]]
        missing = sorted(_pill_codes(image["file_name"]) - {a["category_id"] for a in anns})
        if missing:
            issues["missing_labels"].append({"file_name": image["file_name"], "category_ids": missing})

        box_count = Counter(tuple(a["bbox"]) for a in anns)
        for a in anns:
            x, y, w, h = a["bbox"]
            entry = {"file_name": image["file_name"], "annotation_id": a["id"],
                     "category_id": a["category_id"], "bbox": a["bbox"]}
            if box_count[tuple(a["bbox"])] > 1:
                issues["same_box"].append(entry)
            if x < 0 or y < 0 or x + w > image["width"] or y + h > image["height"]:
                issues["out_of_image"].append(entry)
    return issues


FIX_KEYS = ("exclude_images", "drop_annotations", "fix_bboxes", "add_annotations")  # data_fixes.yaml에 쓸 수 있는 키
ADD_ANNOTATION_KEYS = {"file_name", "category_id", "bbox"}  # add_annotations 항목 하나에 쓰는 키


def _check_added_annotations(added, coco, excluded):
    """add_annotations 항목을 검사한다. 잘못 적은 라벨이 오류 없이 학습에 들어가는 일을 막는다.

    Args:
        added (list[dict]): data_fixes.yaml의 add_annotations 목록.
        coco (dict): 고치기 전의 COCO 형식 dict.
        excluded (set[str]): exclude_images로 뺄 이미지 파일 이름.

    Raises:
        ValueError: 키가 틀렸거나, 원본에 없거나 뺀 이미지이거나, 이미지 이름에 없는 알약 코드이거나,
            bbox가 숫자 4개가 아니거나, bbox가 이미지 밖에 있을 때.
    """
    images = {im["file_name"]: im for im in coco["images"]}
    for i, item in enumerate(added, start=1):
        where = f"data_fixes.yaml add_annotations의 {i}번째 항목 {item}"
        if not isinstance(item, dict) or set(item) != ADD_ANNOTATION_KEYS:
            raise ValueError(f"{where}: file_name, category_id, bbox 세 키를 적습니다.")
        image = images.get(item["file_name"])
        if image is None or item["file_name"] in excluded:
            raise ValueError(f"{where}: 원본에 없거나 exclude_images로 뺀 이미지입니다.")
        if item["category_id"] not in _pill_codes(item["file_name"]):
            raise ValueError(f"{where}: 이미지 이름에 없는 알약 코드입니다. category_id를 확인하세요.")
        bbox = item["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            raise ValueError(f"{where}: bbox는 [x, y, w, h] 숫자 4개입니다.")
        _, _, w, h = clip_xywh(bbox, image["width"], image["height"])
        if w <= 1 or h <= 1:
            raise ValueError(f"{where}: bbox가 이미지({image['width']}x{image['height']}) 밖에 있습니다.")


def apply_fixes(coco, fixes):
    """data_fixes.yaml의 목록을 반영하고 이미지 밖으로 나간 bbox를 잘라낸다.

    잘라낸 뒤 너비나 높이가 1픽셀 이하인 bbox는 버린다. 이런 라벨이 하나라도 있으면
    Ultralytics가 그 이미지 전체를 손상된 것으로 보고 학습에서 빼기 때문이다.

    add_annotations로 넣은 라벨의 annotation_id는 원본에서 가장 큰 번호 다음부터 적힌 순서대로 붙인다.
    FIX_KEYS에 없는 키는 오류로 막는다. 그냥 두면 아무것도 고쳐지지 않는데 data_version만 바뀐다.

    Args:
        coco (dict): 원본 COCO 형식 dict. 바꾸지 않는다.
        fixes (dict): data_fixes.yaml을 읽은 dict. 키는 FIX_KEYS 중에서 쓴다.

    Returns:
        tuple[dict, list[int]]: 고친 새 COCO 형식 dict와, 잘라내니 크기가 없어져 버린 annotation_id 목록.

    Raises:
        ValueError: 모르는 키가 있거나, 원본에 없는 이미지 이름·annotation_id를 적었거나,
            add_annotations 항목이 잘못됐을 때.
    """
    unknown_keys = sorted(set(fixes) - set(FIX_KEYS))
    if unknown_keys:
        raise ValueError(f"data_fixes.yaml에 모르는 키가 있습니다: {unknown_keys}  쓸 수 있는 키: {list(FIX_KEYS)}\n"
                         "  오타가 아니라 새 종류의 수정이라면 convert.apply_fixes()에 처리 코드를 먼저 추가하세요.")

    excluded = set(fixes.get("exclude_images") or [])
    dropped = set(fixes.get("drop_annotations") or [])
    new_bboxes = fixes.get("fix_bboxes") or {}
    added = fixes.get("add_annotations") or []

    # 오타로 아무것도 안 고쳐지는 일을 막는다
    ann_ids = {a["id"] for a in coco["annotations"]}
    unknown = {
        "exclude_images": sorted(excluded - {im["file_name"] for im in coco["images"]}),
        "drop_annotations": sorted(dropped - ann_ids),
        "fix_bboxes": sorted(set(new_bboxes) - ann_ids),
    }
    if any(unknown.values()):
        raise ValueError(f"data_fixes.yaml에 원본에 없는 값이 있습니다: {unknown}")
    _check_added_annotations(added, coco, excluded)

    images = [im for im in coco["images"] if im["file_name"] not in excluded]
    sizes = {im["id"]: (im["width"], im["height"]) for im in images}
    annotations, empty = [], []
    for a in coco["annotations"]:
        if a["image_id"] not in sizes or a["id"] in dropped:
            continue
        bbox = clip_xywh(new_bboxes.get(a["id"], a["bbox"]), *sizes[a["image_id"]])
        if bbox[2] <= 1 or bbox[3] <= 1:
            empty.append(a["id"])
            continue
        annotations.append({**a, "bbox": bbox})

    image_ids = {im["file_name"]: im["id"] for im in images}
    next_id = max(ann_ids, default=0) + 1  # 원본 annotation_id와 겹치지 않게 한다
    for i, item in enumerate(added):
        image_id = image_ids[item["file_name"]]
        annotations.append({"id": next_id + i, "image_id": image_id, "category_id": item["category_id"],
                            "bbox": clip_xywh(item["bbox"], *sizes[image_id])})
    return {**coco, "images": images, "annotations": annotations}, empty


def make_group_split(coco, val_ratio=0.2, seed=42):
    """알약 조합 단위로 train/val을 나눈다.

    같은 조합은 카메라 각도만 다른 거의 같은 사진이라, 이미지 단위로 나누면 val 점수가 부풀려진다.
    val로 보낼 조합을 무작위로 고르되, 어떤 클래스든 train에 최소 1개는 남긴다.

    Args:
        coco (dict): COCO 형식 dict. 보통 apply_fixes()를 거친 것.
        val_ratio (float): val로 보낼 이미지 비율의 목표값. 조합 단위로 옮기므로 정확히 맞지는 않는다.
        seed (int): 조합을 섞는 난수 seed. 같으면 같은 split이 나온다.

    Returns:
        dict: split 파일 내용. {seed, val_ratio, train: [파일 이름], val: [파일 이름]}
    """
    image_cats = defaultdict(Counter)
    for ann in coco["annotations"]:
        image_cats[ann["image_id"]][ann["category_id"]] += 1

    groups = defaultdict(list)          # 조합 → 이미지 이름 목록
    group_cats = defaultdict(Counter)   # 조합 → 클래스별 인스턴스 수
    for image in coco["images"]:
        combo = combo_of(image["file_name"])
        groups[combo].append(image["file_name"])
        group_cats[combo].update(image_cats[image["id"]])

    train_cats = sum(group_cats.values(), Counter())
    order = sorted(groups)
    random.Random(seed).shuffle(order)

    val_target = round(len(coco["images"]) * val_ratio)
    val_groups, n_val = set(), 0
    for combo in order:
        if n_val >= val_target:
            break
        if any(train_cats[c] <= n for c, n in group_cats[combo].items()):
            continue  # 이 조합을 빼면 train에서 사라지는 클래스가 생김
        val_groups.add(combo)
        n_val += len(groups[combo])
        train_cats -= group_cats[combo]

    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "train": sorted(f for g in groups if g not in val_groups for f in groups[g]),
        "val": sorted(f for g in val_groups for f in groups[g]),
    }


SPLITS_DIR = Path("configs/splits")  # split_v1.json, split_v2.json, ... 한 번 만든 파일은 고치지 않는다


def split_versions(splits_dir=SPLITS_DIR):
    """split_v<번호>.json 파일을 번호 순서로 돌려준다. 숫자로 정렬하므로 v10이 v9 뒤에 온다.

    Args:
        splits_dir (str | Path): split 파일이 있는 폴더.

    Returns:
        list[Path]: 번호 순서의 split 파일 경로. split_v2_old.json처럼 형식이 다른 이름은 뺀다.
    """
    found = []
    for path in Path(splits_dir).glob("split_v*.json"):
        number = path.stem.removeprefix("split_v")
        if number.isdigit():  # split_v2_old.json 같은 이름은 무시한다
            found.append((int(number), path))
    return [path for _, path in sorted(found)]


def latest_split(splits_dir=SPLITS_DIR):
    """번호가 가장 큰 split 파일을 찾는다.

    Args:
        splits_dir (str | Path): split 파일이 있는 폴더.

    Returns:
        Path | None: 번호가 가장 큰 split 파일. 하나도 없으면 None.
    """
    versions = split_versions(splits_dir)
    return versions[-1] if versions else None


def next_split_path(splits_dir=SPLITS_DIR):
    """새 split을 저장할 다음 번호의 경로를 정한다.

    Args:
        splits_dir (str | Path): split 파일이 있는 폴더.

    Returns:
        Path: 번호가 가장 큰 파일 다음 번호의 경로. 하나도 없으면 split_v1.json.
    """
    latest = latest_split(splits_dir)
    number = int(latest.stem.removeprefix("split_v")) + 1 if latest else 1
    return Path(splits_dir) / f"split_v{number}.json"


def check_split(split):
    """split이 조합 단위로 나뉘었는지 검사한다. 손으로 고친 파일도 여기서 잡는다.

    Args:
        split (dict): split 파일 내용. train, val에 이미지 파일 이름 목록이 있다.

    Raises:
        ValueError: 같은 이미지가 train과 val에 모두 있거나, 같은 조합이 train과 val로 나뉘었을 때.
    """
    both = sorted(set(split["train"]) & set(split["val"]))
    if both:
        raise ValueError(f"split 파일에 train과 val에 모두 있는 이미지가 있습니다: {both[:5]}")
    shared = sorted({combo_of(f) for f in split["train"]} & {combo_of(f) for f in split["val"]})
    if shared:
        raise ValueError(f"split 파일에 train과 val로 나뉜 알약 조합이 있습니다: {shared[:5]}\n"
                         "  같은 조합은 거의 같은 사진이라 한쪽에만 두어야 val 점수가 부풀려지지 않습니다.")


def _link_or_copy(src, dst):
    """같은 디스크면 하드링크(공간·시간 절약), 아니면 복사한다.

    Args:
        src (Path): 원본 파일 경로.
        dst (Path): 만들 파일 경로.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_dataset(coco, split, raw_dir, out_dir):
    """out_dir/{images,labels}/{train,val}에 이미지와 YOLO 라벨을 쓴다.

    이전 실행 결과가 섞이지 않게 out_dir/images, out_dir/labels를 먼저 지운다.

    Args:
        coco (dict): apply_fixes()를 거친 COCO 형식 dict.
        split (dict): split 파일 내용. coco에 없는 이미지(exclude_images로 뺀 것)는 건너뛴다.
        raw_dir (str | Path): train_images/ 폴더가 있는 원본 폴더.
        out_dir (str | Path): 결과를 쓸 폴더. 예) data/processed

    Returns:
        dict[str, int]: subset별로 쓴 이미지 수. {"train": 186, "val": 46}
    """
    out_dir = Path(out_dir)
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)  # 이전 실행 결과가 섞이지 않게 비운다

    anns_by_image = _annotations_by_image(coco)
    image_by_name = {im["file_name"]: im for im in coco["images"]}
    counts = {}
    for subset in ("train", "val"):
        img_dir, lbl_dir = out_dir / "images" / subset, out_dir / "labels" / subset
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n = 0
        for file_name in split[subset]:
            image = image_by_name.get(file_name)
            if image is None:
                continue  # data_fixes.yaml에서 뺀 이미지
            _link_or_copy(Path(raw_dir) / "train_images" / file_name, img_dir / file_name)
            lines = []
            for ann in anns_by_image[image["id"]]:
                cx, cy, w, h = xywh_to_yolo(ann["bbox"], image["width"], image["height"])
                lines.append(f"{category_to_yolo(ann['category_id'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += 1
        counts[subset] = n
    return counts
