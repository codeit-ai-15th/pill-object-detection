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
    """train_annotations 아래 JSON을 모두 읽어 COCO 형식 dict 하나로 합친다."""
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
    """이미지 이름에서 알약 조합을 꺼낸다. 'K-003351-016232-033880_0_2_..png' → 'K-003351-016232-033880'"""
    return file_name.split("_")[0]


def _annotations_by_image(coco):
    grouped = defaultdict(list)
    for ann in coco["annotations"]:
        grouped[ann["image_id"]].append(ann)
    return grouped


def inspect(coco):
    """학습 전에 알아야 할 라벨 문제를 찾는다. 고칠지는 사람이 정해서 configs/data_fixes.yaml에 적는다.

    - missing_labels: 이미지 이름에는 있는 알약 코드인데 라벨이 없다 (사진 속 알약이 배경으로 학습됨)
    - same_box: 같은 이미지에서 bbox 좌표가 완전히 같은 라벨이 2개 이상 있다
    - out_of_image: bbox가 이미지 밖으로 나간다 (apply_fixes가 잘라낸다)
    """
    anns_by_image = _annotations_by_image(coco)
    issues = {"missing_labels": [], "same_box": [], "out_of_image": []}
    for image in coco["images"]:
        anns = anns_by_image[image["id"]]
        expected = {int(code) for code in combo_of(image["file_name"]).split("-")[1:]}
        missing = sorted(expected - {a["category_id"] for a in anns})
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


def apply_fixes(coco, fixes):
    """data_fixes.yaml의 목록을 반영하고 이미지 밖으로 나간 bbox를 잘라낸다.

    잘라낸 뒤 너비나 높이가 1픽셀 이하인 bbox는 버린다. 이런 라벨이 하나라도 있으면
    Ultralytics가 그 이미지 전체를 손상된 것으로 보고 학습에서 빼기 때문이다.
    새 dict와 버린 annotation_id 목록을 돌려준다.
    """
    excluded = set(fixes.get("exclude_images") or [])
    dropped = set(fixes.get("drop_annotations") or [])
    new_bboxes = fixes.get("fix_bboxes") or {}

    # 오타로 아무것도 안 고쳐지는 일을 막는다
    ann_ids = {a["id"] for a in coco["annotations"]}
    unknown = {
        "exclude_images": sorted(excluded - {im["file_name"] for im in coco["images"]}),
        "drop_annotations": sorted(dropped - ann_ids),
        "fix_bboxes": sorted(set(new_bboxes) - ann_ids),
    }
    if any(unknown.values()):
        raise ValueError(f"data_fixes.yaml에 원본에 없는 값이 있습니다: {unknown}")

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
    return {**coco, "images": images, "annotations": annotations}, empty


def make_group_split(coco, val_ratio=0.2, seed=42):
    """알약 조합 단위로 train/val을 나눈다.

    같은 조합은 카메라 각도만 다른 거의 같은 사진이라, 이미지 단위로 나누면 val 점수가 부풀려진다.
    val로 보낼 조합을 무작위로 고르되, 어떤 클래스든 train에 최소 1개는 남긴다.
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


def _link_or_copy(src, dst):
    """같은 디스크면 하드링크(공간·시간 절약), 아니면 복사한다."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_dataset(coco, split, raw_dir, out_dir):
    """out_dir/{images,labels}/{train,val}에 이미지와 YOLO 라벨을 쓰고, subset별 이미지 수를 돌려준다."""
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
