"""조합 단위 train/val split을 다음 번호로 새로 만든다. (EXPERIMENTS.md 규칙 4)

    uv run python scripts/make_split.py                            # val 20%, seed 42
    uv run python scripts/make_split.py --val-ratio 0.15 --seed 7

configs/splits/에서 번호가 가장 큰 파일 다음 번호(split_v2.json, split_v3.json, ...)로 저장한다.
나누는 규칙을 바꾸려면 convert.make_group_split()을 고친 뒤 실행한다.
이미 있는 split 파일은 고치지 않는다. 이전 실험을 재현할 때 필요하다.
만든 파일을 커밋하면 prepare_data.py가 번호가 가장 큰 split을 쓰므로, main에 머지되는 순간 팀 전체에 반영된다.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from pill_detection.data import convert
from pill_detection.data.classes import NUM_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw", help="train_images, train_annotations가 있는 폴더")
    parser.add_argument("--fixes", default="configs/data_fixes.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_summary(split, coco):
    cats_by_name = {im["file_name"]: set() for im in coco["images"]}
    image_names = {im["id"]: im["file_name"] for im in coco["images"]}
    for ann in coco["annotations"]:
        cats_by_name[image_names[ann["image_id"]]].add(ann["category_id"])
    val_cats = set().union(*(cats_by_name[f] for f in split["val"]))
    print(f"  train {len(split['train'])}장, val {len(split['val'])}장 "
          f"(val 조합 {len({convert.combo_of(f) for f in split['val']})}개, "
          f"val 클래스 {len(val_cats)}/{NUM_CLASSES}개)")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if not (raw_dir / "train_annotations").is_dir() or not (raw_dir / "train_images").is_dir():
        print(f"[오류] {raw_dir}에 train_annotations/, train_images/ 폴더가 없습니다. --raw-dir를 확인하세요.")
        return 1

    coco = convert.load_raw_annotations(raw_dir)
    fixes = yaml.safe_load(Path(args.fixes).read_text(encoding="utf-8")) or {}
    coco, _ = convert.apply_fixes(coco, fixes)
    split = convert.make_group_split(coco, args.val_ratio, args.seed)
    convert.check_split(split)  # make_group_split()을 고치다 조합 단위 분할이 깨지면 여기서 멈춘다

    # 실수로 두 번 실행해서 내용이 같은 새 번호가 생기는 것을 막는다
    for path in convert.split_versions():
        old = json.loads(path.read_text(encoding="utf-8"))
        if (old["train"], old["val"]) == (split["train"], split["val"]):
            print(f"[안내] 결과가 {path}의 train/val과 같아서 새로 만들지 않았습니다.\n"
                  "  make_group_split()을 고쳤거나 --val-ratio, --seed를 바꿨는지 확인하세요.")
            return 1

    path = convert.next_split_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] 새로 만듦: {path}  (val_ratio {args.val_ratio}, seed {args.seed})")
    print_summary(split, coco)
    print(f"  새로 만든 {path}는 git에 커밋하세요. prepare_data.py는 가장 큰 버전의 split을 씁니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
