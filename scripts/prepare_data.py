"""data/raw → data/processed(YOLO 형식), configs/data.yaml 생성.

    uv run python scripts/prepare_data.py                         # 로컬
    uv run python scripts/prepare_data.py --raw-dir /content/raw  # Colab 등 원본이 다른 곳에 있을 때
    uv run python scripts/prepare_data.py --split configs/splits/split_v1.json  # 이전 실험 재현

split은 configs/splits/에서 번호가 가장 큰 파일을 쓴다. 새 split은 scripts/make_split.py로 만든다.
같은 split과 data_fixes.yaml이면 누가 어디서 실행해도 같은 데이터가 만들어진다.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from pill_detection.data import convert
from pill_detection.data.classes import CLASSES, yolo_names

DATA_YAML = Path("configs/data.yaml")


def parse_args():
    """명령줄 옵션을 읽는다.

    Returns:
        argparse.Namespace: raw_dir, out_dir, split, fixes.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw", help="train_images, train_annotations가 있는 폴더")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--split", help="기본: configs/splits/에서 번호가 가장 큰 split. 이전 실험을 재현할 때만 지정")
    parser.add_argument("--fixes", default="configs/data_fixes.yaml")
    return parser.parse_args()


def short_hash(text):
    """data_version에 넣을 짧은 해시를 만든다.

    Args:
        text (str): split 파일이나 data_fixes.yaml의 내용.

    Returns:
        str: 내용의 sha256 앞 8자리. 내용이 한 글자라도 바뀌면 달라진다.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def print_report(raw_issues, issues):
    """점검 리포트를 출력한다. 항목마다 원본 대비 남은 개수와, 남은 문제의 상세 목록을 보여 준다.

    Args:
        raw_issues (dict): 원본의 convert.inspect() 결과.
        issues (dict): data_fixes.yaml을 반영한 뒤의 convert.inspect() 결과.
    """
    print("[점검 리포트] data_fixes.yaml을 반영한 뒤 남은 문제입니다. 고칠지는 EDA 후 정해서 configs/data_fixes.yaml에 적습니다.")
    units = {"missing_labels": "장", "same_box": "개", "out_of_image": "개"}
    counts = {k: f"원본 {len(raw_issues[k])}{u} → 남음 {len(issues[k])}{u}" for k, u in units.items()}
    print(f"  라벨 누락 (이미지 이름에는 있는데 라벨이 없는 알약): {counts['missing_labels']}")
    for item in issues["missing_labels"]:
        print(f"    {item['file_name']}  category_id {item['category_ids']}")
    print(f"  bbox가 완전히 같은 라벨: {counts['same_box']}")
    for item in issues["same_box"]:
        print(f"    {item['file_name']}  annotation_id {item['annotation_id']}  "
              f"category_id {item['category_id']}  bbox {item['bbox']}")
    # apply_fixes가 잘라내므로 반영 후에는 항상 0이다
    print(f"  이미지 밖으로 나간 bbox: {counts['out_of_image']} (잘라내고, 크기가 없어진 라벨은 위 [경고]에 나옴)")


def load_split(split_text, coco):
    """split 파일 내용을 읽고 조합 단위로 나뉘었는지 검사한다.

    split 파일에 없는 이미지가 있으면 학습에서 빠진다고 경고를 출력한다.

    Args:
        split_text (str): split 파일 내용.
        coco (dict): apply_fixes()를 거친 COCO 형식 dict.

    Returns:
        dict: split 파일 내용. {seed, val_ratio, train, val}
    """
    split = json.loads(split_text)
    convert.check_split(split)
    not_in_split = {im["file_name"] for im in coco["images"]} - set(split["train"]) - set(split["val"])
    if not_in_split:
        print(f"[경고] split 파일에 없는 이미지 {len(not_in_split)}장은 학습에서 빠집니다: {sorted(not_in_split)[:5]} ...")
    return split


def write_data_yaml(out_dir, data_version):
    """configs/data.yaml을 만든다. names는 classes.py에서 가져온다.

    Args:
        out_dir (Path): YOLO 데이터셋 폴더. data.yaml의 path가 된다.
        data_version (str): split과 data_fixes.yaml을 구분하는 값. train.py가 W&B에 기록한다.
    """
    content = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "data_version": data_version,
        "names": yolo_names(),
    }
    header = "# prepare_data.py가 classes.py로 생성한다. 손으로 고치지 않는다.\n"
    DATA_YAML.write_text(header + yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    """원본에 data_fixes.yaml을 반영해 YOLO 형식으로 쓰고 configs/data.yaml을 만든다.

    Returns:
        int: 종료 코드. 0 성공, 1 원본 폴더·split 파일이 없거나 classes.py에 없는 category_id가 있을 때.
    """
    args = parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    if not (raw_dir / "train_annotations").is_dir() or not (raw_dir / "train_images").is_dir():
        print(f"[오류] {raw_dir}에 train_annotations/, train_images/ 폴더가 없습니다. --raw-dir를 확인하세요.")
        return 1
    split_path = Path(args.split) if args.split else convert.latest_split()
    if split_path is None or not split_path.is_file():  # 경로 오타일 때 새로 만들지 않는다
        print(f"[오류] split 파일이 없습니다: {split_path or convert.SPLITS_DIR / 'split_v*.json'}\n"
              "  경로를 확인하세요. 새 split은 uv run python scripts/make_split.py로 만듭니다.")
        return 1

    coco = convert.load_raw_annotations(raw_dir)
    unknown = sorted({a["category_id"] for a in coco["annotations"]} - {cid for cid, _ in CLASSES})
    if unknown:
        print(f"[오류] classes.py에 없는 category_id가 있습니다: {unknown}\n"
              "  원본 데이터가 바뀌었다면 팀과 상의한 뒤 classes.py 목록 끝에 추가하세요.")
        return 1
    print(f"[원본] 이미지 {len(coco['images'])}장, 라벨 {len(coco['annotations'])}개, "
          f"클래스 {len(coco['categories'])}개")
    raw_issues = convert.inspect(coco)

    fixes_path = Path(args.fixes)
    fixes_text = fixes_path.read_text(encoding="utf-8")
    coco, empty = convert.apply_fixes(coco, yaml.safe_load(fixes_text) or {})
    if empty:
        print(f"[경고] 이미지 밖에 있어 잘라내니 크기가 없어진 라벨 {len(empty)}개를 뺐습니다: annotation_id {empty}\n"
              "  좌표 오타라면 data_fixes.yaml의 fix_bboxes에 올바른 좌표를 적으세요.")
    print(f"[전처리] 이미지 {len(coco['images'])}장, 라벨 {len(coco['annotations'])}개 (data_fixes.yaml 반영)")
    # 학습에 들어가는 데이터를 다시 점검해야 고친 것이 반영됐는지, 고치다 새로 생긴 문제가 없는지 알 수 있다
    print_report(raw_issues, convert.inspect(coco))

    print(f"[split] {split_path} 사용" + ("" if args.split else " (번호가 가장 큰 파일)"))
    split_text = split_path.read_text(encoding="utf-8")
    split = load_split(split_text, coco)
    counts = convert.write_yolo_dataset(coco, split, raw_dir, out_dir)

    # split이나 data_fixes.yaml의 내용이 바뀌면 data_version도 바뀐다. train.py가 W&B에 기록한다.
    data_version = f"{split_path.stem}-{short_hash(split_text)}+fixes-{short_hash(fixes_text)}"
    write_data_yaml(out_dir, data_version)
    print(f"[완료] train {counts['train']}장, val {counts['val']}장 → {out_dir}")
    print(f"       {DATA_YAML} 생성 (data_version: {data_version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
