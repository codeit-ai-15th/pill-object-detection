"""data/datasets/의 목록 CSV대로 한 방법(v1~v4)의 학습 폴더를 만든다. (만들기 단계)

팀원이 학습할 때마다 실행한다. 결과 폴더는 커밋하지 않는다.

사용법 (레포 최상위에서)
    uv run scripts/build_dataset.py --variant v2 --out data/processed/exp2/v2
    # Colab (uv 없이)
    PYTHONPATH=src python scripts/build_dataset.py --variant v2 --out /content/exp2/v2

인자
    --variant   v1(전부), v2(클래스별 최소 200), v3(대회 클래스가 든 이미지), v4(대회 클래스 밖 → others)
    --out       만들 폴더. 없거나 비어 있어야 한다. 레포 안이면 data/processed/ 아래여야 한다(git에 올라가지 않게).
    --format    yolo, coco, both (기본 both). 라벨 형식만 고른다. 이미지와 eval/val_gt.json은 형식과 상관없이 늘 둔다.
    --link      hardlink(기본, 안 되면 복사), symlink, copy. 이미지를 --out에 두는 방법이다.
    --data-dir  정제한 학습 데이터가 있는 data 폴더 (기본: 레포의 data)

만드는 것 (pill_detection.datasets 모듈 docstring 참고)
    <out>/images/{train,val}             # 이미지. YOLO와 COCO가 같이 쓴다
    <out>/labels, data.yaml              # YOLO (Ultralytics: data=<out>/data.yaml)
    <out>/coco/train.json, val.json      # COCO (torchvision CocoDetection: root=<out>/images/train, annFile=<out>/coco/train.json)
    <out>/eval/val_gt.json               # 채점용 val 정답 (pill_detection.evaluate.evaluate_val)
    <out>/classes.csv, dataset_info.json
"""

import argparse
import sys
from pathlib import Path

from pill_detection import datasets as ds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIST_DIR = PROJECT_ROOT / "data" / "datasets"


def main(argv=None):
    """학습 폴더를 만들고 개수를 출력한다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    parser = argparse.ArgumentParser(description="data/datasets/의 목록대로 한 방법의 학습 폴더(YOLO, COCO)를 만든다. "
                                                 "자세한 설명은 이 파일 맨 위를 본다.")
    parser.add_argument("--variant", required=True, choices=ds.VARIANTS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--format", default="both", choices=("yolo", "coco", "both"))
    parser.add_argument("--link", default="hardlink", choices=("hardlink", "symlink", "copy"))
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args(argv)

    images_csv, classes_csv = LIST_DIR / "images.csv", LIST_DIR / "classes.csv"
    missing = [p.relative_to(PROJECT_ROOT).as_posix() for p in (images_csv, classes_csv) if not p.is_file()]
    if missing:
        print(f"[오류] 목록 CSV가 없습니다: {', '.join(missing)}\n[해결] git pull로 레포를 최신으로 맞추세요.")
        return 1
    print(f"[{args.variant}] 학습 데이터를 읽고 {args.out}에 학습 폴더를 만듭니다.", flush=True)
    try:
        info = ds.build_dataset(args.variant, args.out, args.data_dir, images_csv, classes_csv, fmt=args.format, link=args.link)
    except (FileNotFoundError, ValueError) as error:
        print(f"[오류] {error}")
        return 1
    print(f"클래스 {info['nc']}개, 이미지 train {info['n_images']['train']:,}장 / val {info['n_images']['val']:,}장, "
          f"객체 train {info['n_objects']['train']:,}개 / val {info['n_objects']['val']:,}개")
    dropped = {k: v for k, v in info["dropped_objects"].items() if v}
    clipped = {k: v for k, v in info["clipped_out_objects"].items() if v}
    if dropped:
        print(f"모델 클래스가 아니라서 라벨에서 뺀 객체: {dropped} (채점용 eval/val_gt.json에는 남아 있다)")
    if clipped:
        print(f"사진 밖으로 나가 너비나 높이가 0이 된 box: {clipped}")
    print(f"이미지: {info['placed']}")
    out = args.out.resolve()
    if args.format in ("yolo", "both"):
        print(f"YOLO 학습: data={out / 'data.yaml'}")
    if args.format in ("coco", "both"):
        print(f"COCO 학습: root={out / 'images' / 'train'}, annFile={out / 'coco' / 'train.json'} (val도 같은 모양)")
    print(f"채점 정답: {(args.out / 'eval' / 'val_gt.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
