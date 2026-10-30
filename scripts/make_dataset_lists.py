"""2차 실험의 val과 방법별 train을 정해 목록 CSV 2개를 쓰고 보고표를 출력한다. (결정 단계)

val을 정할 때 한 번 실행하고 결과를 커밋한다. 커밋한 뒤에는 val을 바꾸지 않는다. 바꾸면 그 전에 돌린 실험과 비교할 수 없다.
팀원은 이 스크립트를 실행하지 않고, 커밋된 CSV로 scripts/build_dataset.py를 실행한다.

사용법 (레포 최상위에서)
    uv run scripts/make_dataset_lists.py --exp1-split <1차 split.csv> [--test-crops <검수표>] [--out data/datasets] [--check]

    # 처음 만들 때 (1차 split.csv는 Drive에서 받는다)
    uv run scripts/make_dataset_lists.py --exp1-split ~/Downloads/split.csv
    # 커밋한 목록을 다시 만들어 같은지만 볼 때. 파일은 쓰지 않는다 (1차 분할은 images.csv의 exp1_split 열을 쓴다)
    uv run scripts/make_dataset_lists.py --check

입력
- 정제한 학습 데이터 (uv run scripts/prepare_train_data.py, 또는 Drive zip을 레포 최상위에 푼 것)
- 1차 split.csv: Kaggle 232장의 1차 train/val (file_name, split 열). 없으면 이미 쓴 images.csv의 exp1_split 열을 읽는다.
- test 알약 검수표: notebooks/eda/02_raewoo_test_crops.csv (노트북 02). 대회 클래스와 클래스별 test 객체 수를 센다.

출력 (pill_detection.datasets 모듈 docstring 참고. 열과 줄 설명: data/datasets/README.md)
- data/datasets/images.csv: 이미지 1장 = 1줄. split(2차 train/val), exp1_split, v1~v4, 객체 수
- data/datasets/classes.csv: 클래스 1개 = 1줄 + others. 그룹, 대회 클래스, 방법별 모델 클래스 번호, test/val/train 객체 수
- 같은 입력으로 다시 실행하면 파일을 건드리지 않고 "변경 없음"을 출력한다.
- --check면 파일을 쓰지 않고 비교만 한다. 다르면 바뀐 줄 수를 출력하고 종료 코드 1로 끝난다.
"""

import argparse
import sys
from pathlib import Path

from pill_detection import datasets as ds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_TEST_CROPS = PROJECT_ROOT / "notebooks" / "eda" / "02_raewoo_test_crops.csv"


def print_table(header, rows):
    """열 너비를 맞춘 글자 표를 출력한다."""
    widths = [max(len(str(v)) for v in column) for column in zip(header, *rows)]
    for line in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(str(v).rjust(w) if i else str(v).ljust(w) for i, (v, w) in enumerate(zip(line, widths))))


def report(result, images, test_counts):
    """보고표: 출처별 장수, 클래스별 객체 수와 비율, 목표 미달·상한·채점 안 되는 클래스, val의 대회 클래스 밖 비율."""
    competition, val, variants = result["competition"], result["val"], result["variants"]
    sets = {"2차 val": val["files"], **{f"train {v}": variants[v] for v in ds.VARIANTS}}

    print("\n[1] 출처별 이미지 수 (조합 수)")
    rows = []
    for name, files in sets.items():
        cells = []
        for source in ds.SOURCES:
            chosen = [n for n in files if images[n]["source"] == source]
            cells.append(f"{len(chosen):,} ({len({images[n]['combo'] for n in chosen}):,})")
        rows.append([name, *cells, f"{len(files):,}"])
    print_table(["집합", "Kaggle", "AI Hub", "합계"], rows)
    n_aihub = sum(image["source"] == "aihub" for image in images.values())
    n_val_aihub = sum(images[n]["source"] == "aihub" for n in val["files"])
    print(f"val의 AI Hub: 전체 AI Hub 이미지의 {n_val_aihub / n_aihub:.1%} (AI Hub 몫 {val['budget']:,.0f}개 객체)")

    print("\n[2] 대회 클래스별 객체 수 (비율 = 그 집합의 대회 클래스 객체 합계 대비)")
    counts = {"test": dict(test_counts), "val": ds.object_counts(images, val["files"]),
              **{v: ds.object_counts(images, variants[v], v, competition) for v in ds.VARIANTS}}
    totals = {k: sum(c[code] for code in competition if code in c) for k, c in counts.items()}
    rows = []
    for code in sorted(competition, key=lambda c: (-test_counts.get(c, 0), c)):
        rows.append([f"K-{code:06d}", ds.class_group(code, ds.kaggle_classes(images), competition),
                     *(f"{counts[k].get(code, 0):,} ({counts[k].get(code, 0) / totals[k]:.1%})" for k in counts)])
    print_table(["클래스", "그룹", "test", "val", *(f"train {v}" for v in ds.VARIANTS)], rows)

    print("\n[3] val 목표와 상한")
    short = [c for c, t in val["target"].items() if val["count"][c] < t]
    unscored = sorted(c for c in competition if not val["count"][c])
    print(f"목표 미달 {len(short)}개: " + (", ".join(f"K-{c:06d} {val['count'][c]}/{val['target'][c]}" for c in short) or "없음"))
    clipped = sorted(c for c in val["target"] if val["wanted"][c] > val["target"][c])
    print(f"test 비율의 목표가 상한(AI Hub 객체의 {ds.VAL_CLASS_CAP:.0%})에 잘린 클래스 {len(clipped)}개: "
          + (", ".join(f"K-{c:06d} {val['wanted'][c]}→{val['target'][c]}" for c in clipped) or "없음"))
    print(f"상한 때문에 조합을 건너뛴 클래스 {len(val['capped'])}개: " + (", ".join(f"K-{c:06d}" for c in sorted(val["capped"])) or "없음"))
    print(f"val에서 채점되지 않는 대회 클래스 {len(unscored)}개: " + (", ".join(f"K-{c:06d}" for c in unscored) or "없음"))
    no_aihub = sorted(c for c in competition if c not in val["target"])
    print("AI Hub에 없어 1차 val Kaggle 이미지로만 채운 클래스: "
          + ", ".join(f"K-{c:06d} {val['count'][c]}개" for c in no_aihub))
    noncompetition = sum(n for c, n in val["count"].items() if c not in competition)
    print(f"val의 대회 클래스 밖 알약: {noncompetition:,}개 ({noncompetition / sum(val['count'].values()):.1%}). 채점에서 무시 영역으로 쓴다")

    print("\n[4] 방법별 모델 클래스 수")
    print_table(["방법", "클래스 수", "train 이미지"],
                [[v, len(ds.model_classes(images, variants[v], v, competition)), f"{len(variants[v]):,}"] for v in ds.VARIANTS])


def main(argv=None):
    """목록 CSV 2개를 만들고 보고표를 출력한다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    parser = argparse.ArgumentParser(description="2차 실험의 val과 방법별 train을 정해 data/datasets/의 CSV 2개를 쓴다. "
                                                 "자세한 설명은 이 파일 맨 위를 본다.")
    parser.add_argument("--exp1-split", type=Path, help="1차 split.csv. 없으면 --out의 images.csv의 exp1_split 열을 쓴다.")
    parser.add_argument("--test-crops", type=Path, default=DEFAULT_TEST_CROPS, help="노트북 02의 test 알약 검수표.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV를 쓸 폴더.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="정제한 학습 데이터가 있는 data 폴더.")
    parser.add_argument("--check", action="store_true", help="파일을 쓰지 않고 지금 CSV와 같은지만 본다. 다르면 종료 코드 1.")
    args = parser.parse_args(argv)

    images_csv, classes_csv = args.out / "images.csv", args.out / "classes.csv"
    try:
        images, names = ds.load_images(args.data_dir)
        if args.exp1_split:
            exp1_split = ds.read_exp1_split(args.exp1_split, images)
        elif images_csv.is_file():
            exp1_split = ds.exp1_split_from_rows(ds.read_rows(images_csv))
        else:
            raise ValueError(f"1차 분할이 없습니다: --exp1-split을 주지 않았고 {images_csv}도 없습니다.\n"
                             "[해결] Drive의 1차 split.csv를 받아 --exp1-split <경로>로 주세요.")
        if not args.test_crops.is_file():
            raise FileNotFoundError(f"test 알약 검수표가 없습니다: {args.test_crops}\n"
                                    "[해결] notebooks/eda/02_raewoo_test_class_freq.ipynb를 실행해 검수표를 만드세요.")
        test_counts = ds.read_test_counts(args.test_crops)
        result = ds.make_lists(images, names, exp1_split, test_counts)
    except (FileNotFoundError, ValueError) as error:
        print(f"[오류] {error}")
        return 1

    report(result, images, test_counts)
    outputs = [(images_csv, ds.IMAGE_COLUMNS, result["image_rows"]), (classes_csv, ds.CLASS_COLUMNS, result["class_rows"])]
    shown = {path: path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path for path, _, _ in outputs}
    print()
    if args.check:
        diffs = {path: ds.diff_rows(path, columns, rows) for path, columns, rows in outputs}
        for path, diff in diffs.items():
            print(f"{shown[path]}: " + ("같음" if diff is None else f"다름 (새 줄 {diff['added']:,}개, 사라질 줄 {diff['removed']:,}개)"))
        if any(diffs.values()):
            print("[오류] 지금 입력으로 만든 목록이 커밋한 CSV와 다릅니다. 파일은 쓰지 않았습니다.\n"
                  "[해결] 바뀐 입력(학습 데이터, 라벨 정제 목록, 검수표)을 확인하세요. 2차 val을 바꾸면 이미 돌린 실험과 비교할 수 없습니다.")
            return 1
        return 0
    existed = {path for path, _, _ in outputs if path.is_file()}
    changed = [ds.write_rows(path, columns, rows) for path, columns, rows in outputs]
    for (path, _, _), did in zip(outputs, changed):
        print(f"{shown[path]}: {'썼다' if did else '변경 없음'} ({len(ds.read_rows(path)):,}줄)")
    if any(did and path in existed for (path, _, _), did in zip(outputs, changed)):
        folder = shown[images_csv].parent
        print("[주의] 이미 있던 목록을 바꿔 썼습니다. 2차 val이 바뀌었으면 이미 돌린 실험과 비교할 수 없습니다.\n"
              f"[해결] git diff {folder}로 바뀐 줄을 보고, 의도하지 않았으면 git restore {folder}로 되돌리세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
