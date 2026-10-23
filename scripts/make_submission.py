"""test 이미지 검출 결과를 대회 제출 CSV로 저장한다.

    uv run python scripts/make_submission.py --weights runs/yolov8s_640/weights/best.pt
    → submissions/yolov8s_640.csv

대회 형식: 1행 = 검출 객체 1개. image_id는 이미지 파일 이름의 숫자, annotation_id는 행마다 다른 값.
category_id는 classes.py로 바꾼 원본 category_id다. (주최측 확인 전 가정)
"""

import argparse
import csv
import sys
from pathlib import Path

from pill_detection.predict import predict

COLUMNS = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]


def parse_args():
    """명령줄 옵션을 읽는다.

    Returns:
        argparse.Namespace: weights, test_dir, out, conf.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="예) runs/<실험 이름>/weights/best.pt")
    parser.add_argument("--test-dir", default="data/raw/test_images")
    parser.add_argument("--out", help="기본값: submissions/<실험 이름>.csv")
    parser.add_argument("--conf", type=float, default=0.001, help="mAP는 낮은 점수의 검출도 채점에 쓰므로 낮게 둔다")
    return parser.parse_args()


def main():
    """test 이미지를 모두 검출해 대회 제출 CSV로 저장한다.

    Returns:
        int: 종료 코드. 0 성공, 1 가중치 파일이나 test 이미지가 없을 때.
    """
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[오류] 가중치 파일이 없습니다: {weights}")
        return 1
    images = sorted(Path(args.test_dir).glob("*.png"), key=lambda p: int(p.stem))
    if not images:
        print(f"[오류] {args.test_dir}에 png 이미지가 없습니다. --test-dir를 확인하세요.")
        return 1

    # runs/<실험 이름>/weights/best.pt → submissions/<실험 이름>.csv
    out = Path(args.out) if args.out else Path("submissions") / f"{weights.parent.parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    annotation_id = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for i, path in enumerate(images, 1):
            for det in predict(path, weights=weights, conf=args.conf):
                annotation_id += 1
                x, y, w, h = det["bbox"]
                writer.writerow([annotation_id, int(path.stem), det["category_id"],
                                 round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(det["score"], 5)])
            if i % 100 == 0:
                print(f"  {i}/{len(images)}장 처리")

    print(f"[완료] 이미지 {len(images)}장, 검출 {annotation_id}개 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
