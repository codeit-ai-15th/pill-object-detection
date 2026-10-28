"""수정 목록대로 고친 annotation 사본과 변경 기록을 터미널에서 한 번에 만든다.

pill_detection.data.make_fixed_annotations()를 부르고 결과를 요약해 출력하기만 한다.
각 인자의 뜻과 규칙은 그 함수의 docstring을 본다. (src/pill_detection/data.py)
실행 방법과 사용 예시는 README.md의 Scripts 섹션을 본다.
"""

import argparse
import sys
from collections import Counter

from pill_detection.data import make_fixed_annotations

PATH_HINT = """\
[해결] 경로는 지금 터미널이 있는 폴더를 기준으로 찾습니다.
  레포 최상위(pill-object-detection)에서, data/로 시작하는 경로를 넣어 실행했는지 확인하세요.
  노트북에서 쓰던 ../../../data/... 경로는 레포 최상위에서는 맞지 않습니다."""


def parse_args(argv):
    """명령줄 인자를 읽는다. make_fixed_annotations()의 인자 5개를 모두 받는다.

    Args:
        argv (list[str]): 스크립트 이름을 뺀 명령줄 인자.

    Returns:
        argparse.Namespace: fix_csv, base_ann_dir, img_dir, out_ann_dir, change_csv.
    """
    parser = argparse.ArgumentParser(
        description="수정 목록대로 고친 annotation 사본과 변경 기록을 만든다. 경로는 지금 터미널이 있는 폴더 기준이다.",
    )
    parser.add_argument("--fix-csv", required=True,
                        help="수정 목록 CSV. 읽기만 한다. 예: data/label_fix/label_fixes.csv")
    parser.add_argument("--base-ann-dir", required=True,
                        help="기준 annotation 폴더. 읽기만 한다. 1차 가공은 원본. 예: data/raw/train_annotations")
    parser.add_argument("--img-dir", required=True,
                        help="이미지 폴더. 읽기만 한다. 2차 이후에도 원본. 예: data/raw/train_images")
    parser.add_argument("--out-ann-dir", required=True,
                        help="사본 annotation 폴더. 폴더 전체를 지우고 다시 만든다. 예: data/processed/train_annotations")
    parser.add_argument("--change-csv", required=True,
                        help="변경 기록의 1번 파일. 내용이 바뀌면 _v2, _v3 …로 새로 저장한다. 예: data/label_fix/apply_label_fixes.csv")
    return parser.parse_args(argv)


def main(argv):
    """사본과 변경 기록을 만들고 결과를 요약해 출력한다.

    Args:
        argv (list[str]): 스크립트 이름을 뺀 명령줄 인자.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류로 멈춤.
    """
    args = parse_args(argv)
    try:
        result = make_fixed_annotations(
            fix_csv=args.fix_csv,
            base_ann_dir=args.base_ann_dir,
            img_dir=args.img_dir,
            out_ann_dir=args.out_ann_dir,
            change_csv=args.change_csv,
        )
    except FileNotFoundError as e:
        print(f"[오류] {e}\n\n{PATH_HINT}")
        return 1
    except ValueError as e:
        # data.py의 메시지에 어느 줄이 왜 틀렸는지와 해결 방법이 들어 있다
        print(f"[오류] {e}")
        return 1

    counts = Counter(row["action"] for row in result["applied"])
    applied = ", ".join(f"{action} {n}" for action, n in sorted(counts.items()))
    print(f"기준 JSON {result['n_base']}개 → 사본 JSON {result['n_out']}개 ({args.out_ann_dir})")
    print(f"수정 목록 {len(result['applied'])}줄 적용: {applied}")
    print(f"변경 기록: {result['change_csv']} ({len(result['changes'])}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
