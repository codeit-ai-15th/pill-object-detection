"""학습 데이터(Kaggle 원본 + AI Hub 외부 데이터)를 명령 하나로 준비한다.

사용법 (레포 최상위에서)
    uv run scripts/prepare_train_data.py

하는 일
    1. Kaggle 원본(data/raw/train_annotations, train_images)이 있는지 확인한다.
    2. AI Hub zip 16개(허용 조합 라벨 8개, 이미지 8개)를 하나씩 받고, data/external/aihub/raw/에 풀고, zip을 지운다.
        - 이미 다 푼 zip(data/external/aihub/sources.csv에 있고 파일이 모두 있는 zip)은 건너뛴다.
        - 받을 zip이 있을 때만 .env의 AIHUB_APIKEY가 필요하다.
    3. pill_detection.data.make_train_dataset()을 불러 정제 목록(data/label_fix/label_fixes.csv)대로 고친 사본을 만든다.
        - Kaggle 사본: data/processed/train_annotations
        - AI Hub 사본: data/processed/ai_hub/train_annotations, train_images (PNG는 원본의 하드링크)
        - 변경 기록: data/label_fix/apply_label_fixes_v2.csv (Kaggle과 AI Hub를 함께 기록. 같은 정제 목록으로 다시 실행하면 그대로 둔다)
    중간에 멈추면 같은 명령을 다시 실행한다. 끝난 단계는 건너뛴다.

준비
- Kaggle 원본은 README의 데이터 받기를 따라 data/raw/에 둔다.
- AI Hub 576 데이터셋 이용 승인을 받고, 발급받은 API 키를 .env의 AIHUB_APIKEY에 넣는다. (.env.example 참고)
- 디스크: AI Hub 원본 약 28GB + 받는 동안 zip 하나(약 3GB)의 2배 + 여유 2GB. 사본 PNG는 하드링크라 더 들지 않는다.

받기와 풀기만 따로 할 때는 scripts/download_aihub.py, scripts/extract_aihub.py를 쓴다.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import download_aihub  # 같은 scripts/ 폴더의 스크립트 (uv run scripts/...로 실행하면 import 경로에 들어간다)
import extract_aihub

from pill_detection.data import make_train_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_ANN_DIR = DATA_DIR / "raw" / "train_annotations"
KAGGLE_IMG_DIR = DATA_DIR / "raw" / "train_images"
AIHUB_RAW_DIR = extract_aihub.RAW_DIR
AIHUB_SOURCES_CSV = extract_aihub.SOURCES_CSV

# 받을 AI Hub zip (filekey, 파일 이름). 라벨을 먼저 받는다. 금지 파일(TL_2, TS_2)은 넣지 않는다
# 목록: https://api.aihub.or.kr/info/576.do
AIHUB_ZIPS = [
    (66065, "TL_1_조합.zip"), (66067, "TL_3_조합.zip"), (66068, "TL_4_조합.zip"), (66069, "TL_5_조합.zip"),
    (66070, "TL_6_조합.zip"), (66071, "TL_7_조합.zip"), (66072, "TL_8_조합.zip"), (66243, "VL_1_조합.zip"),
    (66154, "TS_1_조합.zip"), (66156, "TS_3_조합.zip"), (66157, "TS_4_조합.zip"), (66158, "TS_5_조합.zip"),
    (66159, "TS_6_조합.zip"), (66160, "TS_7_조합.zip"), (66161, "TS_8_조합.zip"), (66254, "VS_1_조합.zip"),
]


def zips_to_fetch(zips, raw_dir=AIHUB_RAW_DIR, sources_csv=AIHUB_SOURCES_CSV):
    """아직 다 풀지 않은 zip을 고른다. sources_csv에 있고, 거기 적힌 파일이 raw_dir에 모두 있으면 다 푼 zip이다.

    Args:
        zips (list[tuple[int, str]]): (filekey, 파일 이름) 목록.
        raw_dir (Path): AI Hub 원본 폴더. (data/external/aihub/raw)
        sources_csv (Path): extract_aihub.py가 쓰는 출처 기록.

    Returns:
        list[tuple[int, str]]: 받고 풀어야 하는 zip. 순서는 zips 그대로다.
    """
    sources = extract_aihub.read_sources(sources_csv)
    done = {name for name, paths in sources.items() if all((Path(raw_dir) / p).is_file() for p in paths)}
    return [(key, name) for key, name in zips if name not in done]


def fetch_aihub(zips):
    """zip을 하나씩 받고, 풀고, 지운다. 디스크를 아끼려고 다음 zip은 앞 zip을 지운 뒤에 받는다.

    Args:
        zips (list[tuple[int, str]]): zips_to_fetch()의 결과.

    Returns:
        int: 종료 코드. 0이면 성공.
    """
    for i, (key, name) in enumerate(zips, 1):
        print(f"\n[AI Hub {i}/{len(zips)}] {name} (filekey {key})")
        code = download_aihub.main([str(key), "--out", str(AIHUB_RAW_DIR)])
        if code:
            return code
        path = download_aihub.target_path(AIHUB_RAW_DIR, name)
        if not path.is_file():
            print(f"[오류] 받은 파일을 찾지 못했습니다: {path.relative_to(PROJECT_ROOT)}\n"
                  "[해결] AI Hub 목록의 filekey와 파일 이름이 바뀌었는지 확인하고, scripts/prepare_train_data.py의 AIHUB_ZIPS를 고치세요.")
            return 1
        code = extract_aihub.main([str(path)])
        if code:
            return code
    return 0


def main(argv=None):
    """Kaggle 원본을 확인하고, AI Hub를 받고 풀고, 학습 데이터 사본을 만든다.

    Args:
        argv (list[str]): 명령줄 인자. 받는 인자는 없고 --help만 있다. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    argparse.ArgumentParser(
        description="학습 데이터(Kaggle + AI Hub)를 준비한다. AI Hub zip을 받고 풀고, 정제 목록대로 고친 사본과 변경 기록을 만든다. "
                    "인자는 없다. 자세한 설명은 이 파일 맨 위와 README의 학습 데이터 준비를 본다.",
    ).parse_args(argv)
    missing = [p.relative_to(PROJECT_ROOT).as_posix() for p in (KAGGLE_ANN_DIR, KAGGLE_IMG_DIR) if not p.is_dir()]
    if missing:
        print(f"[오류] Kaggle 원본 폴더가 없습니다: {', '.join(missing)}\n"
              "[해결] README의 데이터 받기를 따라 대회 데이터를 data/raw/에 풀고 다시 실행하세요.")
        return 1

    todo = zips_to_fetch(AIHUB_ZIPS)
    print(f"[AI Hub] zip {len(AIHUB_ZIPS)}개 중 다 푼 것 {len(AIHUB_ZIPS) - len(todo)}개, 받을 것 {len(todo)}개")
    code = fetch_aihub(todo)
    if code:
        return code

    print("\n[정제] 정제 목록대로 Kaggle 사본과 AI Hub 사본을 만듭니다. (몇 분 걸립니다)")
    try:
        result = make_train_dataset(
            fix_csv=DATA_DIR / "label_fix" / "label_fixes.csv",
            kaggle_ann_dir=KAGGLE_ANN_DIR,
            kaggle_img_dir=KAGGLE_IMG_DIR,
            aihub_ann_dir=AIHUB_RAW_DIR / "train_annotations",
            aihub_img_dir=AIHUB_RAW_DIR / "train_images",
            kaggle_out_ann_dir=DATA_DIR / "processed" / "train_annotations",
            aihub_out_dir=DATA_DIR / "processed" / "ai_hub",
            change_csv=DATA_DIR / "label_fix" / "apply_label_fixes.csv",
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[오류] {error}")
        return 1

    kaggle, aihub = result["kaggle"], result["aihub"]
    actions = Counter((change["source"], change["action"]) for change in result["changes"])
    print(f"Kaggle: JSON {kaggle['n_base']:,}개 → {kaggle['n_out']:,}개 (data/processed/train_annotations)")
    print(f"AI Hub: 이미지 {aihub['n_images']:,}장, JSON {aihub['n_json']:,}개, 이름을 고친 이미지 {len(aihub['renamed'])}장 "
          f"(data/processed/ai_hub). PNG 하드링크 {aihub['linked']:,}장, 복사 {aihub['copied']:,}장")
    print(f"변경 기록: {result['change_csv'].relative_to(PROJECT_ROOT)} "
          + ", ".join(f"{source} {action} {n:,}" for (source, action), n in sorted(actions.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
