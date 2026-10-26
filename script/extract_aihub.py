"""script/download_aihub.py로 받은 AI Hub 조합 zip을 data/raw와 같은 구조로 푼다.

사용법
    uv run python script/extract_aihub.py [<zip> ...] [--delete-images]

    # 받은 조합 zip을 모두 푼다
    uv run python script/extract_aihub.py
    # 이미지 zip 하나를 풀고 지운다 (디스크가 빠듯할 때 하나씩 받고 풀고 지운다)
    uv run python script/extract_aihub.py data/external/aihub/train_images/TS_1_조합.zip --delete-images

푸는 위치
    data/external/
    ├── aihub_raw/train_annotations/       # 라벨 zip(TL_*, VL_*)의 원본 JSON. zip 안 폴더 구조 그대로
    └── aihub_converted/
        ├── train_images/                  # 이미지 zip(TS_*, VS_*)의 PNG. 하위 폴더 없이 한 폴더에
        └── train_images_sources.csv       # 이미지마다 나온 zip 이름. zip을 지운 뒤에도 출처를 알 수 있게 남긴다
- 원본 JSON을 Kaggle 형식으로 바꾼 결과는 pill_detection.data.convert_aihub_to_kaggle이
  aihub_converted/train_annotations/에 만든다. (notebooks/eda/01_raewoo_external_aihub_eda.ipynb 1-3)
- macOS가 만든 파일(._로 시작하는 파일, .DS_Store)은 풀지 않는다.
- 이미 같은 크기로 풀린 파일은 건너뛴다. 중간에 멈췄으면 같은 명령을 다시 실행한다.
- 대회에서 금지한 TL_2_조합.zip, TS_2_조합.zip은 풀지 않는다. (RULE.md, pill_detection.data.check_aihub_file)
"""

import argparse
import csv
import os
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

from pill_detection.data import check_aihub_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIHUB_DIR = PROJECT_ROOT / "data" / "external" / "aihub"
RAW_ANN_DIR = PROJECT_ROOT / "data" / "external" / "aihub_raw" / "train_annotations"
IMG_DIR = PROJECT_ROOT / "data" / "external" / "aihub_converted" / "train_images"
SOURCES_CSV = IMG_DIR.parent / "train_images_sources.csv"
CHUNK = 2**20


class ExtractError(Exception):
    """원인과 해결 방법을 담아 사용자에게 보여 줄 오류."""


def nfc(text):
    """한글 파일명을 NFC로 맞춘다. macOS는 자모 분리형(NFD)으로 줄 때가 있다.

    Args:
        text (str): 파일 이름.

    Returns:
        str: NFC로 바꾼 이름.
    """
    return unicodedata.normalize("NFC", text)


def is_macos_file(name):
    """macOS가 zip에 넣은 부가 파일(._로 시작하는 파일, .DS_Store)인지 본다.

    Args:
        name (str): zip 안의 경로.

    Returns:
        bool: 부가 파일이면 True.
    """
    base = PurePosixPath(name).name
    return base.startswith("._") or base == ".DS_Store" or "__MACOSX" in PurePosixPath(name).parts


def member_target(name, kind):
    """zip 안의 경로를 풀 위치로 바꾼다. 라벨은 폴더 구조를 그대로 두고, 이미지는 파일 이름만 쓴다.

    Args:
        name (str): zip 안의 경로 (NFC).
        kind (str): "label" 또는 "image".

    Returns:
        Path: 풀 위치.

    Raises:
        ExtractError: 절대 경로나 ..이 들어 있어 풀 폴더 밖으로 나가는 경로일 때.
    """
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).is_absolute() or ".." in parts:
        raise ExtractError(f"[오류] zip에 안전하지 않은 경로가 있어 멈췄습니다: {name}\n"
                           "  AI Hub에서 받은 파일이 아닐 수 있습니다. 이 zip을 지우고 다시 받으세요.")
    if kind == "label":
        return RAW_ANN_DIR.joinpath(*parts)
    return IMG_DIR / parts[-1]


def zip_kind(path):
    """AI Hub 파일 이름의 둘째 글자로 라벨 zip인지 이미지 zip인지 정한다. (TL/VL은 라벨, TS/VS는 이미지)

    Args:
        path (Path): zip 경로.

    Returns:
        str: "label" 또는 "image".

    Raises:
        ExtractError: 조합 zip 이름 형식이 아닐 때.
    """
    name = nfc(path.name)
    if len(name) < 2 or name[0] not in "TV" or name[1] not in "LS" or not name.endswith("_조합.zip"):
        raise ExtractError(f"[오류] {name}은 AI Hub 조합 zip 이름(TL/TS/VL/VS_<번호>_조합.zip)이 아닙니다.\n"
                           "  script/download_aihub.py로 받은 파일 이름을 바꾸지 말고 그대로 주세요.")
    return "label" if name[1] == "L" else "image"


def extract_zip(path):
    """zip 하나를 푼다. 같은 크기로 이미 풀린 파일은 건너뛴다.

    Args:
        path (Path): 풀 zip 경로.

    Returns:
        tuple[int, int, int]: (새로 푼 파일 수, 건너뛴 파일 수, macOS 부가 파일이라 뺀 수).

    Raises:
        ValueError: 대회에서 금지한 파일일 때.
        ExtractError: 경로가 안전하지 않거나, 이미지 zip에 같은 파일 이름이 두 번 있을 때.
    """
    check_aihub_file(path)
    kind = zip_kind(path)
    written = skipped = ignored = 0
    seen = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = nfc(info.filename)
            if info.is_dir():
                continue
            if is_macos_file(name):
                ignored += 1
                continue
            if kind == "image" and not name.lower().endswith(".png"):
                ignored += 1
                continue
            target = member_target(name, kind)
            if target in seen:
                raise ExtractError(f"[오류] {nfc(path.name)} 안에 같은 이름의 이미지가 두 번 있습니다: {seen[target]}, {name}\n"
                                   "  이미지는 한 폴더에 풀기 때문에 이름이 겹치면 하나가 덮어써집니다. 이 zip은 풀지 않았습니다.")
            seen[target] = name
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with zf.open(info) as src, open(partial, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            os.replace(partial, target)
            written += 1
    if kind == "image":
        save_sources(nfc(path.name), sorted(target.name for target in seen))
    return written, skipped, ignored


def save_sources(zip_name, file_names):
    """이미지 zip 하나에서 나온 파일 이름을 SOURCES_CSV에 적는다. 같은 zip의 옛 줄은 바꾼다.

    Args:
        zip_name (str): 이미지 zip 이름. 예: TS_1_조합.zip
        file_names (list[str]): 그 zip에서 푼 PNG 파일 이름.
    """
    rows = []
    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["image_zip"] != zip_name]
    rows += [{"image_zip": zip_name, "file_name": name} for name in file_names]
    SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_zip", "file_name"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["image_zip"], row["file_name"])))


def main(argv=None):
    """주어진 zip(없으면 data/external/aihub/의 조합 zip 전부)을 푼다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    parser = argparse.ArgumentParser(description="AI Hub 조합 zip을 data/raw와 같은 구조로 푼다.")
    parser.add_argument("zips", nargs="*", type=Path, help="풀 zip. 비우면 data/external/aihub/의 조합 zip을 모두 푼다.")
    parser.add_argument("--delete-images", action="store_true",
                        help="다 푼 이미지 zip(TS_*, VS_*)을 지운다. 라벨 zip은 노트북이 직접 읽으므로 지우지 않는다.")
    args = parser.parse_args(argv)
    zips = args.zips or sorted(p for sub in ("train_annotations", "train_images")
                               for p in (AIHUB_DIR / sub).glob("*.zip"))
    if not zips:
        print(f"[오류] 풀 zip이 없습니다: {AIHUB_DIR}\n"
              "[해결] 먼저 script/download_aihub.py로 받으세요. (README의 AI Hub 외부 데이터 받기)")
        return 1
    try:
        for path in zips:
            written, skipped, ignored = extract_zip(path)
            kind = zip_kind(path)
            out = RAW_ANN_DIR if kind == "label" else IMG_DIR
            print(f"{nfc(path.name)} → {out.relative_to(PROJECT_ROOT)}: 새로 푼 파일 {written:,}개, "
                  f"이미 있어 건너뜀 {skipped:,}개, macOS 부가 파일 등 뺀 것 {ignored:,}개")
            if args.delete_images and kind == "image":
                path.unlink()
                print(f"  지움: {path}")
        return 0
    except (ExtractError, ValueError) as error:
        print(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
