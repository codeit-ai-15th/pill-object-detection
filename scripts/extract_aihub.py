"""scripts/download_aihub.py로 받은 AI Hub 조합 zip을 data/raw와 같은 구조로 풀고, 다 풀면 zip을 지운다.

학습 데이터를 준비할 때는 이 스크립트를 직접 쓰지 않고 scripts/prepare_train_data.py를 실행한다.

사용법
    uv run scripts/extract_aihub.py [<zip> ...] [--keep-zip]

    # 받아 둔 조합 zip을 모두 풀고 지운다
    uv run scripts/extract_aihub.py
    # 이미지 zip 하나를 풀고, zip은 남긴다
    uv run scripts/extract_aihub.py data/external/aihub/raw/train_images/TS_1_조합.zip --keep-zip

푸는 위치 (zip이 있던 폴더)
    data/external/aihub/
    ├── raw/
    │   ├── train_annotations/   # 라벨 zip(TL_*, VL_*)의 원본 JSON. zip 안 폴더 구조 그대로
    │   └── train_images/        # 이미지 zip(TS_*, VS_*)의 PNG. 하위 폴더 없이 한 폴더에
    └── sources.csv              # zip마다 나온 파일 (zip, path). path는 raw/ 기준이다
- sources.csv는 zip 하나를 다 푼 뒤에 쓴다. 그래서 zip을 지운 뒤에도 어느 zip을 다 풀었는지, 파일이 어느 zip에서 왔는지 알 수 있다.
- macOS가 만든 파일(._로 시작하는 파일, .DS_Store, __MACOSX/)과 이미지 zip의 PNG가 아닌 파일은 풀지 않는다.
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
RAW_DIR = AIHUB_DIR / "raw"
SOURCES_CSV = AIHUB_DIR / "sources.csv"
SUBDIRS = {"label": "train_annotations", "image": "train_images"}
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
    """macOS가 zip에 넣은 부가 파일(._로 시작하는 파일, .DS_Store, __MACOSX/ 아래)인지 본다.

    Args:
        name (str): zip 안의 경로.

    Returns:
        bool: 부가 파일이면 True.
    """
    base = PurePosixPath(name).name
    return base.startswith("._") or base == ".DS_Store" or "__MACOSX" in PurePosixPath(name).parts


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
                           "  scripts/download_aihub.py로 받은 파일 이름을 바꾸지 말고 그대로 주세요.")
    return "label" if name[1] == "L" else "image"


def member_path(name, kind):
    """zip 안의 경로를 raw/ 기준 경로로 바꾼다. 라벨은 폴더 구조를 그대로 두고, 이미지는 파일 이름만 쓴다.

    Args:
        name (str): zip 안의 경로 (NFC).
        kind (str): "label" 또는 "image".

    Returns:
        str: raw/ 기준 경로(구분자 /). 예: train_images/K-..._0_2_0_2_70_000_200.png

    Raises:
        ExtractError: 절대 경로나 ..이 들어 있어 풀 폴더 밖으로 나가는 경로일 때.
    """
    parts = PurePosixPath(name).parts
    if PurePosixPath(name).is_absolute() or ".." in parts:
        raise ExtractError(f"[오류] zip에 안전하지 않은 경로가 있어 멈췄습니다: {name}\n"
                           "  AI Hub에서 받은 파일이 아닐 수 있습니다. 이 zip을 지우고 다시 받으세요.")
    if kind == "label":
        return "/".join((SUBDIRS[kind], *parts))
    return f"{SUBDIRS[kind]}/{parts[-1]}"


def extract_zip(path, raw_dir=RAW_DIR, sources_csv=SOURCES_CSV, keep_zip=False):
    """zip 하나를 raw_dir에 풀고, 나온 파일을 sources_csv에 적고, zip을 지운다. 같은 크기로 이미 풀린 파일은 건너뛴다.

    Args:
        path (Path): 풀 zip 경로.
        raw_dir (Path): 풀 폴더. 라벨은 raw_dir/train_annotations, 이미지는 raw_dir/train_images에 푼다.
        sources_csv (Path): 출처 기록 CSV. 같은 zip의 옛 줄은 새 줄로 바꾼다.
        keep_zip (bool): True면 다 푼 뒤에도 zip을 지우지 않는다.

    Returns:
        tuple[int, int, int]: (새로 푼 파일 수, 건너뛴 파일 수, 풀지 않은 부가 파일 수).

    Raises:
        ValueError: 대회에서 금지한 파일일 때.
        ExtractError: 경로가 안전하지 않거나, 이미지 zip에 같은 파일 이름이 두 번 있을 때.
    """
    path, raw_dir = Path(path), Path(raw_dir)
    check_aihub_file(path)
    kind = zip_kind(path)
    written = skipped = ignored = 0
    seen = {}  # raw/ 기준 경로 → zip 안 이름
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = nfc(info.filename)
            if info.is_dir():
                continue
            if is_macos_file(name) or (kind == "image" and not name.lower().endswith(".png")):
                ignored += 1
                continue
            rel = member_path(name, kind)
            if rel in seen:
                raise ExtractError(f"[오류] {nfc(path.name)} 안에 같은 이름의 파일이 두 번 있습니다: {seen[rel]}, {name}\n"
                                   "  이미지는 한 폴더에 풀기 때문에 이름이 겹치면 하나가 덮어써집니다. 이 zip은 다 풀지 않았습니다.")
            seen[rel] = name
            target = raw_dir / rel
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            with zf.open(info) as src, open(partial, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            os.replace(partial, target)
            written += 1
    save_sources(sources_csv, nfc(path.name), sorted(seen))
    if not keep_zip:
        path.unlink()
    return written, skipped, ignored


def read_sources(sources_csv=SOURCES_CSV):
    """출처 기록을 읽는다.

    Args:
        sources_csv (Path): 출처 기록 CSV.

    Returns:
        dict[str, list[str]]: zip 이름 → 그 zip에서 나온 파일의 raw/ 기준 경로. 파일이 없으면 빈 dict.
    """
    sources = {}
    if Path(sources_csv).is_file():
        with open(sources_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sources.setdefault(row["zip"], []).append(row["path"])
    return sources


def save_sources(sources_csv, zip_name, paths):
    """zip 하나에서 나온 파일을 출처 기록에 적는다. 같은 zip의 옛 줄은 바꾸고, 다른 zip의 줄은 그대로 둔다.

    Args:
        sources_csv (Path): 출처 기록 CSV.
        zip_name (str): zip 이름. 예: TS_1_조합.zip
        paths (list[str]): 그 zip에서 나온 파일의 raw/ 기준 경로.
    """
    sources = read_sources(sources_csv)
    sources[zip_name] = list(paths)
    rows = sorted((name, p) for name, ps in sources.items() for p in ps)
    Path(sources_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(sources_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["zip", "path"])
        writer.writerows(rows)


def main(argv=None):
    """주어진 zip(없으면 data/external/aihub/raw/의 조합 zip 전부)을 풀고 지운다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    parser = argparse.ArgumentParser(description="AI Hub 조합 zip을 data/raw와 같은 구조로 풀고, 다 풀면 zip을 지운다.")
    parser.add_argument("zips", nargs="*", type=Path, help="풀 zip. 비우면 data/external/aihub/raw/의 조합 zip을 모두 푼다.")
    parser.add_argument("--keep-zip", action="store_true", help="다 푼 zip을 지우지 않는다.")
    args = parser.parse_args(argv)
    zips = args.zips or sorted(p for sub in SUBDIRS.values() for p in (RAW_DIR / sub).glob("*.zip"))
    if not zips:
        print(f"[오류] 풀 zip이 없습니다: {RAW_DIR.relative_to(PROJECT_ROOT)}\n"
              "[해결] 먼저 scripts/download_aihub.py로 받으세요. 학습 데이터를 준비하는 것이면 scripts/prepare_train_data.py를 실행하세요.")
        return 1
    try:
        for path in zips:
            written, skipped, ignored = extract_zip(path, keep_zip=args.keep_zip)
            out = RAW_DIR / SUBDIRS[zip_kind(path)]
            print(f"{nfc(path.name)} → {out.relative_to(PROJECT_ROOT)}: 새로 푼 파일 {written:,}개, "
                  f"이미 있어 건너뜀 {skipped:,}개, macOS 부가 파일 등 뺀 것 {ignored:,}개"
                  + ("" if args.keep_zip else ", zip 지움"))
        return 0
    except (ExtractError, ValueError) as error:
        print(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
