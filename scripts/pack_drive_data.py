"""만든 학습 데이터를 Google Drive에 올릴 zip 4개로 묶는다.

scripts/prepare_train_data.py로 학습 데이터를 만든 사람이 한 번 실행한다. 다른 팀원은 zip을 받아 레포 최상위에서 푼다.

사용법 (레포 최상위에서)
    uv run scripts/pack_drive_data.py --out <레포 밖 폴더> [--only <zip 이름> ...]

    # zip 4개를 모두 만든다
    uv run scripts/pack_drive_data.py --out ~/pill_drive
    # 중간에 멈췄으면 남은 것만 만든다
    uv run scripts/pack_drive_data.py --out ~/pill_drive --only aihub_images test_images

만드는 zip (zip 안 경로는 레포 최상위 기준 data/...이고, 구분자는 OS와 상관없이 /이다)
    kaggle.zip             data/raw/train_images (PNG 232장) + data/processed/train_annotations (JSON 771개)
    aihub_annotations.zip  data/processed/ai_hub/train_annotations (JSON 45,819개)
    aihub_images.zip       data/processed/ai_hub/train_images (PNG 11,974장, 약 21.6GB)
    test_images.zip        data/raw/test_images (PNG 842장)
- 폴더의 파일 수가 위 기대값과 다르면 zip을 만들지 않는다. 학습 데이터를 다시 만들라는 뜻이다.
- PNG는 이미 압축된 형식이라 압축하지 않고 넣는다. JSON은 압축한다. 4GB가 넘는 zip(zip64)도 만든다.
- macOS가 만든 파일(.DS_Store, ._로 시작하는 파일)은 넣지 않는다.
- zip은 <이름>.zip.part로 쓰고 다 쓴 뒤에 이름을 바꾼다. .zip이 있으면 다 만든 zip이다.
- zip은 레포 밖에만 쓴다. 레포 안에 두면 git이나 IDE가 수십 GB를 훑는다.

디스크: zip 합계 약 23.5GB (aihub_images.zip 약 21.6GB, test_images.zip 약 1.5GB)

푸는 법 (레포를 clone한 폴더에서. 경로가 data/...로 시작하므로 그 자리에 풀린다)
    unzip -q <zip 폴더>/kaggle.zip
"""

import argparse
import sys
import unicodedata
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# zip 이름 → [(폴더(레포 최상위 기준), 확장자, 기대 파일 수)]. 기대값은 README 학습 데이터 준비의 결과와 같다
PACKS = {
    "kaggle": [("data/raw/train_images", ".png", 232), ("data/processed/train_annotations", ".json", 771)],
    "aihub_annotations": [("data/processed/ai_hub/train_annotations", ".json", 45819)],
    "aihub_images": [("data/processed/ai_hub/train_images", ".png", 11974)],
    "test_images": [("data/raw/test_images", ".png", 842)],
}
# 압축하지 않고 넣는 확장자. 이미 압축된 형식이라 압축해도 줄지 않고 시간만 든다
STORED_SUFFIXES = {".png"}
PROGRESS_EVERY = 2000


def list_files(root, folder, suffix):
    """폴더 아래(하위 폴더 포함)에서 확장자가 맞는 파일을 고른다. macOS가 만든 숨김 파일은 뺀다.

    Args:
        root (Path): 레포 최상위. zip 안 경로의 기준이다.
        folder (str): root 기준 폴더. (예: data/raw/train_images)
        suffix (str): 고를 확장자. (예: .png) 대소문자는 가리지 않는다.

    Returns:
        list[Path]: 파일 경로. zip 안 경로 순서로 정렬한다.
    """
    files = [p for p in (Path(root) / folder).rglob("*")
             if p.is_file() and p.suffix.lower() == suffix and not p.name.startswith(".")]
    return sorted(files, key=lambda p: arcname(root, p))


def arcname(root, path):
    """zip 안 경로를 만든다. 구분자는 /이고, 한글은 NFC로 맞춘다.

    Args:
        root (Path): 레포 최상위.
        path (Path): 넣을 파일.

    Returns:
        str: root 기준 상대경로. (예: data/raw/train_images/1.png)
    """
    return unicodedata.normalize("NFC", Path(path).relative_to(root).as_posix())


def collect(root, names, packs=PACKS):
    """zip마다 넣을 파일을 모으고, 폴더가 없거나 파일 수가 기대값과 다르면 문제로 적는다.

    Args:
        root (Path): 레포 최상위.
        names (list[str]): 만들 zip 이름. (PACKS의 키)
        packs (dict): zip 이름 → [(폴더, 확장자, 기대 파일 수)]. 테스트에서 바꿔 넣는다.

    Returns:
        tuple[dict[str, list[Path]], list[str]]: (zip 이름 → 넣을 파일, 문제 목록). 문제가 없으면 빈 목록이다.
    """
    members, problems = {}, []
    for name in names:
        members[name] = []
        for folder, suffix, expected in packs[name]:
            if not (Path(root) / folder).is_dir():
                problems.append(f"{name}.zip: 폴더가 없습니다: {folder}")
                continue
            files = list_files(root, folder, suffix)
            if len(files) != expected:
                problems.append(f"{name}.zip: {folder}의 {suffix} 파일이 {len(files):,}개입니다 (기대값 {expected:,}개)")
            members[name].extend(files)
    return members, problems


def write_zip(root, files, zip_path):
    """파일을 zip 하나로 쓴다. <이름>.part에 다 쓴 뒤 이름을 바꾸므로, 중간에 멈추면 .zip이 생기지 않는다.

    Args:
        root (Path): 레포 최상위. zip 안 경로의 기준이다.
        files (list[Path]): 넣을 파일.
        zip_path (Path): 만들 zip.

    Returns:
        int: zip에 넣은 파일 수.
    """
    zip_path = Path(zip_path)
    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for i, path in enumerate(files, 1):
            stored = path.suffix.lower() in STORED_SUFFIXES
            zf.write(path, arcname(root, path), compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            if i % PROGRESS_EVERY == 0:
                print(f"  {i:,}/{len(files):,}", flush=True)
    with zipfile.ZipFile(part) as zf:
        n = len(zf.namelist())
    if n != len(files):
        raise RuntimeError(f"{part}에 파일이 {n:,}개만 들어갔습니다 (넣은 파일 {len(files):,}개)")
    part.replace(zip_path)
    return n


def main(argv=None, root=PROJECT_ROOT, packs=PACKS):
    """학습 데이터 폴더의 파일 수를 확인하고 zip을 만든다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.
        root (Path): 레포 최상위. 테스트에서 바꿔 넣는다.
        packs (dict): zip 이름 → [(폴더, 확장자, 기대 파일 수)]. 테스트에서 바꿔 넣는다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류.
    """
    parser = argparse.ArgumentParser(
        description="학습 데이터를 Google Drive에 올릴 zip으로 묶는다. 자세한 설명은 이 파일 맨 위를 본다.")
    parser.add_argument("--out", required=True, type=Path, help="zip을 쓸 폴더. 레포 밖이어야 한다.")
    parser.add_argument("--only", nargs="+", choices=list(packs), help="이 zip만 만든다. 기본은 전부.")
    args = parser.parse_args(argv)

    root = Path(root).resolve()
    out = args.out.expanduser().resolve()
    if out == root or root in out.parents:
        print(f"[오류] zip을 레포 안에 쓰려고 했습니다: {out}\n"
              "[해결] --out에 레포 밖 폴더를 주세요. 예: --out ~/pill_drive")
        return 1
    names = args.only or list(packs)
    existing = [f"{name}.zip" for name in names if (out / f"{name}.zip").exists()]
    if existing:
        print(f"[오류] 이미 있는 zip이 있습니다: {', '.join(existing)} ({out})\n"
              "[해결] 다시 만들려면 그 zip을 지우고, 남은 zip만 만들려면 --only로 이름을 주세요.")
        return 1

    members, problems = collect(root, names, packs)
    if problems:
        print("[오류] 학습 데이터가 기대한 것과 다릅니다.\n  " + "\n  ".join(problems) + "\n"
              "[해결] uv run scripts/prepare_train_data.py를 다시 실행하고, 요약이 README 학습 데이터 준비와 같은지 확인하세요.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        zip_path = out / f"{name}.zip"
        print(f"[{name}.zip] 파일 {len(members[name]):,}개를 묶습니다.", flush=True)
        n = write_zip(root, members[name], zip_path)
        print(f"[{name}.zip] 완료: 파일 {n:,}개, {zip_path.stat().st_size / 2**30:.2f}GB")
    print(f"\n{out}의 zip을 Drive 공유 폴더에 올리세요. 받은 사람은 레포 최상위에서 unzip -q <zip>으로 풉니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
