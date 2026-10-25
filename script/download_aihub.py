"""AI Hub 576 데이터셋(경구약제 이미지)에서 filekey로 고른 zip을 받는다.

사용법
    uv run python script/download_aihub.py <filekey> [<filekey> ...] [--out <폴더>]

    # EDA 노트북(notebooks/eda/00_raewoo_first_eda.ipynb)에 필요한 허용 조합 라벨 zip 8개 (합계 약 59MB)
    uv run python script/download_aihub.py 66065 66067 66068 66069 66070 66071 66072 66243

준비
- AI Hub에서 576 데이터셋 이용 승인을 받고, 발급받은 API 키를 .env의 AIHUB_APIKEY에 넣는다. (.env.example 참고)
- 파일 이름, 용량, filekey 목록: https://api.aihub.or.kr/info/576.do

폴더 규칙 (data/raw처럼 둔다. zip은 풀지 않고 AI Hub 파일 이름 그대로 둔다)
    data/external/aihub/
    ├── train_annotations/   # 라벨 zip: TL_*, VL_* (파일 이름 둘째 글자가 L)
    └── train_images/        # 이미지 zip: TS_*, VS_* (파일 이름 둘째 글자가 S)
- zip을 푼 결과는 data/raw와 같은 구조로 data/external/aihub_converted/에 둔다. (이 스크립트는 풀지 않는다)
- 대회에서 금지한 TL_2_조합.zip, TS_2_조합.zip은 받지 않는다. (RULE.md, pill_detection.data.check_aihub_file)
- 이미 있는 파일은 다시 받지 않는다.

aihubshell을 쓰지 않는 이유
- aihubshell의 part 병합 단계는 macOS 기본 bash(3.2)에서 한글 파일명을 찾지 못해 빈 zip을 만들고 part 파일까지 지운다.
- 이 스크립트는 같은 API로 tar를 받고, tar 안의 part 파일을 직접 이어 붙인다.
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from pill_detection.data import check_aihub_file

DATASET_KEY = 576
FILE_LIST_URL = f"https://api.aihub.or.kr/info/{DATASET_KEY}.do"
DOWNLOAD_URL = f"https://api.aihub.or.kr/down/0.6/{DATASET_KEY}.do"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "external" / "aihub"
SUBDIRS = {"L": "train_annotations", "S": "train_images"}  # 파일 이름 둘째 글자 → 하위 폴더

NAME_PATTERN = re.compile(r"[TV][LS]_\d+_(조합|단일)\.zip")
LIST_LINE = re.compile(r"([^\s─├└│]+\.zip) \| ([\d.]+) (KB|MB|GB|TB) \| (\d+)")  # 예: ├─TL_1_조합.zip | 8 MB | 66065
PART_NAME = re.compile(r"(.+\.zip)\.part(\d+)")
UNITS = {"KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}

DISK_MARGIN = 2 * 2**30  # 받는 동안 남겨 둘 여유 공간 (2GiB)
CHUNK = 2**20


class DownloadError(Exception):
    """원인과 해결 방법을 담아 사용자에게 보여 줄 오류."""


def nfc(text):
    """한글 파일명을 NFC로 맞춘다. macOS는 자모 분리형(NFD)으로 줄 때가 있다.

    Args:
        text (str): 파일 이름.

    Returns:
        str: NFC로 바꾼 이름.
    """
    return unicodedata.normalize("NFC", text)


def parse_filekeys(values):
    """명령줄의 filekey를 정수 목록으로 바꾼다. 공백과 쉼표 모두로 나눌 수 있다.

    Args:
        values (list[str]): 명령줄 인자. 예: ["66065,66067", "66243"]

    Returns:
        list[int]: 중복을 뺀 filekey. 입력 순서를 유지한다.

    Raises:
        DownloadError: 숫자가 아닌 값이 있을 때.
    """
    keys = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise DownloadError(f"[오류] filekey는 숫자여야 합니다: {part!r}\n"
                                    f"  filekey 목록: {FILE_LIST_URL}")
            if int(part) not in keys:
                keys.append(int(part))
    return keys


def fetch_file_list():
    """AI Hub 파일 목록에서 filekey별 파일 이름과 용량을 읽는다. API 키가 필요 없다.

    Returns:
        dict[int, tuple[str, int]]: filekey → (파일 이름, 대략 용량 바이트). 용량은 목록의 반올림 값이다.

    Raises:
        DownloadError: 목록을 받지 못했거나 읽을 줄이 없을 때.
    """
    status = 200
    try:
        with urllib.request.urlopen(FILE_LIST_URL, timeout=60) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        # AI Hub는 목록 본문을 정상으로 보내면서 상태 코드를 502로 줄 때가 있다. 본문에 목록이 있으면 그대로 쓴다
        status = error.code
        text = error.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as error:
        raise DownloadError(f"[오류] AI Hub 파일 목록을 받지 못했습니다: {error}\n"
                            f"  인터넷 연결을 확인하고 다시 실행하세요. 목록 주소: {FILE_LIST_URL}") from error
    files = {}
    for name, size, unit, key in LIST_LINE.findall(text):
        files[int(key)] = (nfc(name), int(float(size) * UNITS[unit]))
    if not files:
        raise DownloadError(f"[오류] AI Hub 파일 목록에서 filekey를 한 줄도 읽지 못했습니다. (HTTP {status})\n"
                            f"  브라우저로 {FILE_LIST_URL}을 열어 목록이 보이는지 확인하세요. "
                            "보이면 목록 형식이 바뀐 것이고, 안 보이면 잠시 뒤 다시 실행하세요.")
    return files


def target_path(out_dir, name):
    """받은 zip을 둘 경로를 정한다. 라벨 zip은 train_annotations/, 이미지 zip은 train_images/이다.

    Args:
        out_dir (Path): 받을 폴더. 기본값은 data/external/aihub.
        name (str): AI Hub 파일 이름. 예: TL_1_조합.zip

    Returns:
        Path: 예: data/external/aihub/train_annotations/TL_1_조합.zip
    """
    return out_dir / SUBDIRS[name[1]] / name


def already_downloaded(path):
    """같은 이름의 파일이 이미 있는지 본다. 이름은 NFC로 맞춰 비교한다.

    Args:
        path (Path): target_path()의 결과.

    Returns:
        bool: 이미 있으면 True.
    """
    if not path.parent.is_dir():
        return False
    return any(nfc(entry.name) == path.name for entry in path.parent.iterdir())


def load_api_key():
    """레포의 .env와 환경 변수에서 AIHUB_APIKEY를 읽는다. 값은 출력하지 않는다.

    Returns:
        str: API 키.

    Raises:
        DownloadError: 키가 없을 때.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("AIHUB_APIKEY", "").strip()
    if not key:
        raise DownloadError("[오류] AI Hub API 키(AIHUB_APIKEY)를 찾지 못했습니다.\n"
                            "[해결] 1. AI Hub에서 576 데이터셋(경구약제 이미지) 이용 승인을 받고 API 키를 발급받습니다.\n"
                            "       2. 레포의 .env에 아래 줄을 넣습니다. .env가 없으면 .env.example을 복사해 만듭니다.\n"
                            "          AIHUB_APIKEY=<발급받은 키>")
    return key


def check_disk(folder, size):
    """받는 tar와 병합한 zip을 함께 둘 공간이 있는지 본다.

    Args:
        folder (Path): zip을 둘 폴더.
        size (int): 받을 파일의 대략 용량 (바이트).

    Raises:
        DownloadError: 남은 공간이 2 × 용량 + 2GiB보다 작을 때.
    """
    need = 2 * size + DISK_MARGIN
    free = shutil.disk_usage(folder).free
    if free < need:
        raise DownloadError(f"[오류] 디스크 공간이 부족합니다. 필요 {need / 2**30:.1f}GiB, 남은 공간 {free / 2**30:.1f}GiB\n"
                            "  받는 동안 tar와 병합한 zip이 함께 있어 파일 용량의 2배와 여유 2GiB가 필요합니다.\n"
                            "[해결] 공간을 비우거나, --out으로 공간이 넉넉한 다른 폴더를 지정하세요.")


def download_tar(filekey, api_key, tar_path, size):
    """filekey 하나를 tar로 받는다. 진행률을 출력한다.

    Args:
        filekey (int): AI Hub filekey.
        api_key (str): AI Hub API 키.
        tar_path (Path): 받은 tar를 저장할 임시 경로.
        size (int): 대략 용량 (바이트). 진행률 표시에만 쓴다.

    Raises:
        DownloadError: HTTP 오류가 났거나 받은 파일이 tar가 아닐 때.
    """
    request = urllib.request.Request(f"{DOWNLOAD_URL}?fileSn={filekey}", headers={"apikey": api_key})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(tar_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or size)
            received, shown = 0, -1
            while chunk := response.read(CHUNK):
                out.write(chunk)
                received += len(chunk)
                percent = min(100, received * 100 // max(total, 1))
                if percent != shown:
                    print(f"\r  받는 중 {received / 2**20:,.1f}MB / 약 {total / 2**20:,.1f}MB ({percent}%)", end="", flush=True)
                    shown = percent
            print()
    except urllib.error.HTTPError as error:
        body = error.read(300).decode("utf-8", errors="replace")
        if error.code >= 500:
            raise DownloadError(f"[오류] AI Hub 서버가 HTTP {error.code}로 답했습니다. AI Hub 쪽 문제일 수 있습니다.\n"
                                "[해결] 잠시 뒤 같은 명령을 다시 실행하세요. 이미 받은 파일은 건너뜁니다.") from error
        raise DownloadError(f"[오류] AI Hub가 다운로드를 거절했습니다: HTTP {error.code}\n  응답: {body}\n"
                            "  API 키가 맞는지, 576 데이터셋 이용 승인을 받았는지 확인하세요.") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise DownloadError(f"[오류] 다운로드 중 연결이 끊겼습니다: {error}\n"
                            "[해결] 같은 명령을 다시 실행하세요. 이미 받은 파일은 건너뜁니다.") from error
    if not tarfile.is_tarfile(tar_path):
        with open(tar_path, "rb") as f:
            head = f.read(300).decode("utf-8", errors="replace")
        raise DownloadError(f"[오류] 받은 파일이 tar가 아닙니다. AI Hub 응답: {head}\n"
                            "  API 키가 맞는지, 576 데이터셋 이용 승인을 받았는지 확인하세요.")


def merge_parts(tar_path, dest):
    """tar 안의 <이름>.zip.part<N> 파일을 번호 순서로 이어 붙여 zip 하나로 만든다.

    tar는 풀지 않고 멤버를 바로 읽는다. tar 안의 폴더 경로는 버리고 dest에만 쓴다.
    병합한 크기가 part 합계와 같고 zip으로 열릴 때만 최종 이름으로 바꾼다.

    Args:
        tar_path (Path): download_tar()로 받은 tar.
        dest (Path): 최종 zip 경로. target_path()의 결과.

    Raises:
        DownloadError: tar에 이 파일의 part가 없거나, 병합 결과가 올바른 zip이 아닐 때.
    """
    partial = dest.with_name(dest.name + ".partial")
    with tarfile.open(tar_path) as tar:
        parts = []
        for member in tar.getmembers():
            try:
                member = tarfile.data_filter(member, str(dest.parent))  # 절대경로, .., 링크가 있으면 오류
            except tarfile.FilterError as error:
                raise DownloadError(f"[오류] 받은 tar에 안전하지 않은 경로가 있어 멈췄습니다: {nfc(member.name)} ({error})\n"
                                    "  AI Hub에서 받은 파일이 아닐 수 있습니다. 아무것도 저장하지 않았습니다.") from error
            if not member.isfile():
                continue
            base = nfc(os.path.basename(member.name))
            match = PART_NAME.fullmatch(base)
            if match and match.group(1) == dest.name:
                parts.append((int(match.group(2)), member))
            elif base == dest.name:
                parts.append((0, member))
        if not parts:
            names = ", ".join(nfc(m.name) for m in tar.getmembers()[:5])
            raise DownloadError(f"[오류] 받은 tar에 {dest.name}이 없습니다. tar 안의 파일: {names}")
        parts.sort(key=lambda item: item[0])
        expected = sum(member.size for _, member in parts)
        with open(partial, "wb") as out:
            for _, member in parts:
                shutil.copyfileobj(tar.extractfile(member), out, CHUNK)
    if partial.stat().st_size != expected or not zipfile.is_zipfile(partial):
        partial.unlink()
        raise DownloadError(f"[오류] {dest.name} 병합 결과가 올바른 zip이 아닙니다. (part {len(parts)}개, 예상 {expected:,}바이트)\n"
                            "[해결] 같은 명령을 다시 실행하세요.")
    os.replace(partial, dest)
    check_aihub_file(dest)  # 받은 뒤에도 금지 파일이 아닌지 한 번 더 확인한다


def main(argv=None):
    """filekey를 검사한 뒤 하나씩 받아 폴더 규칙대로 저장한다.

    Args:
        argv (list[str]): 명령줄 인자. None이면 sys.argv를 쓴다.

    Returns:
        int: 종료 코드. 0이면 성공, 1이면 오류, 130이면 사용자가 멈춤.
    """
    parser = argparse.ArgumentParser(description="AI Hub 576 데이터셋에서 filekey로 고른 zip을 받는다.")
    parser.add_argument("filekeys", nargs="+", help="받을 filekey. 공백이나 쉼표로 나눈다. 예: 66065 66067")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"받을 폴더. 기본값: {DEFAULT_OUT.relative_to(PROJECT_ROOT)}")
    args = parser.parse_args(argv)
    temp_files = []
    try:
        keys = parse_filekeys(args.filekeys)
        if not keys:
            # fileSn 없이 요청하면 AI Hub가 데이터셋 전체(약 3.7TB)를 보낸다
            raise DownloadError("[오류] filekey가 없습니다. 받을 filekey를 하나 이상 적으세요.")
        files = fetch_file_list()

        # 받기 전에 전체 목록을 먼저 검사한다
        jobs = []
        for key in keys:
            if key not in files:
                raise DownloadError(f"[오류] filekey {key}는 576 데이터셋 목록에 없습니다. 아무것도 받지 않았습니다.\n"
                                    f"  filekey 목록: {FILE_LIST_URL}")
            name, size = files[key]
            try:
                check_aihub_file(name)
            except ValueError as error:
                raise DownloadError(f"[중단] filekey {key}({name})는 대회에서 사용을 금지한 파일입니다(RULE.md). "
                                    "아무것도 받지 않았습니다.\n[해결] 이 filekey를 빼고 다시 실행하세요.") from error
            if not NAME_PATTERN.fullmatch(name):
                raise DownloadError(f"[오류] filekey {key}의 파일 이름 {name}이 예상한 형식"
                                    "(TL/TS/VL/VS_<번호>_조합/단일.zip)이 아닙니다. 아무것도 받지 않았습니다.")
            jobs.append((key, name, size, target_path(args.out, name)))

        todo = []
        for key, name, size, dest in jobs:
            if already_downloaded(dest):
                print(f"[건너뜀] {name} (filekey {key}): 이미 있습니다. {dest}")
            else:
                todo.append((key, name, size, dest))
        if not todo:
            return 0

        api_key = load_api_key()
        for i, (key, name, size, dest) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {name} (filekey {key}, 약 {size / 2**20:,.0f}MB)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            check_disk(dest.parent, size)
            tar_path = args.out / f".download_{key}.tar"
            temp_files[:] = [tar_path, dest.with_name(dest.name + ".partial")]
            download_tar(key, api_key, tar_path, size)
            merge_parts(tar_path, dest)
            tar_path.unlink()
            temp_files.clear()
            print(f"  저장: {dest} ({dest.stat().st_size / 2**20:,.1f}MB)")
        return 0
    except DownloadError as error:
        print(error)
        return 1
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 멈췄습니다. 같은 명령을 다시 실행하면 받지 않은 파일부터 이어서 받습니다.")
        return 130
    finally:
        for path in temp_files:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
