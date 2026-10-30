"""테스트가 같이 쓰는 경로, 기준값, 검사 함수.

기준값은 코드를 고치기 전의 결과에서 잡았다. 기준값이 바뀌면 그 이유가 의도한 변경인지 먼저 확인한다.
- KAGGLE_FIXED_HASH: main(e9100f9)의 make_fixed_annotations가 옛 label_fixes.csv(21줄)로 만든 사본 JSON 771개
- AIHUB_EXPORT_HASH: 01 노트북(eda/raewoo-external-eda 317cd30) 2-7이 내보낸 JSON 45,819개. 2-8에서 점검을 마친 결과다.
    노트북은 annotation id를 5692부터 매겼다. Kaggle 사본(add 줄 5692~5699)과 겹쳐서, 그 결과의 annotation id만 +8(5700부터) 한 해시다.
- AIHUB_FIX_ROWS_SHA: AIHUB_EXPORT_HASH를 만든 정제 목록의 aihub 줄. 이 줄이 바뀌면 AI Hub 결과도 바뀌므로 해시 비교를 건너뛴다.
"""

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_ANN_DIR = DATA_DIR / "raw" / "train_annotations"
KAGGLE_IMG_DIR = DATA_DIR / "raw" / "train_images"
AIHUB_RAW_DIR = DATA_DIR / "external" / "aihub" / "raw"
AIHUB_ANN_DIR = AIHUB_RAW_DIR / "train_annotations"
AIHUB_IMG_DIR = AIHUB_RAW_DIR / "train_images"
LABEL_FIX_CSV = DATA_DIR / "label_fix" / "label_fixes.csv"
CHANGE_CSV_V1 = DATA_DIR / "label_fix" / "apply_label_fixes.csv"  # main의 Kaggle 변경 기록 (21줄)
LEGACY_FIX_CSV = Path(__file__).parent / "fixtures" / "legacy_label_fixes.csv"  # main의 옛 5열 정제 목록 그대로
TEST_OUT_DIR = DATA_DIR / "processed" / "_pytest"  # 하드링크가 되도록 원본과 같은 디스크(data/ 아래)에 만든다

# (JSON 수, 해시)
KAGGLE_FIXED_HASH = (771, "506ee722cba0d070120e94fd1310acd2a390b22cd954f4606aec8d8b96fd80c6")
AIHUB_EXPORT_HASH = (45819, "8472ce82e63ed09463e10eec5cbd33e0478a25f730c80217fe8bbd49c9ee90c5")
AIHUB_FIX_ROWS_SHA = "2bd750a08c083cacaddd40258b6afd84bba6e8e0f5cf6c0a2fcba63dcb20cbc0"

# 01 노트북의 결론 (2-5 정제 라벨 표, 2-7 내보낸 결과 표, 2-8 다시 점검)
AIHUB_N_IMAGES = 11974
AIHUB_N_JSON = 45819
AIHUB_N_CLASSES = 116
AIHUB_N_COMBOS = 3997
AIHUB_N_EXCLUDED = 35  # 찾지 못함 20, 같은 장면 6, 같은 알약 구성 9
AIHUB_START_IMAGE_ID = 1501  # 대회 train·test image id(1~1500) 다음
AIHUB_START_ANN_ID = 5700  # Kaggle 사본 annotation id(원본 ~5691 + add 8줄 5692~5699) 다음

# 2-6 이름 표: 원래 이름 → 정제 라벨로 다시 만든 이름
EXPECTED_RENAMES = {
    "K-003614-027777-028763-030308_0_2_0_2_70_000_200.png": "K-003614-016262-027777-028763_0_2_0_2_70_000_200.png",
    "K-003614-027777-028763-030308_0_2_0_2_75_000_200.png": "K-003614-016262-027777-028763_0_2_0_2_75_000_200.png",
    "K-003614-027777-028763-030308_0_2_0_2_90_000_200.png": "K-003614-016262-027777-028763_0_2_0_2_90_000_200.png",
    "K-012247-016551-024850-029451_0_0_0_2_90_000_200.png": "K-012247-016551-024850-029451_0_2_0_2_90_000_200.png",
    "K-015280-016688-032310_0_2_0_2_70_000_200.png": "K-016206-016688-032310_0_2_0_2_70_000_200.png",
    "K-015280-016688-032310_0_2_0_2_75_000_200.png": "K-016206-016688-032310_0_2_0_2_75_000_200.png",
    "K-015280-016688-032310_0_2_0_2_90_000_200.png": "K-016206-016688-032310_0_2_0_2_90_000_200.png",
    "K-015280-018147-022074_0_2_0_2_70_000_200.png": "K-015280-018147-033026_0_2_0_2_70_000_200.png",
    "K-015280-018147-022074_0_2_0_2_75_000_200.png": "K-015280-018147-033026_0_2_0_2_75_000_200.png",
    "K-015280-018147-022074_0_2_0_2_90_000_200.png": "K-015280-018147-033026_0_2_0_2_90_000_200.png",
    "K-016206-018147-022074_0_2_0_2_70_000_200.png": "K-016206-018147-033026_0_2_0_2_70_000_200.png",
    "K-016206-018147-022074_0_2_0_2_75_000_200.png": "K-016206-018147-033026_0_2_0_2_75_000_200.png",
    "K-016206-018147-022074_0_2_0_2_90_000_200.png": "K-016206-018147-033026_0_2_0_2_90_000_200.png",
}

# 파일명 숫자 7자리 규칙. data.py와 따로 적어서, data.py의 규칙이 바뀌면 테스트가 잡게 한다
NAME_CODE_SPEC = {
    "drug_S": {"정상알약": "0"},
    "back_color": {"검은색 배경": "0", "연회색 배경": "2"},
    "drug_dir": {"앞면": "0", "뒷면": "1"},
    "light_color": {"주백색": "2"},
}

# Kaggle train JSON의 키 순서 (최상위, images[0], annotations[0], categories[0]). 763개가 모두 이 순서다
KAGGLE_ANN_KEYS = ("area", "iscrowd", "bbox", "category_id", "ignore", "segmentation", "id", "image_id")


def tree_hash(folder):
    """폴더 아래 JSON의 (상대경로, sha256) 목록으로 해시 하나를 만든다.

    Args:
        folder (Path): annotation 폴더.

    Returns:
        tuple[int, str]: (JSON 수, 해시). 경로는 NFC로 맞추고 구분자는 /이다.
    """
    lines = sorted(
        f"{unicodedata.normalize('NFC', p.relative_to(folder).as_posix())}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        for p in Path(folder).rglob("*.json")
    )
    return len(lines), hashlib.sha256("".join(lines).encode()).hexdigest()


def fix_rows_sha(csv_path, source):
    """정제 목록에서 한 source의 줄만 모아 해시를 만든다. 줄바꿈 방식과 다른 source의 줄은 결과에 영향이 없다.

    Args:
        csv_path (Path): 합친 정제 목록 CSV.
        source (str): "kaggle" 또는 "aihub".

    Returns:
        str: sha256 hex.
    """
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["source"] == source]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json_bytes(data):
    """대회 train JSON과 같은 저장 형식의 바이트. (4칸 들여쓰기, 한글 그대로, LF, 파일 끝 줄바꿈 없음)"""
    return json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")


def expected_file_name(classes, image):
    """정제 라벨(클래스 목록, 이미지 필드)로 만들어야 하는 이미지 파일명.

    Args:
        classes (Iterable[int]): 이미지의 클래스.
        image (dict): JSON images[0].

    Returns:
        str: 예: K-003614-016262-027777-028763_0_2_0_2_70_000_200.png
    """
    codes = [NAME_CODE_SPEC[field][image[field]] for field in ("drug_S", "back_color", "drug_dir", "light_color")]
    codes += [str(image["camera_la"]), f"{image['camera_lo']:03d}", str(image["size"])]
    return "K-" + "-".join(f"{c:06d}" for c in sorted(classes)) + "_" + "_".join(codes) + ".png"


def read_tree(ann_dir):
    """annotation 폴더의 JSON을 모두 읽는다.

    Args:
        ann_dir (Path): annotation 폴더.

    Returns:
        list[tuple[Path, dict, bytes]]: (ann_dir 기준 경로, JSON, 파일 바이트). 경로 순서다.
    """
    items = []
    for p in sorted(Path(ann_dir).rglob("*.json")):
        raw = p.read_bytes()
        items.append((p.relative_to(ann_dir), json.loads(raw.decode("utf-8")), raw))
    return items


def pill_codes(file_name):
    """파일명 조합 자리의 K-코드 목록. (data.py의 pill_codes와 따로 적는다)"""
    return [int(code) for code in file_name.split("_")[0].split("-")[1:]]


def read_csv_rows(csv_path):
    """CSV를 글자 그대로 읽는다. (BOM과 줄바꿈 방식은 무시)"""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
