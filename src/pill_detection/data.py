"""데이터를 읽고 고치는 함수.

학습 데이터(Kaggle 원본 + AI Hub 외부 데이터)의 라벨 정제에 쓰는 아래 함수가 있다.
read_label_fixes, load_annotations, check_label_fixes, build_aihub_export는 파일을 읽기만 하고, 나머지는 파일을 지우고 쓴다.
- read_label_fixes: 정제 목록 CSV를 읽고 CSV 안의 형식을 검사한다.
- load_annotations: annotation 폴더의 JSON을 모두 읽어 (이미지 파일명, category_id)별 목록을 만든다.
- check_label_fixes: 정제 목록의 각 줄이 원본과 맞는지 확인하고, 줄마다 고치거나 새로 만들 JSON을 정한다.
- copy_annotations: 사본 annotation 폴더를 지우고, 원본 annotation 폴더를 그대로 다시 복사한다. (Kaggle 사본)
- apply_label_fixes: check_label_fixes가 정한 대로 사본의 JSON을 고치고, add 줄은 새 JSON을 만든다. (Kaggle 사본)
- build_aihub_export: AI Hub 사본에 쓸 JSON과 정제 라벨에 맞춘 새 파일명을 원본에서 메모리로 모두 만들고 검사한다.
- write_aihub_export: build_aihub_export의 결과에 id를 매겨 AI Hub 사본 폴더에 쓰고, PNG를 새 이름으로 링크한다.
- write_label_changes: 기준과 결과를 비교해서, 바뀐 annotation을 변경 전후 값과 함께 CSV로 저장한다.
  내용이 바뀌었을 때만 번호를 붙인 새 파일(_v2, _v3 …)로 저장해서, 이전 기록을 덮어쓰지 않는다.
- make_train_dataset: 위 함수들을 차례로 불러 Kaggle 사본, AI Hub 사본, 변경 기록 하나를 한 번에 만든다. (학습 데이터 준비)
- make_fixed_annotations: Kaggle 원본만 고친 사본과 변경 기록을 만든다. (AI Hub 데이터 없이 Kaggle 사본만 다시 만들 때)
AI Hub 파일 이름 도우미: check_aihub_file(금지 파일 검사), combo_name, pill_codes

annotation 폴더 구조
- Kaggle train_annotations는 알약 하나당 JSON 하나다. 경로: <조합>_json/K-<코드>/<이미지 파일명>.json
- JSON 하나에 images, annotations, categories가 1개씩 들어 있다.
- 한 이미지의 라벨은 그 이미지에 있는 알약 수(3~4개)만큼의 JSON에 나뉘어 있다.
- AI Hub 원본(data/external/aihub/raw/train_annotations)도 같은 구조지만 아래가 다르다. (aihub=True로 읽는다)
    - category_id, id가 모두 1이다. 클래스는 images[0].drug_N(K-코드)에서 읽는다.
    - 같은 알약 폴더에 "<이미지 이름> copy.json" 같은 사본 JSON이 있다. 약 폴더 이름에 "-오류"가 붙은 곳, 읽을 수 없는 JSON도 있다.
    - PNG만 있는 이미지와 JSON만 있는 이미지가 있다. "_index.png"는 학습에 쓰지 않는 안내 이미지다.

학습 데이터 사본
- 원본은 읽기만 한다.
    - Kaggle: data/raw → data/processed/train_annotations. 원본 annotation 폴더를 복사한 사본의 JSON만 정제 목록대로 고친다.
      이미지는 복사하지 않고 원본(data/raw/train_images)에서 읽는다.
    - AI Hub: data/external/aihub/raw → data/processed/ai_hub/train_annotations, train_images.
      정제한 JSON을 Kaggle 형식으로 새로 쓴다. 파일명을 정제 라벨에 맞게 고치므로, 이미지는 새 이름의 하드링크로 둔다. (다른 디스크면 복사)
- 정제 목록은 어떻게 고칠지만 적은 CSV 하나다. 지금 위치: data/label_fix/label_fixes.csv (01 AI Hub EDA 노트북 2-4가 aihub 줄을 만든다)
    - source: kaggle 또는 aihub. 어느 원본의 줄인지 나타낸다.
    - file_name: 원본 이미지 파일명.
    - category_id: 원본 클래스. K-코드 숫자다. (K-003351 → 3351)
    - action: add(라벨 JSON이 없는 알약의 라벨을 새로 추가), modify(있는 라벨을 고침), exclude(이미지를 학습 데이터에서 뺌)
    - bbox_after: 고친 뒤의 [x, y, w, h]. 원본 픽셀이다. exclude 줄은 비운다.
    - area_after: 고친 뒤의 면적. bbox_after의 w × h와 같다. exclude 줄은 비운다.
    - category_id_after: 클래스를 바꿀 때만 바꾼 뒤의 클래스. 비어 있으면 그대로 둔다. (aihub만)
    - image_fix: 이미지 필드를 고칠 때만 "필드=값". 예: back_color=연회색 배경 (aihub만)
- 정제 목록의 한 줄은 (source, file_name, category_id)로 구분한다. annotation 하나는 한 줄에서만 고친다.
- 옛 5열 정제 목록(file_name, category_id, action, bbox, area)도 읽는다. source=kaggle로 보고 bbox, area를 _after 값으로 읽는다.

변경 기록 (write_label_changes)
- 기록은 한 묶음이다: data/label_fix/apply_label_fixes.csv(1번), apply_label_fixes_v2.csv, _v3 …
    - 1번은 main의 Kaggle 기록(CHANGE_COLUMNS)이고, v2부터는 Kaggle과 AI Hub를 함께 담는다. (TRAIN_CHANGE_COLUMNS)
- 모든 버전은 원본 대비 누적이다. 최신 파일 하나에 지금까지의 수정 전체가 있고, 두 버전 사이에 바뀐 것은 두 파일을 비교해 본다.
- make_train_dataset, make_fixed_annotations 어느 쪽으로 만들든 같은 묶음에 쌓인다. 최신 기록과 내용이 같으면 새 파일을 만들지 않는다.
- Kaggle만 다시 만들면(make_fixed_annotations) aihub 줄은 최신 기록에서 그대로 이어받는다.
"""

import collections
import csv
import io
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

# 정제 목록 CSV의 열. 이름과 순서가 모두 같아야 한다.
FIX_COLUMNS = ["source", "file_name", "category_id", "action", "bbox_after", "area_after", "category_id_after", "image_fix"]

# 옛 정제 목록 CSV의 열. 이 형식도 읽는다. (모든 줄을 source=kaggle로 본다)
LEGACY_FIX_COLUMNS = ["file_name", "category_id", "action", "bbox", "area"]

FIX_SOURCES = ("kaggle", "aihub")

# add: 라벨 JSON이 없는 알약의 라벨을 새로 추가, modify: 있는 라벨을 고침, exclude: 이미지를 뺌 (옛 형식은 add와 modify만)
FIX_ACTIONS = ("add", "modify", "exclude")
LEGACY_FIX_ACTIONS = ("add", "modify")

# 변경 기록 CSV(write_label_changes)의 열. Kaggle만 기록할 때(1차 가공, make_fixed_annotations)
CHANGE_COLUMNS = ["file_name", "category_id", "action", "bbox_before", "bbox_after", "area_before", "area_after"]

# Kaggle과 AI Hub를 함께 기록할 때(make_train_dataset)의 열. 파일명과 클래스가 바뀐 것도 남긴다
TRAIN_CHANGE_COLUMNS = ["source", "file_name", "file_name_after", "category_id", "category_id_after", "action",
                        "bbox_before", "bbox_after", "area_before", "area_after"]

# images 항목 중 이미지에 대한 값 12개. 같은 이미지의 JSON끼리 값이 같다.
# add 줄의 새 JSON은 이 항목만 이미지 정보 JSON에서 가져오고, 나머지 항목(알약 정보)은 알약 정보 JSON의 값을 쓴다.
IMAGE_FIELDS = ("id", "file_name", "imgfile", "width", "height", "size",
                "back_color", "light_color", "drug_S", "drug_dir", "camera_la", "camera_lo")

# 대회 train JSON의 annotations[0] 키 순서
KAGGLE_ANN_KEYS = ("area", "iscrowd", "bbox", "category_id", "ignore", "segmentation", "id", "image_id")

# 대회에서 사용을 금지한 AI Hub 파일 (RULE.md)
BANNED_AIHUB_FILES = ("TL_2_조합.zip", "TS_2_조합.zip")

# AI Hub 사본의 첫 image id. 대회 train과 test가 image id 1~1500(test는 파일명 숫자)을 쓰므로 그다음부터 매긴다.
# annotation id는 make_train_dataset이 Kaggle 사본의 가장 큰 id 다음부터 매긴다. (Kaggle add 줄이 원본 ~5691 다음 번호를 쓴다)
AIHUB_START_IMAGE_ID = 1501

# 이미지 파일명의 숫자 7자리가 나타내는 이미지 필드 (순서대로). 예: K-..._0_2_0_2_70_000_200.png
NAME_FIELDS = ("drug_S", "back_color", "drug_dir", "light_color", "camera_la", "camera_lo", "size")
# 글자 값인 필드의 값 → 파일명 숫자. 숫자 값인 camera_la, size는 그대로, camera_lo는 3자리(0 → 000)로 적는다.
NAME_CODES = {
    "drug_S": {"정상알약": "0"},
    "back_color": {"검은색 배경": "0", "연회색 배경": "2"},
    "drug_dir": {"앞면": "0", "뒷면": "1"},
    "light_color": {"주백색": "2"},
}


def combo_name(file_name):
    """이미지 파일명에서 조합 이름을 꺼낸다.

    Args:
        file_name (str): 이미지 파일명. (K-001900-016548-019607-029451_0_2_0_2_70_000_200.png)

    Returns:
        str: 조합 이름. (K-001900-016548-019607-029451)
    """
    return file_name.split("_")[0]


def pill_codes(file_name):
    """이미지 파일명에 적힌 알약 K-코드를 정수 리스트로 꺼낸다.

    Args:
        file_name (str): 이미지 파일명.

    Returns:
        list[int]: K-코드 숫자. (K-001900-016548 → [1900, 16548])
    """
    return [int(code) for code in combo_name(file_name).split("-")[1:]]


def check_aihub_file(path):
    """대회에서 금지한 AI Hub 파일이면 오류를 낸다.

    macOS는 한글 파일명을 자모 분리형(NFD)으로 줄 때가 있어, 비교 전에 NFC로 맞춘다.

    Args:
        path (str | Path): AI Hub zip 경로.

    Raises:
        ValueError: TL_2_조합.zip, TS_2_조합.zip일 때.

    Examples:
        >>> check_aihub_file("data/external/aihub/raw/train_annotations/TL_1_조합.zip")  # 허용 파일은 아무것도 돌려주지 않는다
        >>> check_aihub_file("data/external/aihub/raw/train_annotations/TL_2_조합.zip")
        Traceback (most recent call last):
        ValueError: TL_2_조합.zip은 대회에서 사용을 금지한 파일입니다(RULE.md). 이 파일을 지우고 다른 조합 zip만 사용하세요: rm 'data/external/aihub/raw/train_annotations/TL_2_조합.zip'
    """
    name = unicodedata.normalize("NFC", Path(path).name)
    if name in BANNED_AIHUB_FILES:
        raise ValueError(
            f"{name}은 대회에서 사용을 금지한 파일입니다(RULE.md). "
            f"이 파일을 지우고 다른 조합 zip만 사용하세요: rm '{path}'"
        )


def read_label_fixes(csv_path, source=None):
    """정제 목록 CSV를 읽고 형식을 검사한다.

    CSV 안의 형식만 검사한다. 원본 라벨과 맞는지(이미지가 있는지, 고칠 JSON이 있는지)는 여기서 보지 않는다.
    열이 FIX_COLUMNS이면 합친 정제 목록으로, LEGACY_FIX_COLUMNS(옛 5열)이면 모든 줄을 source=kaggle로 읽는다.

    Args:
        csv_path (str | Path): 정제 목록 CSV 경로. UTF-8로 읽으며, 엑셀이 저장할 때 붙이는 BOM이 있어도 된다.
        source (str | None): "kaggle"이나 "aihub"를 주면 그 줄만 돌려준다. 형식 검사는 모든 줄에 한다.

    Returns:
        list[dict]: CSV 한 줄당 dict 하나. 순서는 파일 순서 그대로다.
            key는 FIX_COLUMNS와 같고, 빈칸은 None이다. dict 하나의 구조는 아래와 같다.

            {
                "source": "kaggle",                 # "kaggle" 또는 "aihub"
                "file_name": "K-003351-016262-018357_0_2_0_2_75_000_200.png",  # 원본 이미지 파일명
                "category_id": 18357,               # 원본 클래스 (int)
                "action": "modify",                 # "add", "modify", "exclude"
                "bbox_after": [567, 625, 311, 315], # 고친 뒤의 [x, y, w, h] (int 4개). exclude는 None
                "area_after": 97965,                # 고친 뒤의 면적 (int). w × h와 같다. exclude는 None
                "category_id_after": None,          # 바꾼 뒤의 클래스 (int). 바꾸지 않으면 None
                "image_fix": None,                  # 고칠 이미지 필드 {필드: 값}. 예: {"back_color": "연회색 배경"}
            }

    Raises:
        ValueError: 아래 중 하나라도 맞지 않을 때. 메시지에 CSV의 몇 번째 줄이 왜 틀렸는지 적는다.
            - 열 이름과 순서가 FIX_COLUMNS나 LEGACY_FIX_COLUMNS와 같다.
            - 빈칸이 없다. (합친 목록에서 source, file_name, category_id, action은 꼭 채운다)
            - source가 FIX_SOURCES 중 하나, action이 FIX_ACTIONS 중 하나다. (옛 형식은 add, modify만)
            - add·modify 줄: bbox_after는 정수 4개의 리스트, area_after는 정수이고 bbox_after의 너비 × 높이와 같다.
            - exclude 줄: bbox_after, area_after, category_id_after, image_fix가 비어 있다.
            - category_id, category_id_after는 정수다. image_fix는 "필드=값"이고 필드는 IMAGE_FIELDS 중 하나다.
            - (source, file_name, category_id)가 같은 줄이 두 줄 이상 없다.

    Examples:
        >>> fixes = read_label_fixes("data/label_fix/label_fixes.csv", source="kaggle")
        >>> len(fixes)
        21
        >>> fixes[8]["bbox_after"]
        [567, 625, 311, 315]
    """
    csv_path = Path(csv_path)
    name = csv_path.name  # 오류 메시지에는 파일 이름만 적는다. (내 컴퓨터의 절대경로가 나오지 않도록)

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if columns not in (FIX_COLUMNS, LEGACY_FIX_COLUMNS):
            raise ValueError(
                f"{name}의 열 이름이 다릅니다. 필요한 열(순서대로): {FIX_COLUMNS}, 읽은 열: {columns}\n"
                f"{name}의 첫 줄을 필요한 열 이름으로 고친 뒤 다시 실행하세요. (옛 형식 {LEGACY_FIX_COLUMNS}도 읽습니다)"
            )
        rows = list(reader)

    read_row = _read_legacy_fix_row if columns == LEGACY_FIX_COLUMNS else _read_fix_row
    fixes = []
    first_line = {}  # (source, file_name, category_id) → 처음 나온 줄 번호
    for line, row in enumerate(rows, start=2):  # 1번째 줄은 열 이름이므로 2부터 센다
        where = f"{name} {line}번째 줄"
        fix_hint = f"{name}에서 해당 줄을 고친 뒤 다시 실행하세요."
        # 칸이 열보다 많으면 DictReader가 남는 값을 None key에 모은다
        if None in row:
            raise ValueError(f"{where}: 칸이 {len(columns)}개보다 많습니다. 쉼표가 더 들어갔는지 확인하세요.\n{fix_hint}")
        fix = read_row(row, where, fix_hint)

        key = (fix["source"], fix["file_name"], fix["category_id"])
        if key in first_line:
            key_columns = "file_name, category_id" if columns == LEGACY_FIX_COLUMNS else "source, file_name, category_id"
            raise ValueError(
                f"{where}: {first_line[key]}번째 줄과 {key_columns}가 같습니다. "
                f"annotation 하나는 한 줄에서만 고칩니다.\n{fix_hint}"
            )
        first_line[key] = line
        fixes.append(fix)

    if source is not None:
        fixes = [fix for fix in fixes if fix["source"] == source]
    return fixes


def _read_legacy_fix_row(row, where, fix_hint):
    """옛 5열 정제 목록의 한 줄을 검사하고, 합친 목록과 같은 모양의 dict로 바꾼다."""
    # 칸이 열보다 적으면 모자란 열의 값이 None이 된다
    empty = [col for col in LEGACY_FIX_COLUMNS if not (row[col] or "").strip()]
    if empty:
        raise ValueError(f"{where}: 빈칸이 있습니다 {empty}. 모든 칸을 채워야 합니다.\n{fix_hint}")

    if row["action"] not in LEGACY_FIX_ACTIONS:
        raise ValueError(f"{where}: action '{row['action']}'은 쓸 수 없습니다. {LEGACY_FIX_ACTIONS} 중 하나로 적으세요.\n{fix_hint}")

    try:
        category_id = int(row["category_id"])
        area = int(row["area"])
        bbox = json.loads(row["bbox"])
    except ValueError as e:  # json.loads의 오류(JSONDecodeError)도 ValueError의 한 종류다
        raise ValueError(
            f"{where}: 숫자로 읽을 수 없는 값이 있습니다. category_id와 area는 정수, "
            f"bbox는 [x, y, w, h] 형식이어야 합니다. ({e})\n{fix_hint}"
        ) from e
    # bool과 실수는 받지 않는다. (원본 JSON의 bbox는 정수)
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox)):
        raise ValueError(f"{where}: bbox는 [x, y, w, h] 정수 4개여야 합니다. 읽은 값: {row['bbox']}\n{fix_hint}")

    w, h = bbox[2], bbox[3]
    if area != w * h:
        raise ValueError(f"{where}: area {area}가 bbox 너비 × 높이({w} × {h} = {w * h})와 다릅니다.\n{fix_hint}")

    return {"source": "kaggle", "file_name": row["file_name"], "category_id": category_id, "action": row["action"],
            "bbox_after": bbox, "area_after": area, "category_id_after": None, "image_fix": None}


def _read_fix_row(row, where, fix_hint):
    """합친 정제 목록의 한 줄을 검사하고 dict로 바꾼다. 빈칸은 None으로 둔다."""
    values = {col: (row[col] or "").strip() for col in FIX_COLUMNS}
    empty = [col for col in ("source", "file_name", "category_id", "action") if not values[col]]
    if empty:
        raise ValueError(f"{where}: 빈칸이 있습니다 {empty}. source, file_name, category_id, action은 꼭 채웁니다.\n{fix_hint}")
    if values["source"] not in FIX_SOURCES:
        raise ValueError(f"{where}: source '{values['source']}'은 쓸 수 없습니다. {FIX_SOURCES} 중 하나로 적으세요.\n{fix_hint}")
    action = values["action"]
    if action not in FIX_ACTIONS:
        raise ValueError(f"{where}: action '{action}'은 쓸 수 없습니다. {FIX_ACTIONS} 중 하나로 적으세요.\n{fix_hint}")

    try:
        category_id = int(values["category_id"])
    except ValueError as e:
        raise ValueError(f"{where}: category_id '{values['category_id']}'를 숫자로 읽을 수 없습니다. 정수로 적으세요.\n{fix_hint}") from e

    fix = {"source": values["source"], "file_name": values["file_name"], "category_id": category_id, "action": action,
           "bbox_after": None, "area_after": None, "category_id_after": None, "image_fix": None}

    if action == "exclude":
        filled = [col for col in ("bbox_after", "area_after", "category_id_after", "image_fix") if values[col]]
        if filled:
            raise ValueError(f"{where}: exclude 줄은 이미지를 통째로 빼므로 {filled}를 비웁니다.\n{fix_hint}")
        return fix

    if not (values["bbox_after"] and values["area_after"]):
        raise ValueError(f"{where}: {action} 줄은 bbox_after와 area_after를 채워야 합니다.\n{fix_hint}")
    try:
        bbox = json.loads(values["bbox_after"])
        area = int(values["area_after"])
    except ValueError as e:  # json.loads의 오류(JSONDecodeError)도 ValueError의 한 종류다
        raise ValueError(
            f"{where}: 숫자로 읽을 수 없는 값이 있습니다. bbox_after는 [x, y, w, h] 형식, area_after는 정수여야 합니다. ({e})\n{fix_hint}"
        ) from e
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(type(v) is int for v in bbox) and bbox[2] > 0 and bbox[3] > 0):
        raise ValueError(f"{where}: bbox_after는 너비와 높이가 0보다 큰 [x, y, w, h] 정수 4개여야 합니다. 읽은 값: {values['bbox_after']}\n{fix_hint}")
    w, h = bbox[2], bbox[3]
    if area != w * h:
        raise ValueError(f"{where}: area_after {area}가 bbox_after 너비 × 높이({w} × {h} = {w * h})와 다릅니다.\n{fix_hint}")
    fix["bbox_after"], fix["area_after"] = bbox, area

    if values["category_id_after"]:
        try:
            fix["category_id_after"] = int(values["category_id_after"])
        except ValueError as e:
            raise ValueError(
                f"{where}: category_id_after '{values['category_id_after']}'를 숫자로 읽을 수 없습니다. "
                f"바꾼 뒤의 K-코드 숫자를 적거나, 클래스를 바꾸지 않으면 비웁니다.\n{fix_hint}"
            ) from e

    if values["image_fix"]:
        field, sep, value = values["image_fix"].partition("=")
        if not sep or field not in IMAGE_FIELDS or not value:
            raise ValueError(
                f"{where}: image_fix '{values['image_fix']}'는 '필드=값' 형식이고, 필드는 {IMAGE_FIELDS} 중 하나여야 합니다. "
                f"예: back_color=연회색 배경\n{fix_hint}"
            )
        fix["image_fix"] = {field: int(value) if value.isdigit() else value}  # 숫자 필드(camera_la 등)는 정수로 둔다
    return fix


def load_annotations(ann_dir, aihub=False):
    """annotation 폴더의 JSON을 모두 읽어 (이미지 파일명, category_id)별 목록을 만든다.

    원본 폴더와 사본 폴더에 똑같이 쓴다. 값은 JSON에 적힌 그대로 담고 고치지 않는다.
    이미지 파일이 있는지는 보지 않는다. annotation 폴더만 읽는다.

    aihub=True면 AI Hub 원본 구조로 읽는다. (모듈 docstring의 annotation 폴더 구조 참고)
    - 경로와 이름은 NFC로 맞춘다. 이미지 파일명은 JSON 파일 이름에서 " copy …"를 뗀 이름이고, category_id는 images[0].drug_N이다.
    - 같은 (이미지 파일명, category_id)의 JSON이 여러 개면 경로 순서로 첫 JSON을 값으로 두고, 나머지 경로를 copies에 담는다.
    - 읽을 수 없는 JSON은 건너뛴다. 그 알약은 정제 목록에서 add로 다시 만든다. (check_label_fixes가 확인)
    - 폴더 이름, category_id, annotation id에 대한 검사는 하지 않는다. AI Hub 원본은 이 값이 맞지 않기 때문이다.

    Args:
        ann_dir (str | Path): annotation 폴더. (예: data/raw/train_annotations) 하위 폴더의 JSON을 모두 읽는다.
        aihub (bool): AI Hub 원본 구조로 읽을지.

    Returns:
        dict[tuple[str, int], dict]: (이미지 파일명, category_id)를 key로, 그 JSON의 정보를 value로 담는다.
            key는 JSON 수만큼 있고, json_path 순서로 들어 있다. 정제 목록(read_label_fixes)과 같은 key라 바로 찾을 수 있다.
            value 하나의 구조는 아래와 같다. aihub=True면 copies(같은 key의 나머지 JSON 경로 목록)가 더 있다.

            {
                "json_path": "K-003351-016262-018357_json/K-018357/K-003351-016262-018357_0_2_0_2_75_000_200.json",
                                                   # ann_dir 기준 경로. 구분자는 운영체제와 관계없이 /
                "file_name": "K-003351-016262-018357_0_2_0_2_75_000_200.png",  # images[0].file_name
                "image_id": 239,                   # images[0].id
                "category_id": 18357,              # annotations[0].category_id
                "ann_id": 903,                     # annotations[0].id
                "bbox": [6567, 625, 311, 315],     # annotations[0].bbox. [x, y, w, h] 원본 값 그대로
                "area": 97965,                     # annotations[0].area. 원본 값 그대로라 w × h와 다를 수 있다
                "width": 976,                      # images[0].width (aihub=False만)
                "height": 1280,                    # images[0].height (aihub=False만)
                "category_name": "종근당글리아티린연질캡슐(콜린알포세레이트) ",  # categories[0].name 그대로. 끝 공백도 남긴다 (aihub=False만)
            }

    Raises:
        FileNotFoundError: ann_dir 폴더가 없을 때.
        ValueError: JSON이 하나도 없거나, 아래 중 하나라도 맞지 않을 때. 메시지에 어느 JSON이 왜 틀렸는지 적는다.
            - JSON으로 읽을 수 있다. (aihub=False)
            - JSON마다 images, annotations, categories가 1개씩 있다.
            - 약 폴더 이름(K-코드), annotations의 category_id, categories의 id가 같다. (aihub=False)
            - JSON 파일 이름과 이미지 파일 이름이 같다. (확장자만 다르다) (aihub=False)
            - (이미지 파일명, category_id)가 같은 JSON이 두 개 이상 없다. (aihub=False)
            - annotation id가 서로 겹치지 않는다. (aihub=False)

    Examples:
        >>> records = load_annotations("data/raw/train_annotations")
        >>> len(records)  # JSON 수
        763
        >>> records[("K-003351-016262-018357_0_2_0_2_75_000_200.png", 18357)]["bbox"]  # x 좌표 오타가 있는 원본
        [6567, 625, 311, 315]
    """
    ann_dir = Path(ann_dir)
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"annotation 폴더가 없습니다: {ann_dir.name}\n폴더 경로를 확인하세요. (예: data/raw/train_annotations)")

    # 운영체제와 관계없이 같은 순서로 읽도록, ann_dir 기준 경로(구분자 /)의 글자 순서로 정렬한다
    paths = sorted(ann_dir.rglob("*.json"), key=lambda p: _rel_path(p, ann_dir))
    if not paths:
        raise ValueError(f"{ann_dir.name} 폴더 아래에 JSON이 하나도 없습니다.\nannotation 폴더 경로가 맞는지 확인하세요. (예: data/raw/train_annotations)")
    if aihub:
        return _load_aihub_annotations(ann_dir, paths)

    fix_hint = "이 함수는 Kaggle train_annotations 구조(JSON 하나에 알약 하나)를 전제로 합니다. 폴더 경로가 맞는지, 파일이 바뀌지 않았는지 확인하세요."
    records = {}
    first_ann_path = {}  # annotation id → 처음 나온 JSON 경로
    for path in paths:
        json_path = path.relative_to(ann_dir).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError as e:  # JSON 문법 오류(JSONDecodeError)도 ValueError의 한 종류다
            raise ValueError(f"{json_path}: JSON으로 읽을 수 없습니다. ({e})\n{fix_hint}") from e

        images, annotations, categories = data.get("images", []), data.get("annotations", []), data.get("categories", [])
        counts = (len(images), len(annotations), len(categories))
        if counts != (1, 1, 1):
            raise ValueError(f"{json_path}: images, annotations, categories가 1개씩이 아닙니다. 읽은 수: {counts}\n{fix_hint}")
        image, ann, cat = images[0], annotations[0], categories[0]

        folder = path.parent.name  # 약 폴더 이름 (K-코드)
        folder_id = int(folder.removeprefix("K-")) if folder.removeprefix("K-").isdigit() else None
        if not (folder_id == ann["category_id"] == cat["id"]):
            raise ValueError(
                f"{json_path}: 약 폴더 이름({folder}), annotations의 category_id({ann['category_id']}), "
                f"categories의 id({cat['id']})가 같지 않습니다.\n{fix_hint}"
            )

        if path.stem != Path(image["file_name"]).stem:
            raise ValueError(f"{json_path}: JSON 파일 이름과 이미지 파일 이름({image['file_name']})이 다릅니다.\n{fix_hint}")

        key = (image["file_name"], ann["category_id"])
        if key in records:
            raise ValueError(
                f"{json_path}: {records[key]['json_path']}와 이미지 파일명, category_id가 같습니다. "
                f"알약 하나의 라벨은 JSON 하나에만 있어야 합니다.\n{fix_hint}"
            )
        if ann["id"] in first_ann_path:
            raise ValueError(f"{json_path}: annotation id {ann['id']}가 {first_ann_path[ann['id']]}와 겹칩니다.\n{fix_hint}")
        first_ann_path[ann["id"]] = json_path

        records[key] = {
            "json_path": json_path,
            "file_name": image["file_name"],
            "image_id": image["id"],
            "category_id": ann["category_id"],
            "ann_id": ann["id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "width": image["width"],
            "height": image["height"],
            "category_name": cat["name"],
        }
    return records


def _rel_path(path, root):
    """root 기준 경로(구분자 /, NFC)."""
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())


def _aihub_image_name(json_name):
    """AI Hub JSON 파일 이름에서 이미지 파일명을 만든다. 사본 JSON은 " copy"부터 뗀다. (X copy 2.json → X.png)"""
    return unicodedata.normalize("NFC", json_name).removesuffix(".json").split(" copy")[0] + ".png"


def _load_aihub_annotations(ann_dir, paths):
    """load_annotations(aihub=True)의 본체. paths는 정렬한 JSON 경로다."""
    records = {}
    for path in paths:
        json_path = _rel_path(path, ann_dir)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:  # 읽을 수 없는 JSON은 건너뛴다. 정제 목록이 그 알약을 add로 다시 만든다
            continue
        images, annotations = data.get("images", []), data.get("annotations", [])
        if (len(images), len(annotations), len(data.get("categories", []))) != (1, 1, 1):
            raise ValueError(f"{json_path}: images, annotations, categories가 1개씩이 아닙니다.\nAI Hub 원본 JSON이 바뀌지 않았는지 확인하세요.")
        image, ann = images[0], annotations[0]
        category_id = int(image["drug_N"].removeprefix("K-"))
        key = (_aihub_image_name(path.name), category_id)
        if key in records:
            records[key]["copies"].append(json_path)
            continue
        records[key] = {
            "json_path": json_path,
            "file_name": key[0],
            "image_id": image["id"],
            "category_id": category_id,
            "ann_id": ann["id"],
            "bbox": ann["bbox"],
            "area": ann.get("area"),
            "copies": [],
        }
    return records


def _usable_bbox(bbox):
    """[x, y, w, h] 숫자 4개이고 너비와 높이가 0보다 큰지."""
    return isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox) and bbox[2] > 0 and bbox[3] > 0


def check_label_fixes(fixes, records, img_dir, aihub=False):
    """정제 목록의 각 줄이 원본과 맞는지 확인하고, 줄마다 고치거나 새로 만들 JSON을 정한다.

    파일은 읽기만 한다. 사본을 만들고 고치기 전에 실행해서, 문제가 있는 줄을 찾으면 파일을 건드리기 전에 멈춘다.

    add 줄의 새 JSON은 정제 목록에 없는 값(제품명, 각인, 이미지 크기 등)을 원본 JSON 두 개에서 가져온다.
    - 알약 정보: 같은 조합의 같은 약 폴더에 있는, 다른 각도 사진의 JSON
    - 이미지 정보: 같은 이미지의 다른 알약 JSON
    - 해당하는 JSON이 여러 개면 json_path 순서로 첫 번째를 쓴다. 실행할 때마다 같은 JSON을 쓰기 위해서다.

    aihub=True면 AI Hub 원본에 맞춰 아래가 다르다.
    - 촬영 PNG(_index.png 제외)마다 파일명의 알약을 모두 보고, 사본에 남길 JSON을 알약 하나에 하나씩 정한다.
      돌려주는 목록에는 정제 목록의 줄(add, modify)과 함께, 그대로 남길 알약(action "keep")도 들어 있다.
        - exclude 줄이 있는 이미지는 남기지 않는다.
        - 정제 줄이 없는 알약은 쓸 수 있는(읽을 수 있고 bbox가 맞는) 원본 JSON이 정확히 하나여야 한다.
    - 알약 정보: 같은 조합의 같은 약 폴더에 JSON이 없으면, 같은 클래스의 JSON 중 json_path 순서로 첫 번째를 쓴다.
      (AI Hub는 같은 클래스의 알약 정보가 원본 전체에서 같다) 클래스를 바꾸는 modify 줄도 새 클래스의 알약 정보 JSON을 정한다.
    - 이미지 정보: 같은 이미지의 JSON이 하나도 없으면(PNG만 있는 이미지) None으로 두고, build_aihub_export가 파일명에서 채운다.

    Args:
        fixes (list[dict]): read_label_fixes()의 결과. Kaggle이면 source=kaggle 줄, AI Hub면 source=aihub 줄만 넣는다.
        records (dict[tuple[str, int], dict]): 원본 annotation 폴더로 만든 load_annotations()의 결과. (AI Hub는 aihub=True)
        img_dir (str | Path): 원본 이미지 폴더. (예: data/raw/train_images)
        aihub (bool): AI Hub 원본 규칙으로 확인할지.

    Returns:
        list[dict]: 줄마다 fixes의 dict에 JSON 경로 세 개를 더한 dict. Kaggle은 fixes와 같은 순서다.
            경로는 모두 annotation 폴더 기준이고, 구분자는 /이다.
            - json_path: modify는 고칠 원본 JSON, add는 새로 만들 JSON의 경로
            - drug_json_path: add만. 알약 정보를 가져올 JSON. modify는 None
            - image_json_path: add만. 이미지 정보를 가져올 JSON. modify는 None
            aihub=True면 원본 이미지 파일명 순서로, 남길 알약마다 dict 하나다. 위와 다른 점은 아래와 같다.
            - action: "keep"(그대로 남김), "add", "modify"
            - json_path: keep·modify는 원본 JSON, add는 None
            - target_path: 사본에 남을 JSON 경로. 클래스를 바꾸면 새 클래스의 약 폴더다.
            - drug_json_path: add와 클래스를 바꾸는 modify만. 그 밖에는 None

            {
                "file_name": "K-003351-013900-021325_0_2_0_2_70_000_200.png",
                "category_id": 3351,
                "action": "add",
                "bbox_after": [385, 836, 188, 194],
                "area_after": 36472,
                ...(read_label_fixes의 나머지 key)
                "json_path": "K-003351-013900-021325_json/K-003351/K-003351-013900-021325_0_2_0_2_70_000_200.json",
                "drug_json_path": "K-003351-013900-021325_json/K-003351/K-003351-013900-021325_0_2_0_2_75_000_200.json",
                "image_json_path": "K-003351-013900-021325_json/K-013900/K-003351-013900-021325_0_2_0_2_70_000_200.json",
            }

    Raises:
        FileNotFoundError: img_dir 폴더가 없을 때.
        ValueError: 아래 중 하나라도 맞지 않을 때. 메시지에 어느 줄이 왜 틀렸는지 적는다.
            - 모든 줄: 이미지 파일이 img_dir에 있다.
            - modify 줄: 원본에 그 이미지, 그 클래스의 JSON이 있다.
            - add 줄: 원본에 그 이미지, 그 클래스의 JSON이 아직 없다.
            - add 줄: category_id가 이미지 파일명의 알약 코드에 들어 있다.
            - add 줄: 알약 정보를 가져올 JSON과 이미지 정보를 가져올 JSON이 있다. (AI Hub는 알약 정보 JSON만)
            - Kaggle: exclude 줄, category_id_after, image_fix가 없다. (Kaggle 사본은 bbox와 area만 고친다)
            - AI Hub: 정제 줄이 없는 알약은 쓸 수 있는 원본 JSON이 정확히 하나다.
    """
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {img_dir.name}\n폴더 경로를 확인하세요. (예: data/raw/train_images)")
    if aihub:
        return _check_aihub_fixes(fixes, records, img_dir)

    plans = []
    for fix in fixes:
        file_name, category_id, action = fix["file_name"], fix["category_id"], fix["action"]
        key = (file_name, category_id)
        code = f"K-{category_id:06d}"  # 약 폴더 이름 (3351 → K-003351)
        where = f"수정 목록 ({file_name}, {code}, {action})"

        if action == "exclude" or fix["category_id_after"] is not None or fix["image_fix"] is not None:
            raise ValueError(
                f"{where}: Kaggle 사본은 bbox와 area만 고칩니다. exclude, category_id_after, image_fix는 쓸 수 없습니다.\n"
                f"이 줄이 AI Hub 데이터라면 source를 aihub로 적으세요."
            )

        if not (img_dir / file_name).is_file():
            raise ValueError(f"{where}: 이미지 파일이 {img_dir.name} 폴더에 없습니다.\n수정 목록의 file_name에 오타가 없는지 확인하세요.")

        if action == "modify":
            if key not in records:
                raise ValueError(
                    f"{where}: 원본에 이 이미지, 이 클래스의 JSON이 없습니다.\n"
                    f"라벨이 없는 알약을 새로 추가하는 것이면 action을 add로 적으세요. 아니면 file_name과 category_id를 확인하세요."
                )
            plans.append({**fix, "json_path": records[key]["json_path"], "drug_json_path": None, "image_json_path": None})
            continue

        # 여기부터 add 줄
        if key in records:
            raise ValueError(
                f"{where}: 원본에 이미 JSON이 있습니다. ({records[key]['json_path']})\n"
                f"있는 라벨을 고치는 것이면 action을 modify로 적으세요."
            )

        combo = file_name.split("_")[0]  # 조합 이름. 이미지 파일명의 첫 _ 앞부분 (K-003351-013900-021325)
        combo_codes = combo.split("-")[1:]  # 파일명의 알약 코드 (['003351', '013900', '021325'])
        if f"{category_id:06d}" not in combo_codes:
            raise ValueError(
                f"{where}: 이미지 파일명의 알약 코드({', '.join('K-' + c for c in combo_codes)})에 {code}가 없습니다.\n"
                f"사진에 없는 알약은 추가할 수 없습니다. category_id를 확인하세요."
            )

        # 알약 정보: 같은 조합의 같은 약 폴더(<조합>_json/K-<코드>/)에 있는 다른 각도 사진의 JSON
        drug_dir = f"{combo}_json/{code}/"
        drug_paths = sorted(r["json_path"] for r in records.values() if r["json_path"].startswith(drug_dir))
        if not drug_paths:
            raise ValueError(
                f"{where}: {drug_dir} 폴더에 다른 각도 사진의 JSON이 없어서, 새 JSON에 넣을 알약 정보를 가져올 수 없습니다.\n"
                f"같은 조합에 이 알약의 라벨이 하나도 없는 경우는 지금 처리하지 않습니다."
            )

        # 이미지 정보: 같은 이미지의 다른 알약 JSON
        image_paths = sorted(r["json_path"] for r in records.values() if r["file_name"] == file_name)
        if not image_paths:
            raise ValueError(
                f"{where}: 이 이미지의 JSON이 하나도 없어서, 새 JSON에 넣을 이미지 정보를 가져올 수 없습니다.\n"
                f"라벨이 하나도 없는 이미지는 지금 처리하지 않습니다."
            )

        plans.append({
            **fix,
            "json_path": f"{drug_dir}{Path(file_name).stem}.json",  # 원본과 같은 규칙: <조합>_json/K-<코드>/<이미지 파일명>.json
            "drug_json_path": drug_paths[0],
            "image_json_path": image_paths[0],
        })
    return plans


def _check_aihub_fixes(fixes, records, img_dir):
    """check_label_fixes(aihub=True)의 본체."""
    photos = sorted(unicodedata.normalize("NFC", p.name) for p in img_dir.glob("*.png") if not p.name.endswith("_index.png"))
    photo_set = set(photos)
    fix_of = {}
    excluded = set()
    for fix in fixes:
        file_name, category_id, action = fix["file_name"], fix["category_id"], fix["action"]
        where = f"정제 목록 (aihub, {file_name}, K-{category_id:06d}, {action})"
        if file_name not in photo_set:
            raise ValueError(f"{where}: 촬영 PNG가 {img_dir.name} 폴더에 없습니다.\n정제 목록의 file_name에 오타가 없는지, AI Hub 이미지 zip을 모두 풀었는지 확인하세요.")
        if category_id not in pill_codes(file_name):
            raise ValueError(f"{where}: 이미지 파일명의 알약 코드에 K-{category_id:06d}가 없습니다.\ncategory_id에는 원본 클래스(파일명에 적힌 코드)를 적고, 바꿀 클래스는 category_id_after에 적으세요.")
        if action == "exclude":
            excluded.add(file_name)
            continue
        record = records.get((file_name, category_id))
        if action == "modify" and record is None:
            raise ValueError(f"{where}: 원본에 읽을 수 있는 JSON이 없습니다.\n라벨을 새로 만드는 것이면 action을 add로 적으세요.")
        if action == "add" and record is not None:
            raise ValueError(f"{where}: 원본에 이미 JSON이 있습니다. ({record['json_path']})\n있는 라벨을 고치는 것이면 action을 modify로 적으세요.")
        fix_of[(file_name, category_id)] = fix

    # 알약 정보 JSON 후보: 클래스 → json_path 순서의 원본 JSON 경로
    paths_of_class = {}
    for record in records.values():
        paths_of_class.setdefault(record["category_id"], []).append(record["json_path"])
    image_paths = {}
    for record in records.values():
        image_paths.setdefault(record["file_name"], []).append(record["json_path"])

    def drug_json_path(file_name, category_id, where):
        candidates = sorted(paths_of_class.get(category_id, []))
        same_folder = [p for p in candidates if p.startswith(f"{combo_name(file_name)}_json/K-{category_id:06d}/")]
        if not candidates:
            raise ValueError(f"{where}: 원본에 K-{category_id:06d}의 JSON이 하나도 없어서 알약 정보를 가져올 수 없습니다.\ncategory_id_after를 확인하세요.")
        return (same_folder or candidates)[0]

    plans = []
    targets = {}
    for file_name in photos:
        if file_name in excluded:
            continue
        for category_id in pill_codes(file_name):
            key = (file_name, category_id)
            where = f"AI Hub ({file_name}, K-{category_id:06d})"
            record, fix = records.get(key), fix_of.get(key)
            if fix is None:
                if record is None or record["copies"] or not _usable_bbox(record["bbox"]):
                    reason = ("원본 JSON이 없습니다" if record is None else
                              f"원본 JSON이 {1 + len(record['copies'])}개입니다" if record["copies"] else
                              f"원본 bbox {record['bbox']}를 쓸 수 없습니다")
                    raise ValueError(f"{where}: 정제 목록에 줄이 없는데 {reason}.\n정제 목록(label_fixes.csv)에 이 알약의 add, modify, exclude 줄을 넣으세요.")
                plan = {"source": "aihub", "file_name": file_name, "category_id": category_id, "action": "keep",
                        "bbox_after": record["bbox"], "area_after": record["area"], "category_id_after": None, "image_fix": None,
                        "json_path": record["json_path"], "target_path": record["json_path"],
                        "drug_json_path": None, "image_json_path": None}
            else:
                new_id = fix["category_id_after"] or category_id
                if fix["action"] == "modify":
                    parts = record["json_path"].split("/")
                    target = record["json_path"] if new_id == category_id else f"{parts[0]}/K-{new_id:06d}/{parts[-1]}"
                    drug = drug_json_path(file_name, new_id, where) if new_id != category_id else None
                    plan = {**fix, "json_path": record["json_path"], "target_path": target,
                            "drug_json_path": drug, "image_json_path": None}
                else:  # add
                    paths = sorted(image_paths.get(file_name, []))
                    plan = {**fix, "json_path": None,
                            "target_path": f"{combo_name(file_name)}_json/K-{new_id:06d}/{Path(file_name).stem}.json",
                            "drug_json_path": drug_json_path(file_name, new_id, where),
                            "image_json_path": paths[0] if paths else None}
            if plan["target_path"] in targets:
                raise ValueError(f"{where}: 사본의 같은 JSON 경로({plan['target_path']})에 알약 두 개가 남습니다. ({targets[plan['target_path']]})\n"
                                 f"정제 목록의 category_id_after를 확인하세요.")
            targets[plan["target_path"]] = where
            plans.append(plan)
    return plans


def copy_annotations(raw_ann_dir, out_ann_dir):
    """사본 annotation 폴더를 지우고, 원본 annotation 폴더를 그대로 다시 복사한다.

    매번 지우고 다시 복사하므로, 사본은 항상 원본과 같은 상태에서 시작한다.
    이미지는 복사하지 않는다. 라벨 수정은 JSON만 바꾸므로, 이미지는 원본(data/raw)에서 읽는다.
    사본 annotation 폴더 안에 직접 넣어 둔 파일이 있으면 함께 지워진다.

    Args:
        raw_ann_dir (str | Path): 원본 annotation 폴더. (예: data/raw/train_annotations)
        out_ann_dir (str | Path): 사본 annotation 폴더. (예: data/processed/train_annotations)
            폴더 이름이 원본과 같아야 한다. 상위 폴더가 없으면 만든다.

    Returns:
        int: 복사한 파일 수. (Kaggle train_annotations는 763)

    Raises:
        FileNotFoundError: 원본 annotation 폴더가 없을 때.
        ValueError: 아래 경우. 사본 폴더를 지우는 작업이 다른 파일에 닿을 수 있어서, 아무것도 지우지 않고 멈춘다.
            - 사본 폴더 이름이 원본과 다를 때. (예: data/processed를 넣으면 그 안의 다른 파일까지 지워진다)
            - 원본 폴더와 사본 폴더가 겹칠 때. (같은 폴더이거나 한쪽이 다른 쪽 안에 있을 때)
    """
    raw_ann_dir, out_ann_dir = Path(raw_ann_dir), Path(out_ann_dir)

    # 지우기 전에 모두 확인한다. 오류 메시지에는 폴더 이름만 적는다. (내 컴퓨터의 절대경로가 나오지 않도록)
    if not raw_ann_dir.is_dir():
        raise FileNotFoundError(f"원본 annotation 폴더가 없습니다: {raw_ann_dir.name}\n폴더 경로를 확인하세요. (예: data/raw/train_annotations)")

    if out_ann_dir.name != raw_ann_dir.name:
        raise ValueError(
            f"사본 폴더 이름({out_ann_dir.name})이 원본 폴더 이름({raw_ann_dir.name})과 다릅니다. "
            f"사본 폴더를 통째로 지우고 다시 만들기 때문에, 다른 폴더를 넣으면 그 안의 파일까지 지워질 수 있어서 멈춥니다.\n"
            f"사본 annotation 폴더를 넣으세요. (예: data/processed/{raw_ann_dir.name})"
        )

    raw_abs, out_abs = raw_ann_dir.resolve(), out_ann_dir.resolve()
    if raw_abs == out_abs or out_abs.is_relative_to(raw_abs) or raw_abs.is_relative_to(out_abs):
        raise ValueError(
            f"원본 폴더와 사본 폴더({out_ann_dir.name})가 겹칩니다. 사본을 지우면 원본이 지워질 수 있어서 멈춥니다.\n"
            f"사본 폴더를 원본과 따로 떨어진 곳으로 지정하세요. (예: 원본 data/raw/train_annotations, 사본 data/processed/train_annotations)"
        )

    if out_ann_dir.exists():
        shutil.rmtree(out_ann_dir)
    out_ann_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(raw_ann_dir, out_ann_dir)
    return sum(1 for p in out_ann_dir.rglob("*") if p.is_file())


def _read_json(path):
    """JSON 파일을 읽는다. BOM이 있어도 읽는다."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, data):
    """원본 annotation JSON과 같은 형식으로 저장한다.

    원본 JSON 763개가 모두 이 형식이다.
    4칸 들여쓰기, 한글 그대로(ensure_ascii=False), BOM 없음, 줄바꿈 LF, 파일 끝 줄바꿈 없음.
    Windows에서 텍스트 모드로 쓰면 줄바꿈이 CRLF로 바뀌므로, 글자를 바이트로 바꿔 저장한다.
    """
    Path(path).write_bytes(_json_bytes(data))


def _json_bytes(data):
    """_write_json이 저장하는 바이트."""
    return json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8")


def apply_label_fixes(plans, records, raw_ann_dir, out_ann_dir):
    """check_label_fixes()가 정한 대로 사본의 JSON을 고치고, add 줄은 새 JSON을 만든다. (Kaggle 사본)

    AI Hub 사본은 build_aihub_export, write_aihub_export로 만든다.
    copy_annotations()로 원본 annotation 폴더를 사본에 복사한 뒤에 실행한다. 원본 폴더는 읽기만 한다.
    모든 줄을 먼저 확인하고 고친 내용을 메모리에서 만든 뒤에 저장한다.
    그래서 문제가 있는 줄이 있으면 아무 파일도 건드리지 않고 멈춘다.

    줄마다 하는 일
    - modify: 사본의 json_path 파일에서 annotation의 bbox와 area를 정제 목록 값(bbox_after, area_after)으로 바꾼다.
    - add: 새 JSON을 만들어 사본의 json_path에 저장한다.
        - 알약 정보 JSON(drug_json_path, 원본)을 통째로 바탕으로 쓴다. 알약 정보, categories, type이 여기서 온다.
        - images의 IMAGE_FIELDS 12개 항목은 이미지 정보 JSON(image_json_path, 원본)의 값으로 바꾼다.
        - annotation의 id는 원본에서 가장 큰 id 다음부터 plans 순서대로 매기고,
          image_id는 이미지 정보 JSON의 이미지 id로, bbox와 area는 정제 목록 값으로 바꾼다.
    - 저장 형식은 원본 JSON과 같다. (_write_json)

    Args:
        plans (list[dict]): check_label_fixes()의 결과.
        records (dict[tuple[str, int], dict]): 원본 annotation 폴더로 만든 load_annotations()의 결과.
            새 annotation id를 원본에서 가장 큰 id 다음부터 매기는 데 쓴다.
        raw_ann_dir (str | Path): 원본 annotation 폴더. (예: data/raw/train_annotations) 읽기만 한다.
        out_ann_dir (str | Path): 사본 annotation 폴더. (예: data/processed/train_annotations) 이 폴더의 파일만 고치고 만든다.

    Returns:
        list[dict]: plans와 같은 순서로, 줄마다 적용한 결과. key는 file_name, category_id, action,
            json_path(사본 annotation 폴더 기준), ann_id, bbox, area다.
            ann_id는 modify면 원래 id, add면 새로 매긴 id다.

    Raises:
        FileNotFoundError: 원본이나 사본 annotation 폴더가 없을 때, 또는 modify 줄의 JSON이 사본에 없을 때.
        ValueError: add 줄의 새 JSON이 사본에 이미 있을 때, 또는 알약 정보 JSON의 category_id가 그 줄과 다를 때.
    """
    raw_ann_dir, out_ann_dir = Path(raw_ann_dir), Path(out_ann_dir)
    if not raw_ann_dir.is_dir():
        raise FileNotFoundError(f"원본 annotation 폴더가 없습니다: {raw_ann_dir.name}\n폴더 경로를 확인하세요. (예: data/raw/train_annotations)")
    if not out_ann_dir.is_dir():
        raise FileNotFoundError(f"사본 annotation 폴더가 없습니다: {out_ann_dir.name}\ncopy_annotations()로 사본을 먼저 만드세요.")

    # 1단계: 모든 줄을 확인하고, 저장할 내용을 메모리에서 만든다. 여기서는 파일을 쓰지 않는다.
    next_ann_id = max(r["ann_id"] for r in records.values()) + 1
    writes = []   # (저장할 경로, 저장할 JSON)
    applied = []  # 돌려줄 결과
    for plan in plans:
        file_name, category_id, action = plan["file_name"], plan["category_id"], plan["action"]
        where = f"수정 목록 ({file_name}, K-{category_id:06d}, {action})"
        target = out_ann_dir / plan["json_path"]

        if action == "modify":
            if not target.is_file():
                raise FileNotFoundError(
                    f"{where}: 사본에 고칠 JSON이 없습니다. ({plan['json_path']})\n"
                    f"copy_annotations()로 사본을 먼저 만든 뒤 실행하세요."
                )
            data = _read_json(target)
        else:  # add
            if target.exists():
                raise ValueError(
                    f"{where}: 사본에 새로 만들 JSON이 이미 있습니다. ({plan['json_path']})\n"
                    f"이미 고친 사본에 다시 실행한 것이면, copy_annotations()로 사본을 다시 만든 뒤 실행하세요."
                )
            data = _read_json(raw_ann_dir / plan["drug_json_path"])  # 알약 정보: 통째로 바탕으로 쓴다
            if data["annotations"][0]["category_id"] != category_id:
                raise ValueError(
                    f"{where}: 알약 정보 JSON({plan['drug_json_path']})의 category_id가 "
                    f"{data['annotations'][0]['category_id']}입니다.\ncheck_label_fixes()의 결과를 그대로 넣었는지 확인하세요."
                )
            image = _read_json(raw_ann_dir / plan["image_json_path"])["images"][0]  # 이미지 정보
            for field in IMAGE_FIELDS:
                data["images"][0][field] = image[field]
            data["annotations"][0]["id"] = next_ann_id
            data["annotations"][0]["image_id"] = image["id"]
            next_ann_id += 1

        ann = data["annotations"][0]
        ann["bbox"] = list(plan["bbox_after"])
        ann["area"] = plan["area_after"]
        writes.append((target, data))
        applied.append({
            "file_name": file_name,
            "category_id": category_id,
            "action": action,
            "json_path": plan["json_path"],
            "ann_id": ann["id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
        })

    # 2단계: 확인이 모두 끝난 뒤에 저장한다.
    for target, data in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, data)
    return applied


def _image_fields_from_name(file_name):
    """이미지 파일명의 숫자 7자리를 이미지 필드 값으로 되돌린다. (NAME_CODES의 반대)"""
    codes = Path(file_name).stem.split("_")[1:]
    fields = {"file_name": file_name, "imgfile": file_name}
    for field, code in zip(NAME_FIELDS, codes):
        if field in NAME_CODES:
            values = [value for value, c in NAME_CODES[field].items() if c == code]
            if not values:
                raise ValueError(f"{file_name}: 파일명의 {field} 코드 {code}를 NAME_CODES에서 찾을 수 없습니다.\n"
                                 f"새 코드가 생겼으면 data.py의 NAME_CODES에 값을 더하세요.")
            fields[field] = values[0]
        else:
            fields[field] = int(code)
    return fields


def _fixed_aihub_json(plan, raw_ann_dir):
    """build_aihub_export에서 알약 하나(plans의 한 줄)의 정제한 JSON을 원본에서 만든다. (Kaggle 형식으로 바꾸기 전)"""
    if plan["action"] == "keep":
        return _read_json(raw_ann_dir / plan["json_path"])
    where = f"정제 목록 (aihub, {plan['file_name']}, K-{plan['category_id']:06d}, {plan['action']})"
    if plan["action"] == "modify":
        data = _read_json(raw_ann_dir / plan["json_path"])
        if plan["drug_json_path"]:  # 클래스를 바꾼다: 알약 정보는 새 클래스 JSON, 이미지 필드는 이 JSON의 값
            own_image = data["images"][0]
            data = {**data, **{k: v for k, v in _read_json(raw_ann_dir / plan["drug_json_path"]).items() if k != "annotations"}}
            data["images"] = [dict(data["images"][0])]
            for field in IMAGE_FIELDS:
                data["images"][0][field] = own_image[field]
    else:  # add: 알약 정보 JSON을 바탕으로, 이미지 필드만 같은 이미지 JSON(없으면 파일명)의 값으로 바꾼다
        data = _read_json(raw_ann_dir / plan["drug_json_path"])
        image = (_read_json(raw_ann_dir / plan["image_json_path"])["images"][0] if plan["image_json_path"]
                 else _image_fields_from_name(plan["file_name"]))
        for field in IMAGE_FIELDS:
            if field in image:
                data["images"][0][field] = image[field]
    new_id = plan["category_id_after"] or plan["category_id"]
    if data["images"][0]["drug_N"] != f"K-{new_id:06d}":
        raise ValueError(f"{where}: 알약 정보 JSON의 drug_N이 {data['images'][0]['drug_N']}입니다. (K-{new_id:06d}이어야 함)\n"
                         f"check_label_fixes()의 결과를 그대로 넣었는지 확인하세요.")
    if plan["image_fix"]:
        data["images"][0].update(plan["image_fix"])
    ann = data["annotations"][0]
    ann["bbox"] = list(plan["bbox_after"])
    ann["area"] = plan["area_after"]
    return data


def _export_name(categories, image):
    """정제 라벨로 이미지 파일명을 만든다. 조합 자리는 클래스를 작은 번호부터, 숫자 자리는 이미지 필드의 코드다."""
    codes = []
    for field in NAME_FIELDS:
        value = image[field]
        if field in NAME_CODES:
            if value not in NAME_CODES[field]:
                raise ValueError(f"{image['file_name']}: {field} 값 '{value}'의 파일명 코드를 NAME_CODES에서 찾을 수 없습니다.\n"
                                 f"새 값이 생겼으면 data.py의 NAME_CODES에 코드를 더하세요.")
            codes.append(NAME_CODES[field][value])
        else:
            codes.append(f"{value:03d}" if field == "camera_lo" else str(value))
    return "K-" + "-".join(f"{c:06d}" for c in sorted(categories)) + "_" + "_".join(codes) + ".png"


def build_aihub_export(plans, raw_ann_dir, img_dir):
    """AI Hub 사본에 쓸 JSON과 정제 라벨에 맞춘 새 파일명을 원본에서 메모리로 모두 만들고 검사한다. 파일은 읽기만 한다.

    write_aihub_export가 이 결과를 그대로 쓴다. 사본을 지우기 전에 실행해서, 문제가 있으면 사본을 건드리기 전에 멈춘다.

    알약마다(plans의 한 줄) 정제한 JSON을 원본에서 만든다.
    - keep: 원본 JSON(json_path)을 그대로 쓴다.
    - modify: 원본 JSON(json_path)의 bbox와 area를 정제 목록 값으로 바꾼다.
      category_id_after가 있으면 알약 정보를 새 클래스 JSON(drug_json_path)의 값으로 바꾼다. 이미지 필드는 그대로다.
    - add: 알약 정보 JSON(drug_json_path)을 바탕으로, 이미지 필드(IMAGE_FIELDS)는 같은 이미지의 JSON(image_json_path)에서 가져온다.
      그 JSON이 없으면(PNG만 있는 이미지) 파일명 숫자(NAME_CODES)로 채운다.
    - image_fix가 있으면 images[0]의 그 필드를 바꾼다.

    정제로 클래스나 이미지 필드가 바뀐 이미지는 파일명이 라벨과 어긋나므로, 모든 이미지의 이름을 정제 라벨에서 한 규칙으로 다시 만든다.
    대부분은 원래 이름과 같다.
    - 이름: 조합 자리는 클래스를 작은 번호부터, 숫자 7자리는 이미지 필드(NAME_FIELDS)의 코드(NAME_CODES)다.
    - JSON: 대회 train JSON과 같은 키와 순서로 만든다.
        - category_id, categories.id, dl_idx = K-코드 숫자, categories.name = dl_name, iscrowd = 0, area = 너비 × 높이
        - image id와 annotation id는 None으로 두고, write_aihub_export가 매긴다.

    Args:
        plans (list[dict]): check_label_fixes(aihub=True)의 결과.
        raw_ann_dir (str | Path): AI Hub 원본 annotation 폴더. 읽기만 한다. (예: data/external/aihub/raw/train_annotations)
        img_dir (str | Path): AI Hub 원본 이미지 폴더. PNG가 있는지만 본다. (예: data/external/aihub/raw/train_images)

    Returns:
        dict: {"names": {원래 이름: 새 이름} (모든 이미지),
               "images": {새 이름: 그 이미지의 JSON 목록 (클래스 순서)} (새 이름 순서)}

    Raises:
        ValueError: 알약 정보 JSON의 클래스가 정제 클래스와 다를 때, 같은 이미지의 JSON끼리 이미지 필드가 다를 때,
            같은 이미지에 같은 클래스가 두 번 있을 때, 이미지 필드 값의 파일명 코드가 NAME_CODES에 없을 때, 새 이름이 겹칠 때.
        FileNotFoundError: 원본 PNG가 없을 때.
    """
    raw_ann_dir, img_dir = Path(raw_ann_dir), Path(img_dir)

    # 알약마다 정제한 JSON을 만들고 이미지별로 모은다
    by_image = {}
    for plan in plans:
        by_image.setdefault(plan["file_name"], []).append(_fixed_aihub_json(plan, raw_ann_dir))

    names = {}
    for file_name, items in by_image.items():
        fields = {tuple(data["images"][0][f] for f in IMAGE_FIELDS) for data in items}
        if len(fields) != 1:
            raise ValueError(f"{file_name}: 같은 이미지의 JSON끼리 이미지 필드 값이 다릅니다.\n정제 목록의 image_fix를 그 이미지의 모든 알약 줄에 넣었는지 확인하세요.")
        categories = [int(data["images"][0]["drug_N"].removeprefix("K-")) for data in items]
        if len(set(categories)) != len(categories):
            raise ValueError(f"{file_name}: 같은 클래스의 알약이 두 번 있습니다 {sorted(categories)}.\n정제 목록의 category_id_after를 확인하세요.")
        names[file_name] = _export_name(categories, items[0]["images"][0])
    if len(set(names.values())) != len(names):
        duplicated = [n for n, count in collections.Counter(names.values()).items() if count > 1]
        raise ValueError(f"정제 라벨로 다시 만든 파일명이 겹칩니다: {duplicated[:3]}\n같은 사진이 다른 조합 이름으로 들어 있지 않은지 확인하고, 하나를 exclude 하세요.")
    for file_name in names:
        if not (img_dir / file_name).is_file():
            raise FileNotFoundError(f"{file_name}: 원본 PNG가 {img_dir.name} 폴더에 없습니다.\nAI Hub 이미지 zip을 모두 풀었는지 확인하세요.")

    # Kaggle 형식 JSON을 만든다. id 자리는 키 순서를 지키려고 미리 만들어 두고, write_aihub_export가 채운다
    images = {}
    for file_name, new in sorted(names.items(), key=lambda t: t[1]):
        items = sorted(by_image[file_name], key=lambda d: int(d["images"][0]["drug_N"].removeprefix("K-")))
        images[new] = []
        for data in items:
            category_id = int(data["images"][0]["drug_N"].removeprefix("K-"))
            x, y, w, h = data["annotations"][0]["bbox"]
            image = {**data["images"][0], "file_name": new, "imgfile": new, "id": None, "dl_idx": str(category_id)}
            values = {"area": w * h, "iscrowd": 0, "bbox": [x, y, w, h], "category_id": category_id, "ignore": 0,
                      "segmentation": [], "id": None, "image_id": None}
            images[new].append({
                "images": [image], "type": "instances", "annotations": [{key: values[key] for key in KAGGLE_ANN_KEYS}],
                "categories": [{"supercategory": "pill", "id": category_id, "name": image["dl_name"]}],
            })
    return {"names": names, "images": images}


def write_aihub_export(export, raw_ann_dir, raw_img_dir, out_dir, start_ann_id):
    """build_aihub_export()의 결과에 id를 매겨 AI Hub 사본 폴더에 쓰고, PNG를 새 이름으로 둔다.

    out_dir 아래에 원본과 같은 이름의 annotation 폴더, 이미지 폴더를 사본으로 둔다. (예: data/processed/ai_hub/train_annotations)
    - JSON: 사본 annotation 폴더를 지우고, <조합>_json/K-<코드>/<새 이름>.json에 원본 annotation JSON과 같은 저장 형식으로 쓴다.
        - image id는 새 이름 순서로 AIHUB_START_IMAGE_ID부터, annotation id는 같은 순서(이미지 안에서는 클래스 순서)로 start_ann_id부터
    - PNG: 사본 이미지 폴더의 PNG를 지우고, 새 이름의 하드링크로 둔다. 원본과 다른 디스크라 하드링크를 만들 수 없으면 복사한다.

    Args:
        export (dict): build_aihub_export()의 결과. 이 dict 안의 JSON에 id를 채운다.
        raw_ann_dir (str | Path): AI Hub 원본 annotation 폴더. 사본 폴더 이름을 정하고, 겹치는지 보는 데만 쓴다.
        raw_img_dir (str | Path): AI Hub 원본 이미지 폴더. 읽기만 한다. (예: data/external/aihub/raw/train_images)
        out_dir (str | Path): AI Hub 사본 폴더. (예: data/processed/ai_hub) 상위 폴더가 없으면 만든다.
        start_ann_id (int): 첫 annotation id. make_train_dataset은 Kaggle 사본의 가장 큰 id 다음 번호를 넣는다. (지금 정제 목록이면 5700)

    Returns:
        dict: {"n_images": 이미지 수, "n_json": JSON 수, "renamed": {원래 이름: 새 이름} (바뀐 이미지만),
               "names": {원래 이름: 새 이름} (모든 이미지), "linked": 하드링크 수, "copied": 복사한 수}

    Raises:
        ValueError: 사본 폴더가 원본 폴더와 겹칠 때. (같은 폴더이거나 한쪽이 다른 쪽 안에 있을 때) 아무것도 지우지 않고 멈춘다.
    """
    raw_ann_dir, raw_img_dir, out_dir = Path(raw_ann_dir), Path(raw_img_dir), Path(out_dir)
    out_ann_dir, out_img_dir = out_dir / raw_ann_dir.name, out_dir / raw_img_dir.name

    # 지우기 전에 사본 폴더가 원본 폴더와 겹치지 않는지 확인한다
    for raw in (raw_ann_dir, raw_img_dir):
        for out in (out_ann_dir, out_img_dir):
            raw_abs, out_abs = raw.resolve(), out.resolve()
            if raw_abs == out_abs or out_abs.is_relative_to(raw_abs) or raw_abs.is_relative_to(out_abs):
                raise ValueError(
                    f"AI Hub 사본 폴더({out.name})가 원본 폴더({raw.name})와 겹칩니다. 사본을 지우면 원본이 지워질 수 있어서 멈춥니다.\n"
                    f"AI Hub 사본 폴더를 원본과 따로 떨어진 곳으로 지정하세요. (예: data/processed/ai_hub)"
                )

    # id를 매긴다
    names, images = export["names"], export["images"]
    writes = {}
    ann_id = start_ann_id
    for image_id, new in enumerate(sorted(images), start=AIHUB_START_IMAGE_ID):
        for data in images[new]:
            ann = data["annotations"][0]
            data["images"][0]["id"] = ann["image_id"] = image_id
            ann["id"] = ann_id
            ann_id += 1
            writes[f"{combo_name(new)}_json/K-{ann['category_id']:06d}/{Path(new).stem}.json"] = data

    # 지우고 다시 쓴다
    if out_ann_dir.exists():
        shutil.rmtree(out_ann_dir)
    for json_path, data in writes.items():
        target = out_ann_dir / json_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, data)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for path in out_img_dir.glob("*.png"):
        path.unlink()
    linked = copied = 0
    for file_name, new in names.items():
        try:
            os.link(raw_img_dir / file_name, out_img_dir / new)
            linked += 1
        except OSError:  # 원본과 다른 디스크라 하드링크를 만들 수 없으면 복사한다
            shutil.copy2(raw_img_dir / file_name, out_img_dir / new)
            copied += 1
    return {"n_images": len(names), "n_json": len(writes), "renamed": {old: new for old, new in names.items() if old != new},
            "names": names, "linked": linked, "copied": copied}


def write_label_changes(base_records, out_records, csv_path, key_map=None, sources=None):
    """기준과 결과를 비교해서, 바뀐 annotation을 변경 전후 값과 함께 CSV로 저장한다.

    기준(base_records)에는 원본을 넣는다. 기록 묶음의 모든 버전은 원본 대비 누적이라서, 최신 파일 하나에 지금까지의 수정 전체가 있다.
    두 목록을 같은 key(이미지 파일명, category_id)끼리 비교한다. bbox와 area만 비교하고,
    나머지 값(제품명, image_id 등)이 바뀌었는지는 보지 않는다.

    key_map과 sources를 주면(make_train_dataset) Kaggle과 AI Hub를 함께 기록한다.
    - 기준 key를 key_map으로 결과 key에 이어서 비교한다. 파일명이나 클래스가 바뀐 annotation도 한 줄(modify)로 남는다.
    - 열은 TRAIN_CHANGE_COLUMNS다. file_name_after, category_id_after는 결과의 값이고, delete 줄은 비운다.

    묶음의 최신 기록이 TRAIN_CHANGE_COLUMNS 형식이면, key_map과 sources를 주지 않은 호출(make_fixed_annotations)도 그 형식으로 쓴다.
    - sources가 없으면 모든 key를 kaggle로 본다. (sources에 없는 key도 kaggle로 본다)
    - 이번 호출이 다루지 않은 출처(sources의 값에 없는 출처, sources가 없으면 kaggle이 아닌 출처)의 줄은
      최신 기록에서 글자 그대로 이어받는다. 그 출처의 사본은 다시 만들지 않았으니 변경도 최신 기록 그대로이기 때문이다.
    - 그래서 Kaggle만 다시 만들어도 AI Hub 줄이 빠진 기록이 생기지 않고, 정제 목록이 그대로면 새 파일이 생기지 않는다.
    - 묶음이 비었거나 최신 기록이 CHANGE_COLUMNS 형식이면 지금까지처럼 CHANGE_COLUMNS로 쓴다.

    action은 비교 결과로 정한다.
    - add: 결과에만 있다. before 열은 빈칸이다.
    - modify: 둘 다 있고 bbox나 area가 다르다. (key_map을 주면 파일명이나 클래스가 다른 것도)
    - delete: 기준에만 있다. after 열은 빈칸이다. Kaggle 정제 목록으로는 생기지 않으며, 생기면 숨기지 않고 기록한다.

    줄 순서는 add, modify, delete 순서이고, 같은 action 안에서는 JSON 경로 순서다. 이어받은 줄은 같은 action 안에서 뒤에 둔다.
    CSV는 UTF-8(BOM 없음)로 저장하고, bbox는 정제 목록 CSV와 같은 [x, y, w, h] 형식으로 적는다.

    저장할 파일은 아래 규칙으로 정한다. 이전 기록을 덮어쓰지 않는다.
    - csv_path와 같은 이름의 파일(묶음) 중 번호가 가장 큰 것을 찾는다. 번호 없는 csv_path가 1번,
      그다음은 이름 뒤에 _v2, _v3 …을 붙인 파일이다. (apply_label_fixes.csv, apply_label_fixes_v2.csv …)
      csv_path가 _vN(N≥2)으로 끝나면 그 부분을 뗀 이름을 1번 파일로 본다. 그래서 _v2 경로를 넣어도 같은 묶음에 쌓인다.
    - 하나도 없으면 1번 파일로 저장한다.
    - 가장 최근 파일과 새로 만든 내용이 같으면 저장하지 않고, 그 파일의 경로를 돌려준다.
      같은 정제 목록으로 다시 실행해도 파일이 늘지 않게 하려는 것이다.
      줄바꿈 방식(CRLF/LF), BOM, 줄 순서 차이는 같은 내용으로 본다. (git이 줄바꿈을 바꿔 꺼낼 수 있고, 이어받은 줄은 순서가 달라질 수 있기 때문)
    - 다르면 다음 번호의 파일로 저장한다.

    Args:
        base_records (dict[tuple[str, int], dict]): 기준 annotation 폴더로 만든 load_annotations()의 결과.
        out_records (dict[tuple[str, int], dict]): 결과(사본) annotation 폴더로 만든 load_annotations()의 결과.
        csv_path (str | Path): 1번 기록 파일의 경로. (예: data/label_fix/apply_label_fixes.csv)
            2번부터는 같은 폴더에 번호를 붙인 이름으로 저장한다.
        key_map (dict[tuple[str, int], tuple[str, int]] | None): 기준 key → 결과 key. 없는 key는 그대로 잇는다.
            add 줄도 넣으면 원래 파일명과 클래스가 file_name, category_id 열에 남는다.
        sources (dict[tuple[str, int], str] | None): 기준 key와 결과 key → "kaggle"이나 "aihub". source 열에 쓴다.
            없으면 모든 key를 kaggle로 본다.

    Returns:
        tuple[Path, list[dict]]: (기록 파일 경로, 변경 목록).
            기록 파일 경로는 새로 저장한 파일, 또는 내용이 같아서 그대로 둔 가장 최근 파일이다.
            변경 목록은 기록 파일의 줄과 같다. (이어받은 줄도 들어 있다)
            변경 목록의 key는 CHANGE_COLUMNS(또는 TRAIN_CHANGE_COLUMNS)다. bbox는 리스트, area는 정수, 빈칸은 None이다.

            {
                "file_name": "K-003351-016262-018357_0_2_0_2_75_000_200.png",
                "category_id": 18357,
                "action": "modify",
                "bbox_before": [6567, 625, 311, 315],
                "bbox_after": [567, 625, 311, 315],
                "area_before": 97965,
                "area_after": 97965,
            }
    """
    # 1. 같은 묶음의 기존 기록 파일을 찾는다. 번호 없는 파일이 1번, <이름>_v2, _v3 …이 2번, 3번 …이다.
    csv_path = Path(csv_path)
    numbered_name = re.fullmatch(r"(.+)_v(\d+)", csv_path.stem)
    if numbered_name and int(numbered_name.group(2)) >= 2:  # _vN 파일 경로를 받아도 같은 묶음에 쌓는다
        csv_path = csv_path.with_name(numbered_name.group(1) + csv_path.suffix)
    versions = {}
    if csv_path.is_file():
        versions[1] = csv_path
    numbered = re.compile(rf"{re.escape(csv_path.stem)}_v(\d+){re.escape(csv_path.suffix)}")
    for p in csv_path.parent.glob(f"{csv_path.stem}_v*{csv_path.suffix}"):
        match = numbered.fullmatch(p.name)
        if match and int(match.group(1)) >= 2:
            versions[int(match.group(1))] = p
    latest_text = ""
    if versions:
        latest_text = versions[max(versions)].read_bytes().decode("utf-8-sig", errors="replace").replace("\r\n", "\n")

    # 2. 기록 형식을 정한다. 최신 기록이 Kaggle + AI Hub 형식이면 그 형식으로 쓰고,
    #    이번 호출이 다루지 않은 출처의 줄은 최신 기록에서 이어받는다.
    def parse(column, value):
        """기록 CSV 칸의 글자를 변경 목록의 값으로 바꾼다. (아래 cell의 반대)

        AI Hub 원본에는 bbox가 []이고 area에 bbox 리스트가 든 JSON도 있어서, 숫자 칸도 JSON으로 읽고 안 되면 글자 그대로 둔다.
        """
        if value == "":
            return None
        if column in ("source", "file_name", "file_name_after", "action"):
            return value
        try:
            return json.loads(value)
        except ValueError:
            return value

    latest_train = latest_text.split("\n", 1)[0] == ",".join(TRAIN_CHANGE_COLUMNS)
    train = key_map is not None or sources is not None or latest_train
    covered = set(sources.values()) if sources else {"kaggle"}
    key_map, sources = key_map or {}, sources or {}
    carried = []
    if latest_train:
        for row in csv.DictReader(io.StringIO(latest_text)):
            if row["source"] not in covered:
                carried.append({column: parse(column, row[column]) for column in TRAIN_CHANGE_COLUMNS})

    # 3. 기준과 결과를 비교해서 변경 목록을 만든다.
    action_order = {"add": 0, "modify": 1, "delete": 2}
    # (기준 key, 결과 key) 짝. key_map에 없는 기준 key는 같은 key와 잇되, 다른 기준 key가 이미 이어진 결과 key면 delete로 둔다
    claimed = set(key_map.values())
    pairs = [(base_key, key_map.get(base_key, None if base_key in claimed else base_key)) for base_key in base_records]
    pairs += [(base_key, out_key) for base_key, out_key in key_map.items() if base_key not in base_records]
    linked = {out_key for _, out_key in pairs}
    pairs += [(out_key, out_key) for out_key in out_records if out_key not in linked]

    changes = []
    for base_key, out_key in pairs:
        base, out = base_records.get(base_key), out_records.get(out_key) if out_key else None
        if base is None and out is None:
            continue
        if base is None:
            action = "add"
        elif out is None:
            action = "delete"
        elif base["bbox"] != out["bbox"] or base["area"] != out["area"] or base_key != out_key:
            action = "modify"
        else:
            continue  # 바뀌지 않은 annotation
        change = {
            "file_name": base_key[0],
            "category_id": base_key[1],
            "action": action,
            "bbox_before": base["bbox"] if base else None,
            "bbox_after": out["bbox"] if out else None,
            "area_before": base["area"] if base else None,
            "area_after": out["area"] if out else None,
            "_json_path": (out or base)["json_path"],  # 정렬에만 쓰고 저장하지 않는다
        }
        if train:
            change["source"] = sources.get(base_key) or sources.get(out_key) or "kaggle"
            change["file_name_after"] = out_key[0] if out else None
            change["category_id_after"] = out_key[1] if out else None
        changes.append(change)
    changes.sort(key=lambda c: (action_order[c["action"]], c["_json_path"]))
    for change in changes:
        del change["_json_path"]
    # 이어받은 줄은 같은 action 안에서 새로 계산한 줄 뒤에 둔다. (sort는 순서가 같은 줄끼리 원래 순서를 지킨다)
    changes = sorted(changes + carried, key=lambda c: action_order[c["action"]])

    # 4. 저장할 CSV 내용을 글자로 먼저 만든다.
    def cell(value):
        """CSV 칸에 적을 글자. 빈칸은 None, bbox(리스트)는 [x, y, w, h] 형식."""
        if value is None:
            return ""
        return json.dumps(value) if isinstance(value, list) else str(value)

    columns = TRAIN_CHANGE_COLUMNS if train else CHANGE_COLUMNS
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for change in changes:
        writer.writerow([cell(change[col]) for col in columns])
    text = buffer.getvalue()

    # 5. 가장 최근 파일과 내용이 같으면 저장하지 않고, 다르면 다음 번호로 저장한다.
    def lines(csv_text):
        """(열 이름 줄, 나머지 줄을 정렬한 목록). 줄 순서만 다른 기록을 같은 내용으로 보려고 쓴다."""
        header, *rows = csv_text.replace("\r\n", "\n").split("\n")
        return header, sorted(row for row in rows if row)

    if versions:
        latest = max(versions)
        if lines(latest_text) == lines(text):
            return versions[latest], changes
        target = csv_path.with_name(f"{csv_path.stem}_v{latest + 1}{csv_path.suffix}")
    else:
        target = csv_path
    target.write_bytes(text.encode("utf-8"))
    return target, changes


def make_train_dataset(fix_csv, kaggle_ann_dir, kaggle_img_dir, aihub_ann_dir, aihub_img_dir,
                       kaggle_out_ann_dir, aihub_out_dir, change_csv):
    """정제 목록대로 고친 Kaggle 사본과 AI Hub 사본, 두 데이터의 변경 기록 하나를 한 번에 만든다. (학습 데이터 준비)

    아래 함수를 이 순서로 부르기만 한다. 한 단계씩 확인하거나 테스트할 때는 각 함수를 따로 쓴다.
        1. read_label_fixes(fix_csv)                              정제 목록 읽기 (kaggle 줄, aihub 줄로 나눔)
        2. load_annotations(...), check_label_fixes(...)          두 원본을 읽고 정제 목록과 대조 (AI Hub는 aihub=True)
        3. build_aihub_export                                     AI Hub 사본의 JSON과 새 파일명을 메모리에서 만들고 검사
        4. copy_annotations, apply_label_fixes, load_annotations  Kaggle 사본 만들고 고치고, 엄격하게 다시 읽기
        5. write_aihub_export                                     AI Hub 사본 쓰기, PNG 링크. annotation id는 Kaggle 사본 다음 번호부터
        6. load_annotations(AI Hub 사본), write_label_changes      AI Hub 사본을 엄격하게 다시 읽고, 변경 기록 하나 저장

    1~3에서 문제가 있으면 파일을 하나도 건드리지 않고 멈춘다. 그래서 이전에 만든 사본도 그대로 남는다.
    AI Hub annotation id는 Kaggle 사본(add 줄이 원본 다음 번호를 받는다)의 가장 큰 id 다음부터 매겨서, 두 사본의 id가 겹치지 않는다.
    변경 기록은 Kaggle 줄과 AI Hub 줄을 함께 담는다. (source 열로 구분, 열은 TRAIN_CHANGE_COLUMNS)
    기록은 원본 대비 누적이다. 묶음의 최신 기록과 내용이 다르면 다음 번호(_v2, _v3 …)로 쌓고, 같으면 그대로 둔다.

    경로는 상대경로도 되며, 실행하는 위치(현재 작업 폴더)를 기준으로 찾는다. 레포 최상위에서 실행할 때의 예:
        make_train_dataset(
            fix_csv="data/label_fix/label_fixes.csv",
            kaggle_ann_dir="data/raw/train_annotations",
            kaggle_img_dir="data/raw/train_images",
            aihub_ann_dir="data/external/aihub/raw/train_annotations",
            aihub_img_dir="data/external/aihub/raw/train_images",
            kaggle_out_ann_dir="data/processed/train_annotations",
            aihub_out_dir="data/processed/ai_hub",
            change_csv="data/label_fix/apply_label_fixes.csv",
        )

    Args:
        fix_csv (str | Path): 합친 정제 목록 CSV. (read_label_fixes 참고)
        kaggle_ann_dir, kaggle_img_dir (str | Path): Kaggle 원본 annotation, 이미지 폴더. 읽기만 한다.
        aihub_ann_dir, aihub_img_dir (str | Path): AI Hub 원본 annotation, 이미지 폴더. 읽기만 한다.
        kaggle_out_ann_dir (str | Path): Kaggle 사본 annotation 폴더. 폴더 전체를 지우고 다시 만든다. (copy_annotations 참고)
        aihub_out_dir (str | Path): AI Hub 사본 폴더. 그 안의 train_annotations는 지우고 다시 만들고, train_images의 PNG도 다시 만든다.
        change_csv (str | Path): 변경 기록의 1번 파일 경로. (write_label_changes 참고)

    Returns:
        dict: 결과 요약. (값은 지금 정제 목록의 예)
            {
                "kaggle": {"n_base": 763, "n_out": 771, "applied": [...]},
                "aihub": {"n_base": 원본 key 수, "n_images": 11974, "n_json": 45819, "renamed": {원래 이름: 새 이름},
                          "linked": 하드링크 수, "copied": 복사한 수},
                "change_csv": Path(...),   # 변경 기록 파일 경로
                "changes": [...],          # write_label_changes()의 변경 목록
            }

    Raises:
        FileNotFoundError, ValueError: 각 단계 함수의 오류를 그대로 낸다.
    """
    fixes = read_label_fixes(fix_csv)
    kaggle_fixes = [fix for fix in fixes if fix["source"] == "kaggle"]
    aihub_fixes = [fix for fix in fixes if fix["source"] == "aihub"]

    kaggle_base = load_annotations(kaggle_ann_dir)
    kaggle_plans = check_label_fixes(kaggle_fixes, kaggle_base, kaggle_img_dir)
    aihub_base = load_annotations(aihub_ann_dir, aihub=True)
    aihub_plans = check_label_fixes(aihub_fixes, aihub_base, aihub_img_dir, aihub=True)
    overlap = sorted(set(kaggle_base) & set(aihub_base))
    if overlap:
        combos = sorted({combo_name(file_name) for file_name, _ in overlap})
        ann_name, img_name = Path(aihub_ann_dir).name, Path(aihub_img_dir).name
        raise ValueError(
            f"Kaggle 원본과 AI Hub 원본에 같은 (파일명, 클래스)가 {len(overlap)}개 있습니다. "
            f"예: {overlap[0][0]}, K-{overlap[0][1]:06d}\n"
            f"변경 기록에서 둘을 구별할 수 없어서 멈춥니다. 대회 데이터는 금지 zip({', '.join(BANNED_AIHUB_FILES)})과 같은 출처로 보여서, "
            f"AI Hub 원본에 금지 zip이나 대회 데이터를 직접 푼 파일이 섞였을 수 있습니다. "
            f"(data/external/aihub/sources.csv에 없는 파일이 직접 넣은 파일입니다)\n"
            f"겹친 조합 {len(combos)}개: {', '.join(combos[:5])}{' …' if len(combos) > 5 else ''}\n"
            f"AI Hub 원본 폴더(예: data/external/aihub/raw)에서 겹친 조합의 파일을 지우고 다시 실행하세요. "
            f"예: rm -r {ann_name}/{combos[0]}_json {img_name}/{combos[0]}_*.png"
        )
    aihub_export = build_aihub_export(aihub_plans, aihub_ann_dir, aihub_img_dir)  # 여기까지는 파일을 건드리지 않는다

    copy_annotations(kaggle_ann_dir, kaggle_out_ann_dir)
    kaggle_applied = apply_label_fixes(kaggle_plans, kaggle_base, kaggle_ann_dir, kaggle_out_ann_dir)
    kaggle_out = load_annotations(kaggle_out_ann_dir)

    # Kaggle add 줄이 원본 다음 번호를 받으므로, AI Hub는 Kaggle 사본의 가장 큰 id 다음부터 매긴다
    renamed = write_aihub_export(aihub_export, aihub_ann_dir, aihub_img_dir, aihub_out_dir,
                                 start_ann_id=max(r["ann_id"] for r in kaggle_out.values()) + 1)
    aihub_out = load_annotations(Path(aihub_out_dir) / Path(aihub_ann_dir).name)  # Kaggle 형식 검사를 그대로 통과해야 한다
    key_map = {(plan["file_name"], plan["category_id"]):
               (renamed["names"][plan["file_name"]], plan["category_id_after"] or plan["category_id"]) for plan in aihub_plans}
    sources = {key: "kaggle" for key in kaggle_base.keys() | kaggle_out.keys()}
    sources.update({key: "aihub" for key in aihub_base.keys() | aihub_out.keys() | key_map.keys()})
    change_path, changes = write_label_changes({**kaggle_base, **aihub_base}, {**kaggle_out, **aihub_out},
                                               change_csv, key_map=key_map, sources=sources)
    return {
        "kaggle": {"n_base": len(kaggle_base), "n_out": len(kaggle_out), "applied": kaggle_applied},
        "aihub": {"n_base": len(aihub_base), "n_images": renamed["n_images"], "n_json": renamed["n_json"],
                  "renamed": renamed["renamed"], "linked": renamed["linked"], "copied": renamed["copied"]},
        "change_csv": change_path,
        "changes": changes,
    }


def make_fixed_annotations(fix_csv, base_ann_dir, img_dir, out_ann_dir, change_csv):
    """정제 목록대로 고친 Kaggle annotation 사본과 변경 기록을 한 번에 만든다.

    학습 데이터 전체(Kaggle + AI Hub)는 make_train_dataset으로 만든다. 이 함수는 AI Hub 데이터 없이 Kaggle 사본만 다시 만들 때 쓴다.
    정제 목록에 aihub 줄이 있으면 kaggle 줄만 쓴다.
    변경 기록은 make_train_dataset과 같은 묶음에 쌓는다. 최신 기록의 aihub 줄은 그대로 이어받는다. (write_label_changes 참고)
    kaggle add 줄을 더하면 Kaggle 사본의 annotation id가 늘어나서, 이미 만든 AI Hub 사본(Kaggle 사본 다음 번호부터)과 id가 겹친다.
    add 줄을 더했으면 make_train_dataset(scripts/prepare_train_data.py)으로 두 사본을 함께 다시 만든다.

    아래 함수를 이 순서로 부르기만 한다. 한 단계씩 확인하거나 테스트할 때는 각 함수를 따로 쓴다.
        1. read_label_fixes(fix_csv, source="kaggle")  정제 목록 읽기
        2. load_annotations(base_ann_dir)            기준 annotation 목록 만들기
        3. check_label_fixes(..., img_dir)           정제 목록을 기준과 대조하고, 고치거나 만들 JSON 정하기
        4. copy_annotations(base_ann_dir, out_ann_dir)   기준 annotation 폴더를 사본 위치에 다시 복사
        5. apply_label_fixes(..., base_ann_dir, out_ann_dir)   사본의 JSON 고치기, add 줄은 새 JSON 만들기
        6. load_annotations(out_ann_dir), write_label_changes(..., change_csv)   기준과 사본을 비교해 변경 기록 저장

    1~3에서 문제가 있으면 파일을 하나도 건드리지 않고 멈춘다.
    5의 사전 확인에서 멈추면, 사본은 4에서 막 복사한 기준 그대로 남는다. (일부만 고쳐진 상태로 남지 않는다)

    경로 넣는 법
    - 경로는 상대경로도 되며, 실행하는 위치(현재 작업 폴더)를 기준으로 찾는다.
      레포 최상위에서 실행하면 "data/raw/...", notebooks/eda/label_fix의 노트북에서 실행하면 "../../../data/raw/..."처럼 쓴다.
    - 읽기만 하는 경로: fix_csv, base_ann_dir, img_dir
    - 지우고 쓰는 경로: out_ann_dir (폴더 전체를 지우고 다시 만든다), change_csv (번호를 붙인 새 파일을 만든다)
    - 고칠 것이 생기면 정제 목록에 줄을 더하고 원본부터 다시 만든다. 이전 사본을 기준으로 한 번 더 고치지 않는다.
      변경 기록이 원본 대비 누적이어야 최신 기록 하나에 수정 전체가 남기 때문이다.

    Args:
        fix_csv (str | Path): 정제 목록 CSV. (예: data/label_fix/label_fixes.csv) kaggle 줄만 쓴다. (read_label_fixes 참고)
        base_ann_dir (str | Path): 기준 annotation 폴더. 읽기만 한다. 원본(data/raw/train_annotations)을 넣는다.
            정제 목록의 각 줄을 이 폴더와 대조하고(3), 이 폴더를 사본 위치에 복사하고(4),
            add 줄의 새 JSON에 넣을 알약 정보와 이미지 정보를 이 폴더에서 읽는다(5).
        img_dir (str | Path): 이미지 폴더. 읽기만 한다. (예: data/raw/train_images)
            정제 목록의 이미지 파일이 있는지 확인하는 데만 쓴다(3).
        out_ann_dir (str | Path): 고친 annotation을 만들 사본 폴더. (예: data/processed/train_annotations)
            폴더 전체를 지우고 base_ann_dir에서 다시 복사한 뒤 고친다(4, 5).
            - 폴더 이름이 base_ann_dir와 같아야 한다. (train_annotations) 다르면 멈춘다.
            - base_ann_dir와 겹치면(같은 폴더이거나 한쪽이 다른 쪽 안에 있으면) 멈춘다.
            - git에 올라가지 않도록 data/processed 아래에 둔다. (.gitignore가 data/processed/*를 제외한다)
        change_csv (str | Path): 변경 기록의 1번 파일 경로. (예: data/label_fix/apply_label_fixes.csv)
            이전 기록과 내용이 같으면 저장하지 않고, 다르면 _v2, _v3 …을 붙인 새 파일로 저장한다. (write_label_changes 참고)

    Returns:
        dict: 결과 요약. (값은 지금 정제 목록으로 원본을 1차 가공했을 때의 예)
            {
                "n_base": 763,               # 기준 JSON 수
                "n_out": 771,                # 사본 JSON 수 (add 줄만큼 늘어난다)
                "applied": [...],            # apply_label_fixes()의 결과. 정제 목록 한 줄당 dict 하나 (21개)
                "change_csv": Path(...),     # 변경 기록 파일 경로. 새로 저장했거나, 내용이 같아서 그대로 둔 파일
                "changes": [...],            # write_label_changes()의 변경 목록. 기록 파일의 줄과 같다
                                             # (묶음이 비었으면 21개, 최신 기록이 Kaggle + AI Hub 형식이면 이어받은 aihub 줄도 들어 있다)
            }

    Raises:
        FileNotFoundError, ValueError: 각 단계 함수의 오류를 그대로 낸다.
            메시지에 어느 줄(또는 JSON)이 왜 틀렸는지 나오고, 오류 출력(traceback)에서 어느 함수에서 멈췄는지 볼 수 있다.
    """
    fixes = read_label_fixes(fix_csv, source="kaggle")
    base_records = load_annotations(base_ann_dir)
    plans = check_label_fixes(fixes, base_records, img_dir)
    copy_annotations(base_ann_dir, out_ann_dir)
    applied = apply_label_fixes(plans, base_records, base_ann_dir, out_ann_dir)
    out_records = load_annotations(out_ann_dir)
    change_path, changes = write_label_changes(base_records, out_records, change_csv)
    return {
        "n_base": len(base_records),
        "n_out": len(out_records),
        "applied": applied,
        "change_csv": change_path,
        "changes": changes,
    }
