"""데이터를 읽고 고치는 함수.

지금은 라벨 수정 사본 만들기에 쓰는 아래 함수가 있다. 앞의 세 함수는 파일을 읽기만 하고, copy_annotations부터 파일을 지우고 쓴다.
- read_label_fixes: 수정 목록 CSV를 읽고 CSV 안의 형식을 검사한다.
- load_annotations: annotation 폴더의 JSON을 모두 읽어 (이미지 파일명, category_id)별 목록을 만든다.
- check_label_fixes: 수정 목록의 각 줄이 원본과 맞는지 확인하고, 줄마다 고치거나 새로 만들 JSON을 정한다.
- copy_annotations: 사본 annotation 폴더를 지우고, 원본 annotation 폴더를 그대로 다시 복사한다.
- apply_label_fixes: check_label_fixes가 정한 대로 사본의 JSON을 고치고, add 줄은 새 JSON을 만든다.
- write_label_changes: 기준과 결과를 비교해서, bbox나 area가 바뀐 annotation을 변경 전후 값과 함께 CSV로 저장한다.
  내용이 바뀌었을 때만 번호를 붙인 새 파일(_v2, _v3 …)로 저장해서, 이전 기록을 덮어쓰지 않는다.
- make_fixed_annotations: 위 함수들을 차례로 불러, 수정 목록대로 고친 annotation 사본과 변경 기록을 한 번에 만든다.

annotation 폴더 구조
- Kaggle train_annotations는 알약 하나당 JSON 하나다. 경로: <조합>_json/K-<코드>/<이미지 파일명>.json
- JSON 하나에 images, annotations, categories가 1개씩 들어 있다.
- 한 이미지의 라벨은 그 이미지에 있는 알약 수(3~4개)만큼의 JSON에 나뉘어 있다.

라벨 수정 사본
- 원본(data/raw)은 읽기만 하고, 원본 annotation 폴더를 복사한 사본(data/processed/train_annotations)의 JSON만 수정 목록대로 고친다.
- 이미지는 복사하지 않는다. 사본 라벨을 쓸 때 이미지는 원본(data/raw/train_images)에서 읽는다.
- 수정 목록은 고친 뒤의 annotation 값을 원본 JSON과 같은 이름과 형식으로 적은 CSV다.
  지금 위치: data/label_fix/label_fixes.csv
    - file_name: 이미지 파일명. 원본 JSON의 images[0].file_name과 같다.
    - category_id: 클래스. K-코드 숫자다. (K-003351 → 3351)
    - action: add(라벨 JSON이 없는 알약의 라벨을 새로 추가) 또는 modify(있는 라벨의 bbox와 area를 바꿈)
    - bbox: 고친 뒤의 [x, y, w, h]. 원본 픽셀이다.
    - area: 고친 뒤의 면적. bbox의 w × h와 같다.
- 수정 목록의 한 줄은 (file_name, category_id)로 구분한다. annotation 하나는 한 줄에서만 고친다.
"""

import csv
import io
import json
import re
import shutil
from pathlib import Path

# 수정 목록 CSV의 열. 이름과 순서가 모두 같아야 한다.
FIX_COLUMNS = ["file_name", "category_id", "action", "bbox", "area"]

# add: 라벨 JSON이 없는 알약의 라벨을 새로 추가, modify: 있는 라벨의 bbox와 area를 바꿈
FIX_ACTIONS = ("add", "modify")

# 변경 기록 CSV(write_label_changes)의 열
CHANGE_COLUMNS = ["file_name", "category_id", "action", "bbox_before", "bbox_after", "area_before", "area_after"]

# images 항목 중 이미지에 대한 값 12개. 같은 이미지의 JSON끼리 값이 같다.
# add 줄의 새 JSON은 이 항목만 이미지 정보 JSON에서 가져오고, 나머지 항목(알약 정보)은 알약 정보 JSON의 값을 쓴다.
IMAGE_FIELDS = ("id", "file_name", "imgfile", "width", "height", "size",
                "back_color", "light_color", "drug_S", "drug_dir", "camera_la", "camera_lo")


def read_label_fixes(csv_path):
    """라벨 수정 목록 CSV를 읽고 형식을 검사한다.

    CSV 안의 형식만 검사한다. 원본 라벨과 맞는지(이미지가 있는지, 고칠 JSON이 있는지)는 여기서 보지 않는다.

    Args:
        csv_path (str | Path): 수정 목록 CSV 경로. UTF-8로 읽으며, 엑셀이 저장할 때 붙이는 BOM이 있어도 된다.

    Returns:
        list[dict]: CSV 한 줄당 dict 하나. 순서는 파일 순서 그대로다.
            dict 하나의 구조는 아래와 같다.

            {
                "file_name": "K-003351-016262-018357_0_2_0_2_75_000_200.png",  # 이미지 파일명
                "category_id": 18357,          # 클래스 (int)
                "action": "modify",            # "add" 또는 "modify"
                "bbox": [567, 625, 311, 315],  # 고친 뒤의 [x, y, w, h] (int 4개)
                "area": 97965,                 # 고친 뒤의 면적 (int). w × h와 같다
            }

    Raises:
        ValueError: 아래 중 하나라도 맞지 않을 때. 메시지에 CSV의 몇 번째 줄이 왜 틀렸는지 적는다.
            - 열 이름과 순서가 FIX_COLUMNS와 같다.
            - 빈칸이 없다.
            - action이 FIX_ACTIONS 중 하나다.
            - category_id와 area는 정수이고, bbox는 정수 4개의 리스트다.
            - area가 bbox의 너비 × 높이와 같다.
            - (file_name, category_id)가 같은 줄이 두 줄 이상 없다.

    Examples:
        >>> fixes = read_label_fixes("data/label_fix/label_fixes.csv")
        >>> len(fixes)
        21
        >>> fixes[8]
        {'file_name': 'K-003351-016262-018357_0_2_0_2_75_000_200.png', 'category_id': 18357, 'action': 'modify', 'bbox': [567, 625, 311, 315], 'area': 97965}
    """
    csv_path = Path(csv_path)
    name = csv_path.name  # 오류 메시지에는 파일 이름만 적는다. (내 컴퓨터의 절대경로가 나오지 않도록)
    fix_hint = f"{name}에서 해당 줄을 고친 뒤 다시 실행하세요."

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIX_COLUMNS:
            raise ValueError(
                f"{name}의 열 이름이 다릅니다. 필요한 열(순서대로): {FIX_COLUMNS}, 읽은 열: {reader.fieldnames}\n"
                f"{name}의 첫 줄을 필요한 열 이름으로 고친 뒤 다시 실행하세요."
            )
        rows = list(reader)

    fixes = []
    first_line = {}  # (file_name, category_id) → 처음 나온 줄 번호
    for line, row in enumerate(rows, start=2):  # 1번째 줄은 열 이름이므로 2부터 센다
        where = f"{name} {line}번째 줄"

        # 칸이 열보다 많으면 DictReader가 남는 값을 None key에 모은다
        if None in row:
            raise ValueError(f"{where}: 칸이 {len(FIX_COLUMNS)}개보다 많습니다. 쉼표가 더 들어갔는지 확인하세요.\n{fix_hint}")
        # 칸이 열보다 적으면 모자란 열의 값이 None이 된다
        empty = [col for col in FIX_COLUMNS if not (row[col] or "").strip()]
        if empty:
            raise ValueError(f"{where}: 빈칸이 있습니다 {empty}. 모든 칸을 채워야 합니다.\n{fix_hint}")

        if row["action"] not in FIX_ACTIONS:
            raise ValueError(f"{where}: action '{row['action']}'은 쓸 수 없습니다. {FIX_ACTIONS} 중 하나로 적으세요.\n{fix_hint}")

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

        key = (row["file_name"], category_id)
        if key in first_line:
            raise ValueError(
                f"{where}: {first_line[key]}번째 줄과 file_name, category_id가 같습니다. "
                f"annotation 하나는 한 줄에서만 고칩니다.\n{fix_hint}"
            )
        first_line[key] = line

        fixes.append({
            "file_name": row["file_name"],
            "category_id": category_id,
            "action": row["action"],
            "bbox": bbox,
            "area": area,
        })
    return fixes


def load_annotations(ann_dir):
    """annotation 폴더의 JSON을 모두 읽어 (이미지 파일명, category_id)별 목록을 만든다.

    원본 폴더와 사본 폴더에 똑같이 쓴다. 값은 JSON에 적힌 그대로 담고 고치지 않는다.
    이미지 파일이 있는지는 보지 않는다. annotation 폴더만 읽는다.

    Args:
        ann_dir (str | Path): annotation 폴더. (예: data/raw/train_annotations) 하위 폴더의 JSON을 모두 읽는다.

    Returns:
        dict[tuple[str, int], dict]: (이미지 파일명, category_id)를 key로, 그 JSON의 정보를 value로 담는다.
            key는 JSON 수만큼 있고, json_path 순서로 들어 있다. 수정 목록(read_label_fixes)과 같은 key라 바로 찾을 수 있다.
            value 하나의 구조는 아래와 같다.

            {
                "json_path": "K-003351-016262-018357_json/K-018357/K-003351-016262-018357_0_2_0_2_75_000_200.json",
                                                   # ann_dir 기준 경로. 구분자는 운영체제와 관계없이 /
                "file_name": "K-003351-016262-018357_0_2_0_2_75_000_200.png",  # images[0].file_name
                "image_id": 239,                   # images[0].id
                "category_id": 18357,              # annotations[0].category_id
                "ann_id": 903,                     # annotations[0].id
                "bbox": [6567, 625, 311, 315],     # annotations[0].bbox. [x, y, w, h] 원본 값 그대로
                "area": 97965,                     # annotations[0].area. 원본 값 그대로라 w × h와 다를 수 있다
            }

    Raises:
        FileNotFoundError: ann_dir 폴더가 없을 때.
        ValueError: JSON이 하나도 없거나, 아래 중 하나라도 맞지 않을 때. 메시지에 어느 JSON이 왜 틀렸는지 적는다.
            - JSON으로 읽을 수 있다.
            - JSON마다 images, annotations, categories가 1개씩 있다.
            - 약 폴더 이름(K-코드), annotations의 category_id, categories의 id가 같다.
            - JSON 파일 이름과 이미지 파일 이름이 같다. (확장자만 다르다)
            - (이미지 파일명, category_id)가 같은 JSON이 두 개 이상 없다.
            - annotation id가 서로 겹치지 않는다.

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
    paths = sorted(ann_dir.rglob("*.json"), key=lambda p: p.relative_to(ann_dir).as_posix())
    if not paths:
        raise ValueError(f"{ann_dir.name} 폴더 아래에 JSON이 하나도 없습니다.\nannotation 폴더 경로가 맞는지 확인하세요. (예: data/raw/train_annotations)")

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
        }
    return records


def check_label_fixes(fixes, records, img_dir):
    """수정 목록의 각 줄이 원본과 맞는지 확인하고, 줄마다 고치거나 새로 만들 JSON을 정한다.

    파일은 읽기만 한다. 사본을 만들고 고치기 전에 실행해서, 문제가 있는 줄을 찾으면 파일을 건드리기 전에 멈춘다.

    add 줄의 새 JSON은 수정 목록에 없는 값(제품명, 각인, 이미지 크기 등)을 원본 JSON 두 개에서 가져온다.
    - 알약 정보: 같은 조합의 같은 약 폴더에 있는, 다른 각도 사진의 JSON
    - 이미지 정보: 같은 이미지의 다른 알약 JSON
    - 해당하는 JSON이 여러 개면 json_path 순서로 첫 번째를 쓴다. 실행할 때마다 같은 JSON을 쓰기 위해서다.

    Args:
        fixes (list[dict]): read_label_fixes()의 결과.
        records (dict[tuple[str, int], dict]): 원본 annotation 폴더로 만든 load_annotations()의 결과.
        img_dir (str | Path): 원본 이미지 폴더. (예: data/raw/train_images)

    Returns:
        list[dict]: fixes와 같은 순서로, 줄마다 fixes의 dict에 JSON 경로 세 개를 더한 dict.
            경로는 모두 annotation 폴더 기준이고, 구분자는 /이다.
            - json_path: modify는 고칠 원본 JSON, add는 새로 만들 JSON의 경로
            - drug_json_path: add만. 알약 정보를 가져올 JSON. modify는 None
            - image_json_path: add만. 이미지 정보를 가져올 JSON. modify는 None

            {
                "file_name": "K-003351-013900-021325_0_2_0_2_70_000_200.png",
                "category_id": 3351,
                "action": "add",
                "bbox": [385, 836, 188, 194],
                "area": 36472,
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
            - add 줄: 알약 정보를 가져올 JSON과 이미지 정보를 가져올 JSON이 있다.
    """
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {img_dir.name}\n폴더 경로를 확인하세요. (예: data/raw/train_images)")

    plans = []
    for fix in fixes:
        file_name, category_id, action = fix["file_name"], fix["category_id"], fix["action"]
        key = (file_name, category_id)
        code = f"K-{category_id:06d}"  # 약 폴더 이름 (3351 → K-003351)
        where = f"수정 목록 ({file_name}, {code}, {action})"

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

    원본 JSON 763개가 모두 이 형식이다. (노트북 jaeone_label_fix 4번 저장 형식 확인)
    4칸 들여쓰기, 한글 그대로(ensure_ascii=False), BOM 없음, 줄바꿈 LF, 파일 끝 줄바꿈 없음.
    Windows에서 텍스트 모드로 쓰면 줄바꿈이 CRLF로 바뀌므로, 글자를 바이트로 바꿔 저장한다.
    """
    Path(path).write_bytes(json.dumps(data, ensure_ascii=False, indent=4).encode("utf-8"))


def apply_label_fixes(plans, records, raw_ann_dir, out_ann_dir):
    """check_label_fixes()가 정한 대로 사본의 JSON을 고치고, add 줄은 새 JSON을 만든다.

    copy_annotations()로 원본 annotation 폴더를 사본에 복사한 뒤에 실행한다. 원본 폴더는 읽기만 한다.
    모든 줄을 먼저 확인하고 고친 내용을 메모리에서 만든 뒤에 저장한다.
    그래서 문제가 있는 줄이 있으면 아무 파일도 건드리지 않고 멈춘다.

    줄마다 하는 일
    - modify: 사본의 json_path 파일에서 annotation의 bbox와 area를 수정 목록 값으로 바꾼다.
    - add: 새 JSON을 만들어 사본의 json_path에 저장한다.
        - 알약 정보 JSON(drug_json_path, 원본)을 통째로 바탕으로 쓴다. 알약 정보, categories, type이 여기서 온다.
        - images의 IMAGE_FIELDS 12개 항목은 이미지 정보 JSON(image_json_path, 원본)의 값으로 바꾼다.
        - annotation의 id는 원본에서 가장 큰 id 다음부터 plans 순서대로 매기고,
          image_id는 이미지 정보 JSON의 이미지 id로, bbox와 area는 수정 목록 값으로 바꾼다.
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
        ann["bbox"] = list(plan["bbox"])
        ann["area"] = plan["area"]
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


def write_label_changes(base_records, out_records, csv_path):
    """기준과 결과를 비교해서, bbox나 area가 바뀐 annotation을 변경 전후 값과 함께 CSV로 저장한다.

    기준(base_records)은 원본일 수도 있고, 2차·3차 가공을 할 때의 이전 수정본일 수도 있다.
    두 목록을 같은 key(이미지 파일명, category_id)끼리 비교한다. bbox와 area만 비교하고,
    나머지 값(제품명, image_id 등)이 바뀌었는지는 보지 않는다.

    action은 비교 결과로 정한다.
    - add: 결과에만 있다. before 열은 빈칸이다.
    - modify: 둘 다 있고 bbox나 area가 다르다.
    - delete: 기준에만 있다. after 열은 빈칸이다. 지금 수정 목록으로는 생기지 않으며, 생기면 숨기지 않고 기록한다.

    줄 순서는 add, modify, delete 순서이고, 같은 action 안에서는 JSON 경로 순서다.
    CSV는 UTF-8(BOM 없음)로 저장하고, bbox는 수정 목록 CSV와 같은 [x, y, w, h] 형식으로 적는다.

    저장할 파일은 아래 규칙으로 정한다. 이전 기록을 덮어쓰지 않는다.
    - csv_path와 같은 이름의 파일 중 번호가 가장 큰 것을 찾는다. 번호 없는 csv_path가 1번,
      그다음은 이름 뒤에 _v2, _v3 …을 붙인 파일이다. (apply_label_fixes.csv, apply_label_fixes_v2.csv …)
    - 하나도 없으면 csv_path로 저장한다.
    - 가장 최근 파일과 새로 만든 내용이 같으면 저장하지 않고, 그 파일의 경로를 돌려준다.
      같은 수정 목록으로 다시 실행해도 파일이 늘지 않게 하려는 것이다.
      줄바꿈 방식(CRLF/LF)과 BOM 차이는 같은 내용으로 본다. (git이 줄바꿈을 바꿔 꺼낼 수 있기 때문)
    - 다르면 다음 번호의 파일로 저장한다.

    Args:
        base_records (dict[tuple[str, int], dict]): 기준 annotation 폴더로 만든 load_annotations()의 결과.
        out_records (dict[tuple[str, int], dict]): 결과(사본) annotation 폴더로 만든 load_annotations()의 결과.
        csv_path (str | Path): 1번 기록 파일의 경로. (예: data/label_fix/apply_label_fixes.csv)
            2번부터는 같은 폴더에 번호를 붙인 이름으로 저장한다.

    Returns:
        tuple[Path, list[dict]]: (기록 파일 경로, 변경 목록).
            기록 파일 경로는 새로 저장한 파일, 또는 내용이 같아서 그대로 둔 가장 최근 파일이다.
            변경 목록의 key는 CHANGE_COLUMNS다. bbox는 리스트, area는 정수, 빈칸은 None이다.

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
    # 1. 기준과 결과를 비교해서 변경 목록을 만든다.
    action_order = {"add": 0, "modify": 1, "delete": 2}
    changes = []
    for key in base_records.keys() | out_records.keys():
        base, out = base_records.get(key), out_records.get(key)
        if base is None:
            action = "add"
        elif out is None:
            action = "delete"
        elif base["bbox"] != out["bbox"] or base["area"] != out["area"]:
            action = "modify"
        else:
            continue  # 바뀌지 않은 annotation
        changes.append({
            "file_name": key[0],
            "category_id": key[1],
            "action": action,
            "bbox_before": base["bbox"] if base else None,
            "bbox_after": out["bbox"] if out else None,
            "area_before": base["area"] if base else None,
            "area_after": out["area"] if out else None,
            "_json_path": (out or base)["json_path"],  # 정렬에만 쓰고 저장하지 않는다
        })
    changes.sort(key=lambda c: (action_order[c["action"]], c["_json_path"]))
    for change in changes:
        del change["_json_path"]

    # 2. 저장할 CSV 내용을 글자로 먼저 만든다.
    def cell(value):
        """CSV 칸에 적을 글자. 빈칸은 None, bbox(리스트)는 [x, y, w, h] 형식."""
        if value is None:
            return ""
        return json.dumps(value) if isinstance(value, list) else str(value)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CHANGE_COLUMNS)
    for change in changes:
        writer.writerow([cell(change[col]) for col in CHANGE_COLUMNS])
    text = buffer.getvalue()

    # 3. 같은 이름의 기존 기록 파일을 찾는다. 번호 없는 파일이 1번, <이름>_v2, _v3 …이 2번, 3번 …이다.
    csv_path = Path(csv_path)
    versions = {}
    if csv_path.is_file():
        versions[1] = csv_path
    numbered = re.compile(rf"{re.escape(csv_path.stem)}_v(\d+){re.escape(csv_path.suffix)}")
    for p in csv_path.parent.glob(f"{csv_path.stem}_v*{csv_path.suffix}"):
        match = numbered.fullmatch(p.name)
        if match and int(match.group(1)) >= 2:
            versions[int(match.group(1))] = p

    # 4. 가장 최근 파일과 내용이 같으면 저장하지 않고, 다르면 다음 번호로 저장한다.
    if versions:
        latest = max(versions)
        latest_text = versions[latest].read_bytes().decode("utf-8-sig", errors="replace")
        if latest_text.replace("\r\n", "\n") == text.replace("\r\n", "\n"):
            return versions[latest], changes
        target = csv_path.with_name(f"{csv_path.stem}_v{latest + 1}{csv_path.suffix}")
    else:
        target = csv_path
    target.write_bytes(text.encode("utf-8"))
    return target, changes


def make_fixed_annotations(fix_csv, base_ann_dir, img_dir, out_ann_dir, change_csv):
    """수정 목록대로 고친 annotation 사본과 변경 기록을 한 번에 만든다.

    아래 함수를 이 순서로 부르기만 한다. 한 단계씩 확인하거나 테스트할 때는 각 함수를 따로 쓴다.
        1. read_label_fixes(fix_csv)                 수정 목록 읽기
        2. load_annotations(base_ann_dir)            기준 annotation 목록 만들기
        3. check_label_fixes(..., img_dir)           수정 목록을 기준과 대조하고, 고치거나 만들 JSON 정하기
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

    1차 가공(원본을 기준으로)의 예, 레포 최상위에서 실행할 때:
        make_fixed_annotations(
            fix_csv="data/label_fix/label_fixes.csv",
            base_ann_dir="data/raw/train_annotations",
            img_dir="data/raw/train_images",
            out_ann_dir="data/processed/train_annotations",
            change_csv="data/label_fix/apply_label_fixes.csv",
        )

    2차 가공(1차 사본을 기준으로)을 할 때 바뀌는 것:
    - base_ann_dir에 1차 사본(data/processed/train_annotations)을 넣는다.
    - img_dir는 그대로 원본 이미지 폴더(data/raw/train_images)다. 사본에는 이미지가 없다.
    - out_ann_dir는 1차 사본과 다른 곳이어야 한다. (예: data/processed/round2/train_annotations)
    - fix_csv는 2차에 고칠 줄만 적은 목록, change_csv는 1차와 다른 이름을 쓰면 단계별 기록이 섞이지 않는다.

    Args:
        fix_csv (str | Path): 수정 목록 CSV. (예: data/label_fix/label_fixes.csv)
            열은 file_name, category_id, action, bbox, area다. (read_label_fixes 참고)
        base_ann_dir (str | Path): 기준 annotation 폴더. 읽기만 한다.
            1차 가공은 원본(data/raw/train_annotations), 2차 이후는 이전 사본을 넣는다.
            수정 목록의 각 줄을 이 폴더와 대조하고(3), 이 폴더를 사본 위치에 복사하고(4),
            add 줄의 새 JSON에 넣을 알약 정보와 이미지 정보를 이 폴더에서 읽는다(5).
        img_dir (str | Path): 이미지 폴더. 읽기만 한다. (예: data/raw/train_images)
            수정 목록의 이미지 파일이 있는지 확인하는 데만 쓴다(3). 2차 이후에도 원본 이미지 폴더를 넣는다.
        out_ann_dir (str | Path): 고친 annotation을 만들 사본 폴더. (예: data/processed/train_annotations)
            폴더 전체를 지우고 base_ann_dir에서 다시 복사한 뒤 고친다(4, 5).
            - 폴더 이름이 base_ann_dir와 같아야 한다. (train_annotations) 다르면 멈춘다.
            - base_ann_dir와 겹치면(같은 폴더이거나 한쪽이 다른 쪽 안에 있으면) 멈춘다.
            - git에 올라가지 않도록 data/processed 아래에 둔다. (.gitignore가 data/processed/*를 제외한다)
        change_csv (str | Path): 변경 기록의 1번 파일 경로. (예: data/label_fix/apply_label_fixes.csv)
            이전 기록과 내용이 같으면 저장하지 않고, 다르면 _v2, _v3 …을 붙인 새 파일로 저장한다. (write_label_changes 참고)

    Returns:
        dict: 결과 요약. (값은 지금 수정 목록으로 원본을 1차 가공했을 때의 예)
            {
                "n_base": 763,               # 기준 JSON 수
                "n_out": 771,                # 사본 JSON 수 (add 줄만큼 늘어난다)
                "applied": [...],            # apply_label_fixes()의 결과. 수정 목록 한 줄당 dict 하나 (21개)
                "change_csv": Path(...),     # 변경 기록 파일 경로. 새로 저장했거나, 내용이 같아서 그대로 둔 파일
                "changes": [...],            # write_label_changes()의 변경 목록 (21개)
            }

    Raises:
        FileNotFoundError, ValueError: 각 단계 함수의 오류를 그대로 낸다.
            메시지에 어느 줄(또는 JSON)이 왜 틀렸는지 나오고, 오류 출력(traceback)에서 어느 함수에서 멈췄는지 볼 수 있다.
    """
    fixes = read_label_fixes(fix_csv)
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
