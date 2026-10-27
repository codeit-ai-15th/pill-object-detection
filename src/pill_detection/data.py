"""데이터를 읽고 고치는 함수.

지금은 라벨 수정 사본 만들기에 쓰는 아래 함수가 있다. 앞의 세 함수는 파일을 읽기만 하고, copy_dataset부터 파일을 지우고 쓴다.
- read_label_fixes: 수정 목록 CSV를 읽고 CSV 안의 형식을 검사한다.
- load_annotations: annotation 폴더의 JSON을 모두 읽어 (이미지 파일명, category_id)별 목록을 만든다.
- check_label_fixes: 수정 목록의 각 줄이 원본과 맞는지 확인하고, 줄마다 고치거나 새로 만들 JSON을 정한다.
- copy_dataset: 사본 폴더의 하위 폴더를 지우고, 원본에서 그대로 다시 복사한다.

annotation 폴더 구조
- Kaggle train_annotations는 알약 하나당 JSON 하나다. 경로: <조합>_json/K-<코드>/<이미지 파일명>.json
- JSON 하나에 images, annotations, categories가 1개씩 들어 있다.
- 한 이미지의 라벨은 그 이미지에 있는 알약 수(3~4개)만큼의 JSON에 나뉘어 있다.

라벨 수정 사본
- 원본(data/raw)은 읽기만 하고, 원본을 복사한 사본(data/processed)의 JSON만 수정 목록대로 고친다.
- 수정 목록은 고친 뒤의 annotation 값을 원본 JSON과 같은 이름과 형식으로 적은 CSV다.
  지금 위치: notebooks/eda/label_fix/label_fixes.csv
    - file_name: 이미지 파일명. 원본 JSON의 images[0].file_name과 같다.
    - category_id: 클래스. K-코드 숫자다. (K-003351 → 3351)
    - action: add(라벨 JSON이 없는 알약의 라벨을 새로 추가) 또는 modify(있는 라벨의 bbox와 area를 바꿈)
    - bbox: 고친 뒤의 [x, y, w, h]. 원본 픽셀이다.
    - area: 고친 뒤의 면적. bbox의 w × h와 같다.
- 수정 목록의 한 줄은 (file_name, category_id)로 구분한다. annotation 하나는 한 줄에서만 고친다.
"""

import csv
import json
import shutil
from pathlib import Path

# 원본(data/raw)과 사본(data/processed)에 있는 하위 폴더. copy_dataset()이 기본으로 복사하는 폴더다.
DATA_SUB_DIRS = ("train_images", "train_annotations", "test_images")

# 수정 목록 CSV의 열. 이름과 순서가 모두 같아야 한다.
FIX_COLUMNS = ["file_name", "category_id", "action", "bbox", "area"]

# add: 라벨 JSON이 없는 알약의 라벨을 새로 추가, modify: 있는 라벨의 bbox와 area를 바꿈
FIX_ACTIONS = ("add", "modify")


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
        >>> fixes = read_label_fixes("notebooks/eda/label_fix/label_fixes.csv")
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


def copy_dataset(raw_dir, out_dir, sub_dirs=DATA_SUB_DIRS):
    """사본 폴더의 하위 폴더를 지우고, 원본에서 그대로 다시 복사한다.

    매번 지우고 다시 복사하므로, 사본은 항상 원본과 같은 상태에서 시작한다.
    지우는 것은 out_dir 안의 sub_dirs 폴더뿐이고, out_dir의 다른 파일(.gitkeep 등)은 그대로 둔다.
    sub_dirs 폴더 안에 직접 넣어 둔 파일이 있으면 함께 지워진다.

    Args:
        raw_dir (str | Path): 원본 폴더. (예: data/raw) 이 아래에 sub_dirs 폴더가 모두 있어야 한다.
        out_dir (str | Path): 사본 폴더. (예: data/processed) 없으면 만든다.
        sub_dirs (tuple[str, ...]): 복사할 하위 폴더 이름. 기본값은 DATA_SUB_DIRS(세 폴더 모두)다.

    Returns:
        dict[str, int]: 하위 폴더 이름을 key로, 복사한 파일 수를 value로 담는다.
            예: {"train_images": 232, "train_annotations": 763, "test_images": 842}

    Raises:
        FileNotFoundError: 원본에 sub_dirs 폴더 중 없는 것이 있을 때. 아무것도 지우지 않고 멈춘다.
        ValueError: 원본 폴더와 사본 폴더가 겹칠 때(같은 폴더이거나 한쪽이 다른 쪽 안에 있을 때).
            사본 폴더를 지우는 작업이 원본에 닿을 수 있어서, 아무것도 지우지 않고 멈춘다.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)

    # 지우기 전에 모두 확인한다. 오류 메시지에는 폴더 이름만 적는다. (내 컴퓨터의 절대경로가 나오지 않도록)
    missing = [name for name in sub_dirs if not (raw_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"원본 폴더({raw_dir.name})에 없는 하위 폴더가 있습니다: {missing}\n"
            f"원본 폴더 경로가 맞는지, 데이터 압축을 그 폴더 바로 아래에 풀었는지 확인하세요."
        )

    raw_abs, out_abs = raw_dir.resolve(), out_dir.resolve()
    if raw_abs == out_abs or out_abs.is_relative_to(raw_abs) or raw_abs.is_relative_to(out_abs):
        raise ValueError(
            f"원본 폴더({raw_dir.name})와 사본 폴더({out_dir.name})가 겹칩니다. 사본을 지우면 원본이 지워질 수 있어서 멈춥니다.\n"
            f"사본 폴더를 원본과 따로 떨어진 곳으로 지정하세요. (예: 원본 data/raw, 사본 data/processed)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in sub_dirs:
        target = out_dir / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(raw_dir / name, target)
        counts[name] = sum(1 for p in target.rglob("*") if p.is_file())
    return counts
