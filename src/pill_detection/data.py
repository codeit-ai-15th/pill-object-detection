"""라벨 로딩, AI Hub 조합 라벨 읽기와 변환, 조합 단위 분할.

라벨 구조
- Kaggle train_annotations는 알약 하나당 JSON 하나다. 경로: <조합>_json/K-<코드>/<이미지 파일명>.json
- 이미지 하나의 라벨이 알약 수(3~4개)만큼 흩어져 있어서, load_annotations()로 이미지 단위로 합쳐 쓴다.
- 조합 이름은 이미지 파일명의 첫 "_" 앞부분이다. (K-001900-016548-019607-029451)
- 같은 조합의 사진은 같은 알약 배치를 다른 각도에서 찍은 것이라, 분할은 조합 단위로 한다.

AI Hub 조합 라벨이 Kaggle 라벨과 다른 점 (notebooks/eda/00_raewoo_first_eda.ipynb 6-2)
- Kaggle의 category_id와 dl_idx는 K-코드 숫자(drug_N)다. AI Hub 원본의 dl_idx는 K-코드와 달라서 drug_N으로 만든다.
- AI Hub 원본은 image_id, category_id, annotation id가 모두 1이다.
- AI Hub 원본에는 bbox가 빈 "-오류" 폴더, " copy" 중복 파일, macOS "._" 파일이 섞여 있다.
"""

import json
import os
import random
import unicodedata
import zipfile

# 대회에서 사용을 금지한 AI Hub 파일 (RULE.md)
BANNED_AIHUB_FILES = ("TL_2_조합.zip", "TS_2_조합.zip")


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


def load_annotations(ann_dir):
    """알약별 annotation JSON을 이미지 단위로 합친다.

    Kaggle train_annotations와 convert_aihub_to_kaggle()로 만든 폴더에 같이 쓴다.

    Args:
        ann_dir (str): annotation 폴더. 하위 폴더를 모두 찾는다.

    Returns:
        dict[str, dict]: 이미지 파일명을 key로, 그 이미지의 라벨을 value로 담는다. key는 이미지 수만큼 있다.
            value 하나의 구조는 아래와 같다. (값은 Kaggle train 기준)

            {
                "image": {...},           # 이미지 메타데이터. 이 이미지의 첫 JSON에 있던 images[0]
                "annotations": [          # 알약 하나당 dict 하나. K-코드 폴더 이름 순서
                    {
                        "id": 133,                    # annotation id
                        "image_id": 34,               # 바깥 "image"의 "id"와 같다
                        "category_id": 1900,          # 클래스. K-코드 숫자 (K-001900 → 1900)
                        "bbox": [644, 845, 189, 190], # [x, y, w, h] 픽셀. 왼쪽 위 꼭짓점과 너비·높이
                        "area": 35910,                # 원본 값 그대로. w×h와 다를 때가 있다
                        "iscrowd": 0, "ignore": 0, "segmentation": [],  # 원본 값 그대로
                        "image": {...},               # 이 알약 JSON의 images[0]
                        "json_path": "data/raw/...",  # 이 annotation을 읽은 JSON 경로
                    },
                    ...                               # 알약 수만큼 (보통 3~4개)
                ],
            }

            "image" 메타데이터에는 필드가 두 종류 섞여 있다.
            - 이미지 필드: 같은 이미지의 JSON끼리 값이 같다.
                - file_name, width, height, id, camera_la(촬영 각도 70/75/90), camera_lo, drug_dir(앞면/뒷면),
                  back_color, light_color, drug_S, size, imgfile
            - 알약 필드: JSON(알약)마다 값이 다르다.
                - drug_N(K-코드), dl_idx, dl_name(약 이름), print_front/print_back(각인), color_class1, drug_shape 등
            - 바깥 "image"의 알약 필드는 첫 알약의 값일 뿐이다. 알약 정보는 각 annotation의 "image"에서 읽는다.

            라벨이 누락된 이미지는 "annotations" 수가 파일명의 알약 수보다 적다. (Kaggle train 8장, EDA 2-1)

    Examples:
        >>> records = load_annotations("data/raw/train_annotations")
        >>> len(records)  # 이미지 수 (JSON 763개를 이미지 232장으로 합침)
        232
        >>> record = records["K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"]
        >>> record["image"]["id"], record["image"]["width"], record["image"]["height"]
        (34, 976, 1280)
        >>> [(a["category_id"], a["bbox"]) for a in record["annotations"]]  # 알약 4개, bbox는 xywh
        [(1900, [644, 845, 189, 190]), (16548, [144, 799, 239, 239]),
         (19607, [657, 287, 155, 150]), (29451, [108, 218, 392, 264])]
        >>> ann = record["annotations"][0]
        >>> ann["id"], ann["image_id"], ann["image"]["drug_N"], ann["image"]["dl_name"]
        (133, 34, 'K-001900', '보령부스파정 5mg')
        >>> ann["json_path"]
        'data/raw/train_annotations/K-001900-016548-019607-029451_json/K-001900/K-001900-016548-019607-029451_0_2_0_2_70_000_200.json'
    """
    records = {}
    for root, dirs, files in os.walk(ann_dir):
        dirs.sort()  # 운영체제와 관계없이 같은 순서로 읽도록
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            image = data["images"][0]
            record = records.setdefault(image["file_name"], {"image": image, "annotations": []})
            for ann in data["annotations"]:
                record["annotations"].append({**ann, "image": image, "json_path": path})
    return records


def check_aihub_file(path):
    """대회에서 금지한 AI Hub 파일이면 오류를 낸다.

    macOS는 한글 파일명을 자모 분리형(NFD)으로 줄 때가 있어, 비교 전에 NFC로 맞춘다.

    Args:
        path (str): AI Hub zip 경로.

    Raises:
        ValueError: TL_2_조합.zip, TS_2_조합.zip일 때.

    Examples:
        >>> check_aihub_file("data/external/aihub/train_annotations/TL_1_조합.zip")  # 허용 파일은 아무것도 돌려주지 않는다
        >>> check_aihub_file("data/external/aihub/train_annotations/TL_2_조합.zip")
        Traceback (most recent call last):
        ValueError: TL_2_조합.zip은 대회에서 사용을 금지한 파일입니다(RULE.md). 이 파일을 지우고 다른 조합 zip만 사용하세요: rm 'data/external/aihub/train_annotations/TL_2_조합.zip'
    """
    name = unicodedata.normalize("NFC", os.path.basename(path))
    if name in BANNED_AIHUB_FILES:
        raise ValueError(
            f"{name}은 대회에서 사용을 금지한 파일입니다(RULE.md). "
            f"이 파일을 지우고 다른 조합 zip만 사용하세요: rm '{path}'"
        )


def read_aihub_label_zip(zip_path):
    """AI Hub 조합 라벨 zip을 풀지 않고 읽어 이미지 단위로 합친다.

    - macOS "._" 파일은 건너뛴다.
    - " copy" 중복 파일은 같은 이미지·같은 알약의 원본 JSON을 먼저 쓰고 나머지는 버린다.
    - category_id는 drug_N(K-코드)의 숫자로 바꾼다. 원본 dl_idx는 K-코드와 다르다.
    - 학습에 쓰면 안 되는 이미지는 "errors"에 이유를 적는다.
        - "오류 폴더": "-오류" 폴더에 들어 있는 알약이 있다. (원본에서 bbox가 비어 있다)
        - "JSON 오류": 읽을 수 없는 JSON이 있다.
        - "bbox 오류": 길이 4가 아니거나 너비·높이가 0 이하인 bbox가 있다.
        - "라벨 누락": 파일명에 적힌 알약 수보다 라벨이 적다.

    Args:
        zip_path (str): TL_*_조합.zip 또는 VL_*_조합.zip 경로.

    Returns:
        dict[str, dict]: 이미지 파일명 → {"zip": zip 파일명, "annotations": [{"category_id", "bbox", "image"}],
            "errors": 이유 목록}. annotation의 "image"는 그 알약 JSON의 이미지 메타데이터다.

    Raises:
        ValueError: 대회에서 금지한 파일일 때.

    Examples:
        >>> records = read_aihub_label_zip("data/external/aihub/train_annotations/TL_1_조합.zip")
        >>> len(records), sum(not r["errors"] for r in records.values())  # (전체 이미지, 오류 없는 이미지)
        (1503, 1461)
        >>> record = records["K-000250-000573-002483-006192_0_2_0_2_70_000_200.png"]
        >>> record["zip"], record["errors"]
        ('TL_1_조합.zip', [])
        >>> [(a["category_id"], a["bbox"]) for a in record["annotations"]]  # category_id = K-코드 숫자
        [(250, [553, 184, 272, 279]), (573, [71, 188, 304, 298]),
         (2483, [92, 666, 218, 488]), (6192, [524, 586, 359, 499])]
        >>> image = record["annotations"][0]["image"]  # 원본 메타데이터는 고치지 않고 그대로 둔다
        >>> image["drug_N"], image["dl_idx"], image["dl_name"]
        ('K-000250', '249', '마그밀정(수산화마그네슘)')
        >>> records["K-000250-000573-002483-012778_0_2_0_2_70_000_200.png"]["errors"]  # 알약 하나의 bbox가 비어 있음
        ['bbox 오류', '라벨 누락']
    """
    check_aihub_file(zip_path)
    zip_name = unicodedata.normalize("NFC", os.path.basename(zip_path))
    records = {}
    seen = set()
    with zipfile.ZipFile(zip_path) as zf:
        names = [unicodedata.normalize("NFC", n) for n in zf.namelist()]
        raw_names = dict(zip(names, zf.namelist()))
        # 원본 JSON을 " copy" 파일보다 먼저 읽는다
        json_names = sorted((n for n in names if n.endswith(".json")), key=lambda n: (" copy" in n, n))
        for name in json_names:
            base = os.path.basename(name)
            if base.startswith("._"):
                continue
            stem = base[: -len(".json")].split(" copy")[0]
            file_name = stem + ".png"
            record = records.setdefault(file_name, {"zip": zip_name, "annotations": [], "errors": []})
            if name.split("/")[-2].endswith("-오류"):
                record["errors"].append("오류 폴더")
                continue
            try:
                data = json.loads(zf.read(raw_names[name]))
            except json.JSONDecodeError:
                record["errors"].append("JSON 오류")
                continue
            image = data["images"][0]
            code = int(image["drug_N"][2:])
            if (file_name, code) in seen:
                continue
            seen.add((file_name, code))
            for ann in data["annotations"]:
                bbox = ann.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    record["errors"].append("bbox 오류")
                    continue
                record["annotations"].append({"category_id": code, "bbox": bbox, "image": image})
    for file_name, record in records.items():
        if len({a["category_id"] for a in record["annotations"]}) < len(pill_codes(file_name)):
            record["errors"].append("라벨 누락")
        record["errors"] = sorted(set(record["errors"]))
    return records


def convert_aihub_to_kaggle(records, out_dir, start_image_id, start_annotation_id):
    """read_aihub_label_zip() 결과를 Kaggle train_annotations와 같은 구조의 JSON으로 저장한다.

    대회 측 가공(DATA.md)과 같게 id를 다시 매긴다. "errors"가 있는 이미지는 저장하지 않는다.
    - image_id: start_image_id부터 이미지마다 하나씩
    - category_id, dl_idx: K-코드 숫자 (Kaggle과 같음)
    - annotation id: start_annotation_id부터 annotation마다 하나씩

    Args:
        records (dict[str, dict]): read_aihub_label_zip()의 결과. 여러 zip의 결과를 합쳐도 된다.
        out_dir (str): 저장할 폴더. data/external/aihub_converted/train_annotations에 저장해
            data/raw/train_annotations와 같은 구조로 둔다. load_annotations()로 다시 읽을 수 있다.
        start_image_id (int): 첫 image_id. 대회 train과 test가 1~1500을 나눠 쓰므로(test는 파일명 숫자) 1500보다 크게 준다.
        start_annotation_id (int): 첫 annotation id. Kaggle train의 최댓값(5691)보다 크게 준다.

    Returns:
        int: 저장한 이미지 수. 이미지 하나에 알약 수만큼 JSON 파일이 생긴다.

    Examples:
        >>> records = read_aihub_label_zip("data/external/aihub/train_annotations/TL_1_조합.zip")
        >>> convert_aihub_to_kaggle(records, "data/external/aihub_converted/train_annotations", 1501, 5692)  # 오류 이미지 42장은 건너뜀
        1461

        파일명 순서로 저장하므로, 첫 이미지(알약 4개)는 image_id 1501, annotation id 5692~5695가 된다.
        첫 알약의 파일과 내용 (Kaggle train_annotations와 같은 형식):

        data/external/aihub_converted/train_annotations/K-000250-000573-002483-006192_json/K-000250/
            K-000250-000573-002483-006192_0_2_0_2_70_000_200.json
        {"images": [{"file_name": "K-000250-000573-002483-006192_0_2_0_2_70_000_200.png", ...,
                     "drug_N": "K-000250", "dl_idx": "250", "id": 1501}],
         "type": "instances",
         "annotations": [{"area": 75888, "iscrowd": 0, "bbox": [553, 184, 272, 279], "category_id": 250,
                          "ignore": 0, "segmentation": [], "id": 5692, "image_id": 1501}],
         "categories": [{"supercategory": "pill", "id": 250, "name": "마그밀정(수산화마그네슘)"}]}
    """
    image_id, annotation_id, saved = start_image_id, start_annotation_id, 0
    for file_name in sorted(records):
        record = records[file_name]
        if record["errors"]:
            continue
        for ann in record["annotations"]:
            code = ann["category_id"]
            x, y, w, h = ann["bbox"]
            data = {
                "images": [{**ann["image"], "id": image_id, "dl_idx": str(code)}],
                "type": "instances",
                "annotations": [
                    {
                        "area": w * h,
                        "iscrowd": 0,
                        "bbox": [x, y, w, h],
                        "category_id": code,
                        "ignore": 0,
                        "segmentation": [],
                        "id": annotation_id,
                        "image_id": image_id,
                    }
                ],
                "categories": [{"supercategory": "pill", "id": code, "name": ann["image"]["dl_name"]}],
            }
            folder = os.path.join(out_dir, f"{combo_name(file_name)}_json", ann["image"]["drug_N"])
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, file_name.replace(".png", ".json")), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            annotation_id += 1
        image_id += 1
        saved += 1
    return saved


def make_group_split(image_classes, val_ratio=0.2, seed=42, n_trials=200):
    """조합 단위로 train/val을 나눈다.

    같은 조합의 사진은 같은 알약 배치라, 이미지 단위로 나누면 val 점수가 부풀려진다.
    조합 순서를 n_trials번 섞어 보고, train과 val 양쪽에 있는 클래스 수가 가장 많은 분할을 고른다.
    섞는 순서가 seed로 정해지므로 누구나 같은 분할을 얻는다.

    Args:
        image_classes (dict[str, list[int]]): 이미지 파일명 → 그 이미지에 있는 클래스 목록.
        val_ratio (float): val에 넣을 조합 비율.
        seed (int): 섞는 순서를 정하는 seed.
        n_trials (int): 섞어 볼 횟수.

    Returns:
        tuple[list[str], list[str]]: (train 이미지 파일명 목록, val 이미지 파일명 목록). 각각 정렬되어 있다.

    Examples:
        >>> records = load_annotations("data/raw/train_annotations")
        >>> image_classes = {fn: [a["category_id"] for a in r["annotations"]] for fn, r in records.items()}
        >>> image_classes["K-001900-016548-019607-029451_0_2_0_2_70_000_200.png"]
        [1900, 16548, 19607, 29451]
        >>> train, val = make_group_split(image_classes)  # 조합 114개 중 23개가 val
        >>> len(train), len(val)  # 이미지 수
        (184, 48)
        >>> val[:3]  # 같은 조합의 각도별 사진은 같은 쪽에 모인다
        ['K-001900-016548-019607-029451_0_2_0_2_70_000_200.png',
         'K-001900-016548-019607-029451_0_2_0_2_75_000_200.png',
         'K-001900-016548-019607-029451_0_2_0_2_90_000_200.png']
    """
    combo_images = {}
    for file_name in sorted(image_classes):
        combo_images.setdefault(combo_name(file_name), []).append(file_name)
    combos = sorted(combo_images)
    n_val = round(len(combos) * val_ratio)
    rng = random.Random(seed)
    best_score, best_val = -1, None
    for _ in range(n_trials):
        rng.shuffle(combos)
        val_combos = set(combos[:n_val])
        train_classes, val_classes = set(), set()
        for combo, files in combo_images.items():
            target = val_classes if combo in val_combos else train_classes
            for file_name in files:
                target.update(image_classes[file_name])
        score = len(train_classes & val_classes)
        if score > best_score:
            best_score, best_val = score, val_combos
    train = [f for c in sorted(combo_images) if c not in best_val for f in combo_images[c]]
    val = [f for c in sorted(combo_images) if c in best_val for f in combo_images[c]]
    return train, val
