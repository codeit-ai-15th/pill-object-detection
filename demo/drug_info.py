"""데모가 보여 줄 약 정보: 클래스 표, 허가정보 캐시, 약학정보원 검색 링크.

- 클래스 표는 data/datasets/classes.csv다. category_id(K-코드 숫자) → K-코드, 제품명, 그룹.
- 허가정보 캐시는 demo/drug_info.json이다. demo/build_drug_info.py가 식약처 의약품 제품 허가정보 API로 한 번 받아 둔 것이고,
  데모는 이 파일만 읽는다(라이브 중 외부 호출 없음). 파일 형식은 build_drug_info.py 모듈 docstring에 있다.
- 약학정보원은 세부 페이지 코드를 우리 데이터로 알 수 없어서 제품명 검색 URL만 만든다.
"""

import csv
import json
import unicodedata
from pathlib import Path
from urllib.parse import quote

from pill_detection.datasets import OTHERS_CATEGORY_ID

ROOT = Path(__file__).resolve().parents[1]
CLASSES_CSV = ROOT / "data" / "datasets" / "classes.csv"
CACHE_JSON = Path(__file__).resolve().parent / "drug_info.json"
HEALTH_KR_SEARCH = "https://www.health.kr/searchResult?search_word={}&search_flag=all"
OTHERS_LABEL = "미등록 알약"


def normalize_name(name):
    """제품명 비교용 정규화. NFC, NBSP → 공백, 연속 공백 하나로, 양끝 공백 제거."""
    text = unicodedata.normalize("NFC", str(name)).replace("\xa0", " ")
    return " ".join(text.split())


def load_classes(path=CLASSES_CSV):
    """classes.csv를 읽는다.

    Args:
        path (str | Path): classes.csv 경로.

    Returns:
        dict[int, dict]: category_id → {"k_code", "name", "group", "competition"(bool), "idx": {"v1": int|None, ...}}.
            others 행의 category_id는 pill_detection.datasets.OTHERS_CATEGORY_ID다.
    """
    classes = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category_id = int(row["category_id"])
            classes[category_id] = {
                "k_code": row["k_code"], "name": normalize_name(row["name"]), "group": row["group"],
                "competition": row["competition"] == "1",
                "idx": {v: (int(row[f"idx_{v}"]) if row[f"idx_{v}"] != "" else None) for v in ("v1", "v2", "v3", "v4")},
            }
    return classes


def load_drug_info(path=CACHE_JSON):
    """허가정보 캐시를 읽는다. 파일이 없으면 빈 dict.

    Args:
        path (str | Path): drug_info.json 경로.

    Returns:
        dict[int, dict]: category_id → 캐시 항목(build_drug_info.py 모듈 docstring의 형식).
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def search_name(name):
    """약학정보원 검색어. 첫 괄호와 '/' 앞까지만 쓴다. 꼬리가 붙은 이름은 검색이 안 되는 경우가 있다.

    예) "글리틴정(콜린알포세레이트)" → "글리틴정", "삐콤씨에프정 618.6mg/병" → "삐콤씨에프정 618.6mg"
    """
    text = normalize_name(name)
    for sep in ("(", "/"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip()


def health_kr_url(name):
    """약학정보원 통합 검색 URL. 새 탭에서 열면 검색 결과가 나온다."""
    return HEALTH_KR_SEARCH.format(quote(search_name(name)))


def describe(category_id, classes, drug_info):
    """카드 하나에 필요한 정보를 모은다.

    Args:
        category_id (int): 모델이 낸 K-코드 숫자. others는 OTHERS_CATEGORY_ID.
        classes (dict): load_classes()의 결과.
        drug_info (dict): load_drug_info()의 결과.

    Returns:
        dict: {"category_id", "k_code", "name", "is_others", "url"(없으면 None), "cached"(캐시에 항목이 있음),
               "has_info"(효능·용법·주의 중 하나라도 있음), "etc_otc", "entp_name", "efficacy", "usage", "caution", "storage"}.
            문단 값은 문자열 리스트. cached가 True인데 has_info가 False면 허가정보 API에 없는 품목(허가 취소·수출용 등)이다.
    """
    if category_id == OTHERS_CATEGORY_ID:
        return {"category_id": category_id, "k_code": "others", "name": OTHERS_LABEL, "is_others": True, "url": None,
                "cached": False, "has_info": False, "etc_otc": "", "entp_name": "", "efficacy": [], "usage": [], "caution": [],
                "storage": ""}
    row = classes.get(category_id)
    name = row["name"] if row else f"K-{category_id:06d}"
    info = drug_info.get(category_id) or {}
    has_info = bool(info) and not info.get("missing") and any(info.get(k) for k in ("efficacy", "usage", "caution"))
    return {"category_id": category_id, "k_code": row["k_code"] if row else f"K-{category_id:06d}", "name": name,
            "is_others": False, "url": health_kr_url(name), "cached": bool(info), "has_info": has_info,
            "etc_otc": info.get("etc_otc", ""), "entp_name": info.get("entp_name", ""),
            "efficacy": info.get("efficacy", []), "usage": info.get("usage", []), "caution": info.get("caution", []),
            "storage": info.get("storage", "")}
