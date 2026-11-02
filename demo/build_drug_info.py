"""허가정보 캐시(demo/drug_info.json) 만들기. 식약처 의약품 제품 허가정보 API를 클래스마다 한 번 부른다.

데모는 라이브 중 외부 API를 부르지 않고 이 파일만 읽는다. 키 없이도 데모가 돌도록 결과 파일은 git에 올린다.

준비
- 공공데이터포털(https://www.data.go.kr/data/15095677/openapi.do)에서 "의약품 제품 허가정보" 활용신청(개발계정 자동승인).
- 일반 인증키(Decoding)를 .env의 DATA_GO_API_KEY에 넣는다. Encoding 키를 넣어도 된다(% 가 있으면 풀어서 쓴다).

실행 (레포 최상위에서)
    uv run demo/build_drug_info.py                       # 118개 클래스. 이미 있는 클래스는 건너뜀
    uv run demo/build_drug_info.py --force               # 전부 다시 받음
    uv run demo/build_drug_info.py --data-dir ../pill-object-detection/data   # 별도 워크트리처럼 data/가 없을 때
    uv run demo/build_drug_info.py --endpoint https://apis.data.go.kr/1471000/<서비스>/<오퍼레이션>   # API 버전이 바뀌었을 때
  주소는 --endpoint, .env의 DATA_GO_BASE_URL(서비스 주소만 있으면 상세 조회 오퍼레이션을 붙임), 기본값 순으로 정한다.

어떻게 받나
- 클래스 목록은 data/datasets/classes.csv(others 제외 118개). 품목기준코드(item_seq)·제품명·전문/일반·제조사는 annotation JSON에서 읽는다.
  Kaggle 사본(data/processed/train_annotations)과 AI Hub 사본(data/processed/ai_hub/train_annotations)을 먼저 보고, 없으면
  원본(data/raw/train_annotations, data/external/aihub/raw/train_annotations)을 본다. 폴더 이름이 <조합>_json/<K-코드>/라서
  K-코드 폴더마다 JSON 하나만 읽는다.
- 상세 조회 오퍼레이션에 item_seq로 묻는다. 0건이면 ITEM_SEQ_ALIASES(같은 제품의 새 허가 코드)와 제품명으로 더 묻고, 그래도 없으면
  missing으로 적는다. 허가가 취소됐거나 수출용인 품목은 이 API에 없다(2026-11-02 기준 118개 중 7개).
- 효능효과(EE_DOC_DATA), 용법용량(UD_DOC_DATA), 사용상 주의사항(NB_DOC_DATA)은 <DOC><SECTION><ARTICLE title><PARAGRAPH> XML이라
  ARTICLE 제목("■ 제목" 줄)과 PARAGRAPH 글을 문단 리스트로 푼다. XML이 아니면 태그를 떼고 줄로 나눈다.

결과 파일 형식: {"<category_id>": {...}}
    k_code, name(우리 라벨 제품명), item_seq, api_name(API의 ITEM_NAME), api_item_seq(실제로 받은 품목기준코드. 별칭이면 다름),
    entp_name, etc_otc(전문/일반), chart(성상),
    material(원료), main_ingr(주성분), storage(저장방법), efficacy[], usage[], caution[], missing(bool), fetched_at, source_op
"""

import argparse
import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # pill_detection을 설치하지 않은 환경(Space)용
from dotenv import load_dotenv  # noqa: E402

from drug_info import CACHE_JSON, CLASSES_CSV, ROOT, load_classes, normalize_name  # noqa: E402
from pill_detection.datasets import OTHERS_CATEGORY_ID  # noqa: E402

# 상세 조회 오퍼레이션. 서비스 번호(08)와 오퍼레이션 번호는 data.go.kr가 가끔 올린다.
# 주소는 --endpoint > .env의 DATA_GO_BASE_URL(서비스까지면 오퍼레이션을 붙임) > 아래 기본값 순으로 정한다.
DEFAULT_SERVICE = "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08"
DETAIL_OPERATION = "getDrugPrdtPrmsnDtlInq08"  # 2026-11-02 실제 호출로 확인
DEFAULT_ENDPOINT = f"{DEFAULT_SERVICE}/{DETAIL_OPERATION}"


def resolve_endpoint(arg, base_url):
    """상세 조회 주소를 정한다.

    Args:
        arg (str | None): --endpoint 값.
        base_url (str | None): .env의 DATA_GO_BASE_URL. 서비스 주소(…/DrugPrdtPrmsnInfoService08)거나 오퍼레이션까지 든 주소.

    Returns:
        str: 오퍼레이션까지 든 주소.
    """
    base = (arg or base_url or "").strip().rstrip("/")
    if not base:
        return DEFAULT_ENDPOINT
    if not re.match(r"^https?://", base):  # 스킴 없이 적은 주소
        base = "https://" + base
    if arg:
        return base
    return base if "/get" in base else f"{base}/{DETAIL_OPERATION}"
# 라벨의 품목기준코드가 허가정보 API에 없을 때 대신 조회할 코드. 같은 제품이 다른 허가로 다시 나온 경우만 적는다.
ITEM_SEQ_ALIASES = {
    "199303108": "202106092",  # 타이레놀정500mg: 한국얀센 허가(취소) → 켄뷰코리아 '타이레놀정500밀리그람(아세트아미노펜)'
}
ANNOTATION_DIRS = [  # (data-dir 기준 상대 경로). 앞에서부터 있는 것을 쓴다
    ("processed/train_annotations", "processed/ai_hub/train_annotations"),
    ("raw/train_annotations", "external/aihub/raw/train_annotations"),
]
K_CODE_RE = re.compile(r"^K-\d{6}$")


def parse_doc(text):
    """허가정보 DOC XML을 문단 리스트로 푼다.

    Args:
        text (str | None): EE_DOC_DATA 같은 필드 값.

    Returns:
        list[str]: ARTICLE 제목은 "■ 제목", 본문은 PARAGRAPH 하나가 한 줄. 빈 줄은 뺀다.
    """
    if not text or not str(text).strip():
        return []
    text = str(text).strip()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        plain = html.unescape(re.sub(r"<[^>]+>", "\n", text))
        return [line.strip() for line in plain.splitlines() if line.strip()]
    lines = []
    for article in root.iter("ARTICLE"):
        title = (article.get("title") or "").strip()
        if title:
            lines.append(f"■ {html.unescape(title)}")
        for paragraph in article.iter("PARAGRAPH"):
            body = html.unescape("".join(paragraph.itertext())).strip()
            body = re.sub(r"<[^>]+>", "", body).strip()  # CDATA 안에 HTML이 섞인 문서
            if body:
                lines.append(body)
    if not lines:  # ARTICLE 없이 PARAGRAPH만 있는 문서
        for paragraph in root.iter("PARAGRAPH"):
            body = html.unescape("".join(paragraph.itertext())).strip()
            if body:
                lines.append(body)
    return lines


def collect_label_meta(data_dir, classes):
    """annotation JSON에서 클래스마다 item_seq·제품명·전문/일반·제조사를 읽는다.

    Args:
        data_dir (Path): data/ 폴더.
        classes (dict): load_classes()의 결과.

    Returns:
        dict[int, dict]: category_id → {"item_seq", "dl_name", "etc_otc", "company"}. 못 찾은 클래스는 빠진다.
    """
    wanted = {row["k_code"]: category_id for category_id, row in classes.items() if category_id != OTHERS_CATEGORY_ID}
    found = {}
    for group in ANNOTATION_DIRS:
        dirs = [data_dir / d for d in group if (data_dir / d).is_dir()]
        if not dirs:
            continue
        for ann_dir in dirs:
            for k_dir in sorted(p for p in ann_dir.glob("*/K-*") if p.is_dir() and K_CODE_RE.match(p.name)):
                category_id = wanted.get(k_dir.name)
                if category_id is None or category_id in found:
                    continue
                for json_path in sorted(k_dir.glob("*.json")):
                    with open(json_path, encoding="utf-8") as f:
                        images = json.load(f).get("images") or []
                    image = next((im for im in images if im.get("dl_mapping_code") == k_dir.name), None)
                    if image and image.get("item_seq"):
                        found[category_id] = {"item_seq": str(image["item_seq"]).strip(), "dl_name": normalize_name(image.get("dl_name", "")),
                                              "etc_otc": image.get("di_etc_otc_code", ""), "company": image.get("dl_company", "")}
                        break
        if len(found) == len(wanted):
            break
    return found


def _items(payload):
    """data.go.kr 응답(JSON)에서 items 리스트를 꺼낸다. 0건이면 []."""
    response = payload.get("response", payload)
    header = response.get("header") or {}
    code = str(header.get("resultCode", ""))
    if code in ("03", "300"):
        return []
    if code not in ("00", "0"):
        raise RuntimeError(f"API 오류 resultCode={code} resultMsg={header.get('resultMsg')}")
    items = (response.get("body") or {}).get("items")
    if not items:
        return []
    if isinstance(items, dict):
        items = items.get("item") or []
    if isinstance(items, dict):
        items = [items]
    return [item.get("item", item) if isinstance(item, dict) else item for item in items]


def fetch(endpoint, key, params, retries=2, timeout=30):
    """상세 조회를 한 번 부른다. 실패하면 retries번 더 시도한다.

    Returns:
        list[dict]: items. 0건이면 [].
    """
    query = {"serviceKey": key, "type": "json", "pageNo": 1, "numOfRows": 3, **params}
    url = f"{endpoint}?{urlencode(query)}"
    last = None
    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "pill-detection-demo"}), timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            try:
                return _items(json.loads(raw))
            except json.JSONDecodeError:
                snippet = re.sub(r"\s+", " ", raw)[:300]
                raise RuntimeError(f"JSON이 아닌 응답입니다: {snippet}\n[해결] 키가 맞는지, 오퍼레이션 주소가 맞는지 확인하세요 "
                                   f"(data.go.kr 페이지의 상세기능 → --endpoint).") from None
        except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
            last = e
            if isinstance(e, HTTPError) and e.code in (400, 404):
                body = re.sub(r"\s+", " ", e.read().decode("utf-8", errors="replace"))[:200]
                raise RuntimeError(f"{e.code}: 오퍼레이션 주소나 키가 잘못됐습니다: {endpoint}\n    응답: {body}\n"
                                   "[해결] data.go.kr 페이지의 상세기능에서 요청 주소를 확인해 .env의 DATA_GO_BASE_URL이나 "
                                   "--endpoint로 넣고, 키는 일반 인증키(Decoding)인지 확인하세요.") from None
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"요청 실패: {last}")


def to_entry(category_id, row, meta, item, endpoint):
    """캐시 항목 하나를 만든다. item이 None이면 missing."""
    entry = {"k_code": row["k_code"], "name": row["name"], "item_seq": meta.get("item_seq", ""),
             "api_name": "", "api_item_seq": "", "entp_name": meta.get("company", ""), "etc_otc": meta.get("etc_otc", ""),
             "chart": "", "material": "", "main_ingr": "", "storage": "",
             "efficacy": [], "usage": [], "caution": [], "missing": item is None,
             "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_op": endpoint}
    if item:
        entry.update({"api_name": item.get("ITEM_NAME", ""), "api_item_seq": str(item.get("ITEM_SEQ", "")),
                      "entp_name": item.get("ENTP_NAME") or entry["entp_name"],
                      "etc_otc": item.get("ETC_OTC_CODE") or entry["etc_otc"], "chart": item.get("CHART", ""),
                      "material": item.get("MATERIAL_NAME", ""), "main_ingr": item.get("MAIN_ITEM_INGR", ""),
                      "storage": item.get("STORAGE_METHOD", ""), "efficacy": parse_doc(item.get("EE_DOC_DATA")),
                      "usage": parse_doc(item.get("UD_DOC_DATA")), "caution": parse_doc(item.get("NB_DOC_DATA"))})
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description="허가정보 캐시(demo/drug_info.json) 만들기")
    parser.add_argument("--data-dir", default=str(ROOT / "data"), help="annotation JSON이 있는 data/ 폴더 (기본: 레포의 data/)")
    parser.add_argument("--out", default=str(CACHE_JSON), help="결과 JSON (기본: demo/drug_info.json)")
    parser.add_argument("--endpoint", default=None, help=f"상세 조회 오퍼레이션 주소 (기본: .env의 DATA_GO_BASE_URL, 없으면 {DEFAULT_ENDPOINT})")
    parser.add_argument("--force", action="store_true", help="이미 받은 클래스도 다시 받음")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N개 클래스만 (시험용)")
    parser.add_argument("--sleep", type=float, default=0.2, help="요청 사이 대기 초")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("DATA_GO_API_KEY") or "").strip()
    if not key:
        print("[오류] DATA_GO_API_KEY가 없습니다.\n[해결] https://www.data.go.kr/data/15095677/openapi.do 에서 활용신청 후 "
              "일반 인증키(Decoding)를 .env의 DATA_GO_API_KEY에 넣으세요. (.env.example 참고)", file=sys.stderr)
        return 1
    if "%" in key:
        key = unquote(key)
    args.endpoint = resolve_endpoint(args.endpoint, os.environ.get("DATA_GO_BASE_URL"))
    print(f"상세 조회 주소: {args.endpoint}")

    classes = load_classes(CLASSES_CSV)
    targets = [c for c in classes if c != OTHERS_CATEGORY_ID]
    if args.limit:
        targets = targets[: args.limit]
    data_dir = Path(args.data_dir)
    meta = collect_label_meta(data_dir, classes)
    missing_meta = [classes[c]["k_code"] for c in targets if c not in meta]
    if missing_meta:
        print(f"[오류] annotation JSON에서 item_seq를 못 찾은 클래스 {len(missing_meta)}개: {', '.join(missing_meta[:10])}\n"
              f"[해결] --data-dir가 annotation이 있는 data/ 폴더인지 확인하세요 (지금: {data_dir}).", file=sys.stderr)
        return 1

    out = Path(args.out)
    cache = {}
    if out.is_file():
        with open(out, encoding="utf-8") as f:
            cache = json.load(f)
    counts = {"fetched": 0, "missing": 0, "skipped": 0, "failed": 0}
    for i, category_id in enumerate(targets, 1):
        key_str = str(category_id)
        row = classes[category_id]
        if not args.force and key_str in cache and not cache[key_str].get("missing"):
            counts["skipped"] += 1
            continue
        try:
            item_seq = meta[category_id]["item_seq"]
            items = fetch(args.endpoint, key, {"item_seq": item_seq})
            if not items and item_seq in ITEM_SEQ_ALIASES:
                items = fetch(args.endpoint, key, {"item_seq": ITEM_SEQ_ALIASES[item_seq]})
            if not items:
                items = fetch(args.endpoint, key, {"item_name": meta[category_id]["dl_name"]})
            item = items[0] if items else None
        except RuntimeError as e:
            counts["failed"] += 1
            print(f"[{i}/{len(targets)}] {row['k_code']} {row['name']}: 실패 {e}", file=sys.stderr)
            if "JSON이 아닌" in str(e) or "주소나 키가 잘못" in str(e):
                break  # 키·주소 문제면 전부 같은 결과라 멈춘다
            continue
        cache[key_str] = to_entry(category_id, row, meta[category_id], item, args.endpoint)
        counts["missing" if item is None else "fetched"] += 1
        print(f"[{i}/{len(targets)}] {row['k_code']} {row['name']}: {'없음' if item is None else '받음'}")
        with open(out, "w", encoding="utf-8") as f:  # 중간에 멈춰도 받은 것은 남긴다
            json.dump(dict(sorted(cache.items(), key=lambda kv: int(kv[0]))), f, ensure_ascii=False, indent=1)
        time.sleep(args.sleep)
    print(f"받음 {counts['fetched']}, 없음 {counts['missing']}, 건너뜀 {counts['skipped']}, 실패 {counts['failed']} → {out} ({len(cache)}개)")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
