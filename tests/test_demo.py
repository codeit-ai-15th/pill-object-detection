"""데모(demo/)의 모델 선택·계열 판별·이름 변환·검색 URL·허가정보 파서를 가중치와 네트워크 없이 본다.

실제 가중치는 .env의 PILL_DEMO_* 변수가 가리킬 때만 추가로 본다(없으면 건너뜀).
"""

import os
import sys

import pytest
import torch

from helpers import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "demo"))

import build_drug_info as bdi  # noqa: E402
import draw  # noqa: E402
import drug_info  # noqa: E402
import predict  # noqa: E402
from pill_detection.datasets import OTHERS_CATEGORY_ID  # noqa: E402

CLASSES = drug_info.load_classes()


def fake_frcnn(path, n_outputs=97):
    torch.save({"model": {"roi_heads.box_predictor.cls_score.weight": torch.zeros(n_outputs, 1024),
                          "backbone.body.conv1.weight": torch.zeros(1)}, "epoch": 1}, path)
    return path


def fake_ultralytics(path, rtdetr=False):
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

    cls = RTDETRDetectionModel if rtdetr else DetectionModel
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    torch.save({"model": model, "epoch": 1}, path)
    return path


# ---- 계열 판별 ----

def test_detect_family(tmp_path):
    assert predict.detect_family(fake_frcnn(tmp_path / "a_frcnn.pt")) == "frcnn"
    assert predict.detect_family(fake_ultralytics(tmp_path / "b_yolo.pt")) == "yolo"
    assert predict.detect_family(fake_ultralytics(tmp_path / "c_rtdetr.pt", rtdetr=True)) == "rtdetr"


def test_detect_family_rejects_unknown(tmp_path):
    torch.save({"weights": [1, 2, 3]}, tmp_path / "weird.pt")
    with pytest.raises(predict.DemoError, match="weird.pt"):
        predict.detect_family(tmp_path / "weird.pt")
    torch.save({"model": {"foo": torch.zeros(1)}}, tmp_path / "notfrcnn.pt")
    with pytest.raises(predict.DemoError, match="Faster R-CNN"):
        predict.detect_family(tmp_path / "notfrcnn.pt")


# ---- 환경 변수로 모델 고르기 ----

def test_configured_weights_all_empty():
    with pytest.raises(predict.DemoError, match="PILL_DEMO_YOLO"):
        predict.configured_weights({})


def test_configured_weights(tmp_path):
    yolo = fake_ultralytics(tmp_path / "exp_yolo.pt")
    slots = predict.configured_weights({"PILL_DEMO_YOLO": str(yolo), "PILL_DEMO_FRCNN": str(tmp_path / "nope.pt")})
    assert list(slots) == ["yolo", "rtdetr", "frcnn"]
    assert slots["yolo"].usable and slots["yolo"].path == yolo
    assert slots["rtdetr"].path is None and not slots["rtdetr"].usable
    assert slots["frcnn"].path is not None and "PILL_DEMO_FRCNN" in slots["frcnn"].error and not slots["frcnn"].usable


def test_configured_weights_relative_to_root():
    slots = predict.configured_weights({"PILL_DEMO_YOLO": "runs/models/x.pt"})
    assert slots["yolo"].path == PROJECT_ROOT / "runs" / "models" / "x.pt"


def test_load_model_family_mismatch(tmp_path):
    path = fake_frcnn(tmp_path / "f.pt")
    with pytest.raises(predict.DemoError, match="PILL_DEMO_FRCNN"):
        predict.load_model(path, "yolo")
    with pytest.raises(predict.DemoError, match="없습니다"):
        predict.load_model(tmp_path / "missing.pt", "yolo")


# ---- Faster R-CNN label → category_id ----

def test_variant_from_n_classes():
    assert predict.variant_from_n_classes(118) == "v1"
    assert predict.variant_from_n_classes(96) == "v3"
    assert predict.variant_from_n_classes(75) == "v4"
    with pytest.raises(predict.DemoError, match="해결"):
        predict.variant_from_n_classes(56)


@pytest.mark.parametrize("variant,n", [("v1", 118), ("v3", 96), ("v4", 75)])
def test_frcnn_category_table(variant, n):
    table = predict.frcnn_category_table(variant, CLASSES)
    assert len(table) == n and min(table) == 1 and max(table) == n
    for label, category_id in table.items():
        assert CLASSES[category_id]["idx"][variant] == label - 1
    if variant == "v4":
        assert table[n] == OTHERS_CATEGORY_ID
    else:
        assert OTHERS_CATEGORY_ID not in table.values()


# ---- Ultralytics 클래스 이름 → category_id ----

def test_category_from_model_name():
    assert predict.category_from_model_name("3351", CLASSES) == 3351
    assert predict.category_from_model_name("others", CLASSES) == OTHERS_CATEGORY_ID
    assert predict.category_from_model_name("보령부스파정 5mg", CLASSES) == 1900
    assert predict.category_from_model_name("보령부스파정\xa05mg ", CLASSES) == 1900
    with pytest.raises(predict.DemoError, match="없는약"):
        predict.category_from_model_name("없는약", CLASSES)


# ---- 약학정보원 검색 ----

def test_search_name_and_url():
    assert drug_info.search_name("글리틴정(콜린알포세레이트)") == "글리틴정"
    assert drug_info.search_name("삐콤씨에프정 618.6mg/병") == "삐콤씨에프정 618.6mg"
    assert drug_info.search_name("종근당글리아티린연질캡슐(콜린알포세레이트)\xa0") == "종근당글리아티린연질캡슐"
    assert drug_info.search_name("일양하이트린정 2mg") == "일양하이트린정 2mg"
    url = drug_info.health_kr_url("일양하이트린정 2mg")
    assert url.startswith("https://www.health.kr/searchResult?search_word=") and url.endswith("&search_flag=all")
    assert "%EC%9D%BC%EC%96%91" in url and " " not in url


def test_describe_others_and_unknown_cache():
    card = drug_info.describe(OTHERS_CATEGORY_ID, CLASSES, {})
    assert card["is_others"] and card["url"] is None and not card["has_info"]
    card = drug_info.describe(1900, CLASSES, {})
    assert card["name"] == "보령부스파정 5mg" and card["url"] and not card["has_info"]
    assert not card["cached"]
    card = drug_info.describe(1900, CLASSES, {1900: {"efficacy": ["a"], "usage": [], "caution": [], "etc_otc": "전문의약품"}})
    assert card["has_info"] and card["cached"] and card["etc_otc"] == "전문의약품"
    card = drug_info.describe(1900, CLASSES, {1900: {"efficacy": [], "usage": [], "caution": [], "missing": True}})
    assert card["cached"] and not card["has_info"]


# ---- 그리기 ----

def test_open_image_applies_exif_rotation():
    """세로로 찍은 휴대폰 사진(EXIF orientation 6)은 바로 세워져 세로 이미지가 된다."""
    import io

    from PIL import Image

    image = Image.new("RGB", (40, 20), "white")  # 가로로 저장된 원본
    exif = Image.Exif()
    exif[0x0112] = 6  # 시계 방향 90도 회전해서 봐야 하는 사진
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    assert draw.open_image(buffer.getvalue()).size == (20, 40)


def test_draw_detections_korean_label():
    from PIL import Image

    image = Image.new("RGB", (976, 1280), "gray")
    out = draw.draw_detections(image, [{"category_id": 1900, "bbox": [100, 100, 200, 150], "score": 0.93},
                                       {"category_id": OTHERS_CATEGORY_ID, "bbox": [10, 5, 50, 50], "score": 0.5}],
                               lambda c: "보령부스파정 5mg" if c == 1900 else "미등록 알약")
    assert out.size == image.size and out is not image
    assert out.getpixel((100, 170)) != (128, 128, 128)  # 박스 선이 그려짐
    assert draw.load_font(28).getname()[0].lower().startswith("nanum")  # 동봉 글꼴을 썼다


# ---- 화면 카드 HTML ----

def test_card_html():
    import app

    dets = [{"category_id": 1900, "bbox": [0, 0, 10, 10], "score": 0.93}, {"category_id": 1900, "bbox": [0, 0, 10, 10], "score": 0.7}]
    card = drug_info.describe(1900, CLASSES, {1900: {"efficacy": ["<효능>"], "usage": ["용법"], "caution": [], "etc_otc": "전문의약품",
                                                     "entp_name": "보령", "storage": "실온"}})
    out = app.card_html(card, dets)
    assert "보령부스파정 5mg ×2" in out and "0.93" in out and "&lt;효능&gt;" in out and "target='_blank'" in out
    assert "<details class='sec' open><summary>효능효과" in out and "사용상 주의사항" not in out and "실온" in out
    caution = app.card_html(drug_info.describe(1900, CLASSES, {1900: {"efficacy": [], "usage": [], "caution": ["■ 1. 금기", "■", "본문"]}}),
                            dets[:1])
    assert "<details class='sec caution'><summary>사용상 주의사항" in caution  # 주의사항은 접힌 채 경고색 박스
    assert "<p class='sub'>1. 금기</p><p>본문</p>" in caution  # ■ 줄은 기호 없이 소제목, 기호만 있는 줄은 버림
    others = app.card_html(drug_info.describe(OTHERS_CATEGORY_ID, CLASSES, {}), dets[:1])
    assert "미등록 알약" in others and "약학정보원" not in others
    missing = app.card_html(drug_info.describe(1900, CLASSES, {1900: {"efficacy": [], "usage": [], "caution": [], "missing": True}}), dets[:1])
    assert "허가정보에 없는 품목" in missing and "약학정보원에서 검색" in missing


# ---- 허가정보 DOC 파서 ----

DOC_XML = """<DOC title="효능효과" type="EE"><SECTION title=""><ARTICLE title="1. 고혈압"><PARAGRAPH tagName="p">본태성 고혈압</PARAGRAPH>
<PARAGRAPH tagName="p"><![CDATA[2차성 고혈압 &amp; 신성 고혈압]]></PARAGRAPH></ARTICLE><ARTICLE title=""><PARAGRAPH>  </PARAGRAPH></ARTICLE></SECTION></DOC>"""


def test_parse_doc_xml():
    assert bdi.parse_doc(DOC_XML) == ["■ 1. 고혈압", "본태성 고혈압", "2차성 고혈압 & 신성 고혈압"]
    assert bdi.parse_doc("") == [] and bdi.parse_doc(None) == []


def test_parse_doc_html_fallback():
    assert bdi.parse_doc("<p>첫 줄</p><br>둘째 &lt;줄&gt;") == ["첫 줄", "둘째 <줄>"]


def test_resolve_endpoint():
    assert bdi.resolve_endpoint(None, None) == bdi.DEFAULT_ENDPOINT
    assert bdi.resolve_endpoint("https://x/svc/getOp", "https://y") == "https://x/svc/getOp"
    assert bdi.resolve_endpoint(None, "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/") == \
        "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08/" + bdi.DETAIL_OPERATION
    assert bdi.resolve_endpoint(None, "https://apis.data.go.kr/1471000/Svc09/getDrugPrdtPrmsnDtlInq08") == \
        "https://apis.data.go.kr/1471000/Svc09/getDrugPrdtPrmsnDtlInq08"
    assert bdi.resolve_endpoint(None, "apis.data.go.kr/1471000/Svc08") == "https://apis.data.go.kr/1471000/Svc08/" + bdi.DETAIL_OPERATION


def test_items_unwrap():
    assert bdi._items({"header": {"resultCode": "03"}, "body": {"items": None}}) == []
    assert bdi._items({"header": {"resultCode": "00"}, "body": {"items": [{"ITEM_SEQ": "1"}]}}) == [{"ITEM_SEQ": "1"}]
    assert bdi._items({"header": {"resultCode": "00"}, "body": {"items": {"item": {"ITEM_SEQ": "2"}}}}) == [{"ITEM_SEQ": "2"}]
    with pytest.raises(RuntimeError, match="resultCode=30"):
        bdi._items({"header": {"resultCode": "30", "resultMsg": "SERVICE KEY"}})


# ---- 캐시 파일(있을 때) ----

def test_drug_info_json_if_present():
    info = drug_info.load_drug_info()
    if not info:
        pytest.skip("demo/drug_info.json이 없음 (uv run demo/build_drug_info.py로 만듭니다)")
    expected = {c for c in CLASSES if c != OTHERS_CATEGORY_ID}
    assert set(info) <= expected
    for entry in info.values():
        assert {"k_code", "name", "item_seq", "efficacy", "usage", "caution", "missing"} <= set(entry)


# ---- 실제 가중치(있을 때) ----

@pytest.mark.parametrize("family", ["yolo", "rtdetr", "frcnn"])
def test_real_weights_if_configured(family):
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    value = os.environ.get(predict.ENV_VARS[family], "")
    if not value or not predict.configured_weights()[family].usable:
        pytest.skip(f"{predict.ENV_VARS[family]}가 비어 있거나 파일이 없음")
    detector = predict.load_model(predict.configured_weights()[family].path, family)
    assert detector.kind == family
    assert set(detector.category_of.values()) <= set(CLASSES)
    if family == "frcnn":
        assert detector.variant in ("v1", "v3", "v4")
