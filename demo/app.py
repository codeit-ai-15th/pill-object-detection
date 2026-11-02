"""알약 검출 데모 (Gradio). 사진 한 장을 올리면 bbox·제품명·confidence를 그리고, 약학정보원 검색 링크와 허가정보를 보여 준다.

실행 (레포 최상위에서). 브라우저에서 http://127.0.0.1:7860 을 연다.
    uv run demo/app.py

준비
- .env에 쓸 가중치 경로를 적는다(계열마다 하나). 이름과 예시는 .env.example. 비어 있는 계열은 화면에 안 나온다.
    PILL_DEMO_YOLO, PILL_DEMO_RTDETR, PILL_DEMO_FRCNN
- 허가정보 캐시 demo/drug_info.json은 git에 있다. 없으면 demo/build_drug_info.py로 만든다(DATA_GO_API_KEY 필요).

화면
- 위: 모델 선택(환경 변수로 정해진 계열별 하나. 파일 이름을 같이 보여 준다), confidence 기준값 슬라이더.
- 왼쪽: 사진 업로드와 결과 이미지. 오른쪽: 검출된 알약 종류마다 카드(이름, confidence, 약학정보원 검색, 효능효과·용법용량·주의사항).
- 모델이나 기준값을 바꾸면 올린 사진으로 바로 다시 그린다. 모델은 한 번만 불러오고, 같은 사진·모델의 예측은 다시 돌리지 않는다.

HF Space(Gradio SDK, ZeroGPU 하드웨어)에서도 같은 파일이 돈다. 무료 플랜에서는 Streamlit·Docker SDK와 CPU basic을 쓸 수 없어서다.
ZeroGPU는 `spaces` 패키지의 @spaces.GPU가 붙은 함수(infer)가 도는 동안만 GPU를 준다. 로컬에는 spaces가 없으므로 그냥 함수로 돈다.
pill_detection이 설치되어 있지 않으면 레포의 src/를 sys.path에 넣는다.
"""

import hashlib
import html
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # Space가 모듈로 불러도 같은 폴더의 predict 등을 찾도록
try:
    import pill_detection  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import spaces  # HF ZeroGPU Space에만 있다. torch보다 먼저 import해야 한다
except ImportError:  # 로컬
    spaces = None

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from PIL import ImageOps  # noqa: E402

import draw  # noqa: E402
import drug_info  # noqa: E402
import predict  # noqa: E402

load_dotenv(ROOT / ".env")


def infer(detector, image):
    """추론 한 번. ZeroGPU Space에서는 이 함수가 도는 동안만 GPU를 받는다(@spaces.GPU). 로컬에서는 그냥 함수다."""
    return detector.predict(image, conf=predict.RAW_CONF)


if spaces is not None:
    infer = spaces.GPU(infer)

DEFAULT_CONF = 0.5
TITLE = "💊 알약 검출 데모"
CSS = """
.pill-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; }
.pill-card h4 { margin: 0 0 4px 0; font-size: 1.1em; }
.pill-card .meta { color: var(--body-text-color-subdued); font-size: 0.9em; margin-bottom: 6px; }
.pill-card a.search { display: inline-block; padding: 4px 10px; border: 1px solid var(--border-color-primary);
                      border-radius: 6px; text-decoration: none; font-size: 0.9em; margin-bottom: 6px; }
.pill-card details { margin-top: 6px; } .pill-card summary { cursor: pointer; font-weight: 600; }
.pill-card details p { margin: 4px 0; font-size: 0.92em; line-height: 1.45; }
"""


def card_html(card, dets):
    """검출된 알약 한 종류의 카드 HTML.

    Args:
        card (dict): drug_info.describe()의 결과.
        dets (list[dict]): 그 클래스의 예측 목록(점수 내림차순).

    Returns:
        str: HTML 조각.
    """
    esc = html.escape
    count = f" ×{len(dets)}" if len(dets) > 1 else ""
    parts = [f"<div class='pill-card'><h4>{esc(card['name'])}{count}</h4>",
             f"<div class='meta'>confidence <b>{max(d['score'] for d in dets):.2f}</b> · <code>{esc(card['k_code'])}</code>"]
    meta = " · ".join(x for x in (card["etc_otc"], card["entp_name"]) if x)
    parts.append(f"<br>{esc(meta)}</div>" if meta else "</div>")
    if card["is_others"]:
        parts.append("<div class='meta'>학습 클래스 밖 알약으로 묶인 예측입니다. 이름과 정보가 없습니다.</div></div>")
        return "".join(parts)
    parts.append(f"<a class='search' href='{esc(card['url'])}' target='_blank' rel='noopener'>약학정보원에서 검색 ↗</a>")
    if not card["has_info"]:
        note = ("식약처 허가정보에 없는 품목입니다 (허가 취소·수출용 등). 약학정보원 검색을 이용하세요." if card["cached"]
                else "허가정보 캐시에 없음 (<code>uv run demo/build_drug_info.py</code>로 만듭니다)")
        parts.append(f"<div class='meta'>{note}</div></div>")
        return "".join(parts)
    for title, lines, open_ in (("효능효과", card["efficacy"], True), ("용법용량", card["usage"], True),
                                ("사용상 주의사항", card["caution"], False)):
        if lines:
            body = "".join(f"<p>{esc(line)}</p>" for line in lines)
            parts.append(f"<details{' open' if open_ else ''}><summary>{title}</summary>{body}</details>")
    if card["storage"]:
        parts.append(f"<div class='meta'>저장방법: {esc(card['storage'])}</div>")
    parts.append("</div>")
    return "".join(parts)


def group_by_class(detections):
    """클래스별로 묶는다. 카드는 종류마다 하나이고 최고 점수 순."""
    groups = defaultdict(list)
    for det in detections:
        groups[det["category_id"]].append(det)
    return sorted(groups.items(), key=lambda kv: -max(d["score"] for d in kv[1]))


class Demo:
    """모델·예측 캐시와 화면 동작. Gradio 콜백이 이 객체의 메서드를 부른다."""

    def __init__(self, slots, classes, info):
        self.slots = slots
        self.classes = classes
        self.info = info
        self.usable = [slot for slot in slots.values() if slot.usable]
        self.by_label = {self.label_for(slot): slot for slot in self.usable}
        self._detectors = {}
        self._predictions = {}

    @staticmethod
    def label_for(slot):
        return f"{slot.label} · {slot.path.name}"

    def notes(self):
        """쓸 수 없는 계열의 안내 문구(마크다운)."""
        lines = []
        for slot in self.slots.values():
            if slot.path is None:
                lines.append(f"- {slot.label}: `{slot.env_var}` 비어 있음")
            elif slot.error:
                lines.append(f"- ⚠️ {slot.label}: {slot.error.splitlines()[0]}")
        return "\n".join(lines)

    def detector(self, slot):
        """계열 하나의 모델. 처음 한 번만 불러온다."""
        key = (slot.family, str(slot.path), slot.path.stat().st_mtime)
        if key not in self._detectors:
            self._detectors[key] = predict.load_model(slot.path, slot.family)
        return self._detectors[key]

    def predictions(self, slot, image):
        """같은 모델·같은 사진이면 다시 추론하지 않는다. 기준값과 무관하게 낮은 점수까지 받아 둔다."""
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        key = (slot.family, str(slot.path), slot.path.stat().st_mtime, digest)
        if key not in self._predictions:
            started = time.perf_counter()
            dets = infer(self.detector(slot), image)
            self._predictions[key] = (dets, time.perf_counter() - started)
        return self._predictions[key]

    def name_of(self, category_id):
        return drug_info.describe(category_id, self.classes, self.info)["name"]

    def run(self, image, model_label, conf):
        """콜백. (결과 이미지, 카드 HTML, 상태 마크다운)을 돌려준다."""
        if image is None:
            return None, "", "사진을 올리면 알약마다 박스, 제품명, confidence가 표시됩니다."
        slot = self.by_label.get(model_label) or self.usable[0]
        image = ImageOps.exif_transpose(image).convert("RGB")
        try:
            detections, elapsed = self.predictions(slot, image)
            detector = self.detector(slot)
        except predict.DemoError as e:
            raise gr.Error(str(e)) from None
        shown = [d for d in detections if d["score"] >= conf]
        annotated = draw.draw_detections(image, shown, self.name_of)
        cards = "".join(card_html(drug_info.describe(cid, self.classes, self.info), dets) for cid, dets in group_by_class(shown))
        if not shown:
            cards = "<div class='pill-card'>검출된 알약이 없습니다. confidence 기준값을 낮춰 보세요.</div>"
        variant = f" · {detector.variant}" if detector.variant else ""
        status = (f"**검출된 알약 {len(shown)}개** · {slot.label} `{slot.path.name}` · 클래스 {detector.n_classes}개{variant} · "
                  f"추론 {elapsed:.2f}초")
        return annotated, cards, status


def build():
    """Gradio Blocks를 만든다. 가중치 설정이 잘못됐으면 오류만 보여 주는 화면을 만든다."""
    try:
        slots = predict.configured_weights()
    except predict.DemoError as e:
        with gr.Blocks(title="알약 검출 데모") as app:
            gr.Markdown(f"# {TITLE}\n\n⚠️ {html.escape(str(e))}".replace("\n[해결]", "\n\n[해결]"))
        return app
    demo = Demo(slots, drug_info.load_classes(), drug_info.load_drug_info())
    n_info = sum(1 for v in demo.info.values() if not v.get("missing"))
    with gr.Blocks(title="알약 검출 데모") as app:
        gr.Markdown(f"# {TITLE}\n알약 사진 한 장을 올리면 알약마다 박스·제품명·confidence를 그리고, 약학정보원 검색 링크와 "
                    f"식약처 허가정보를 보여 줍니다. 허가정보 캐시 {n_info}/{len(demo.classes) - 1}개.")
        if not demo.usable:
            gr.Markdown("⚠️ 쓸 수 있는 모델이 없습니다. 아래 안내를 보고 .env를 고친 뒤 다시 켜세요.\n" + demo.notes())
            return app
        labels = list(demo.by_label)
        with gr.Row():
            model = gr.Radio(choices=labels, value=labels[0], label="모델", info="환경 변수(PILL_DEMO_*)가 정한 가중치. 바꾸면 바로 다시 예측합니다",
                             scale=3)
            conf = gr.Slider(0.0, 1.0, value=DEFAULT_CONF, step=0.05, label="confidence 기준값", info="이 값보다 낮은 예측은 숨깁니다", scale=2)
        if demo.notes():
            gr.Markdown(demo.notes())
        with gr.Row():
            with gr.Column(scale=3):
                image_in = gr.Image(type="pil", sources=["upload"], label="알약 사진 한 장을 올리세요", height=260)
                status = gr.Markdown("사진을 올리면 알약마다 박스, 제품명, confidence가 표시됩니다.")
                image_out = gr.Image(type="pil", label="검출 결과", interactive=False)
            with gr.Column(scale=2):
                cards = gr.HTML(label="검출된 알약")
        for component in (image_in, model, conf):
            component.change(demo.run, inputs=[image_in, model, conf], outputs=[image_out, cards, status])
    return app


if __name__ == "__main__":
    build().launch(css=CSS, show_error=True, inbrowser=False)
