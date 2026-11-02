---
title: 알약 검출 데모
emoji: 💊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: demo/app.py
pinned: false
---

# 알약 검출 데모

알약 사진 한 장을 올리면 알약마다 bbox, 제품명, confidence를 그려 주고, 약학정보원 검색 바로가기와 식약처 허가정보(효능효과·용법용량·주의사항)를 보여 줍니다.

- 코드와 설명: [GitHub 레포](https://github.com/codeit-ai-15th/pill-object-detection)의 `demo/`
- 모델은 Space 변수 `PILL_DEMO_YOLO`, `PILL_DEMO_RTDETR`, `PILL_DEMO_FRCNN`이 가리키는 `weights/` 안의 파일입니다. `demo/deploy_space.py`가 올립니다.
