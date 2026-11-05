/* slides.js: 발표 순서·제목·섹션 색의 단일 출처. file은 실제 파일명과 같아야 합니다. */
window.DECK = {
  "title": "경구약제 이미지 객체 검출",
  "subtitle": "코드잇 AI 15기 초급 프로젝트 1 · 2팀",
  "sections": [
    {
      "name": "표지·마무리",
      "color": "#FAB12F"
    },
    {
      "name": "문제 정의",
      "color": "#FAB12F"
    },
    {
      "name": "프로젝트 목표",
      "color": "#FAB12F"
    },
    {
      "name": "최종 결과 요약",
      "color": "#F08C00"
    },
    {
      "name": "1차 EDA",
      "color": "#5B8DEF"
    },
    {
      "name": "1차 실험",
      "color": "#1849B4"
    },
    {
      "name": "2차 EDA",
      "color": "#2BB3A3"
    },
    {
      "name": "2차 실험",
      "color": "#10828E"
    },
    {
      "name": "3차 EDA",
      "color": "#9B7BEA"
    },
    {
      "name": "3차 실험",
      "color": "#6741C7"
    },
    {
      "name": "결과와 최종 모델",
      "color": "#F08C00"
    },
    {
      "name": "한계와 개선 방안",
      "color": "#E5534B"
    },
    {
      "name": "팀원 역할 및 회고",
      "color": "#6B7280"
    },
    {
      "name": "부록",
      "color": "#9CA3AF"
    }
  ],
  "slides": [
    {
      "n": 1,
      "slug": "cover",
      "file": "01-cover.html",
      "title": "표지",
      "section": "표지·마무리"
    },
    {
      "n": 2,
      "slug": "problem",
      "file": "02-problem.html",
      "title": "알약 사진 한 장에서 위치와 이름을, 아주 정확한 box로 찾아야 합니다.",
      "section": "문제 정의"
    },
    {
      "n": 3,
      "slug": "goals",
      "file": "03-goals.html",
      "title": "검출 모델, 복약 안내 서비스, 그리고 데이터의 쓸모 판단이 목표입니다.",
      "section": "프로젝트 목표"
    },
    {
      "n": 4,
      "slug": "pipeline",
      "file": "04-pipeline.html",
      "title": "같은 데이터·같은 채점으로 비교하도록 파이프라인을 고정했습니다.",
      "section": "프로젝트 목표"
    },
    {
      "n": 5,
      "slug": "result-summary",
      "file": "05-result-summary.html",
      "title": "외부 데이터로 Kaggle 점수를 0.42에서 0.63까지 올렸습니다.",
      "section": "최종 결과 요약"
    },
    {
      "n": 6,
      "slug": "label-fix",
      "file": "06-label-fix.html",
      "title": "제공 데이터 232장, 그중 8%에서 라벨 오류를 찾아 고쳤습니다.",
      "section": "1차 EDA"
    },
    {
      "n": 7,
      "slug": "imbalance",
      "file": "07-imbalance.html",
      "title": "클래스 불균형이 52배이고, 30%는 객체가 3개뿐입니다.",
      "section": "1차 EDA"
    },
    {
      "n": 8,
      "slug": "test-only",
      "file": "08-test-only.html",
      "title": "test에만 나오는 18종 때문에 점수 상한이 0.76입니다.",
      "section": "1차 EDA"
    },
    {
      "n": 9,
      "slug": "eda1-summary",
      "file": "09-eda1-summary.html",
      "title": "1차 EDA 요약: 다섯 가지 가설로 실험합니다.",
      "section": "1차 EDA"
    },
    {
      "n": 10,
      "slug": "exp1-score",
      "file": "10-exp1-score.html",
      "title": "세 모델 모두 val은 0.92~0.99인데 Kaggle은 0.38~0.42입니다.",
      "section": "1차 실험"
    },
    {
      "n": 11,
      "slug": "exp1-tuning",
      "file": "11-exp1-tuning.html",
      "title": "증강·해상도·에폭을 바꿔도 점수는 시드 흔들림 안입니다.",
      "section": "1차 실험"
    },
    {
      "n": 12,
      "slug": "exp1-summary",
      "file": "12-exp1-summary.html",
      "title": "1차 실험 요약: 남은 원인은 데이터입니다.",
      "section": "1차 실험"
    },
    {
      "n": 13,
      "slug": "aihub-coverage",
      "file": "13-aihub-coverage.html",
      "title": "test 전용 18종을 AI Hub에서 모두 찾았습니다.",
      "section": "2차 EDA"
    },
    {
      "n": 14,
      "slug": "aihub-clean",
      "file": "14-aihub-clean.html",
      "title": "AI Hub 라벨 오류 후보를 전수 검사해 정제했습니다.",
      "section": "2차 EDA"
    },
    {
      "n": 15,
      "slug": "variants",
      "file": "15-variants.html",
      "title": "합쳐도 불균형이 남아 학습 이미지 선택 방법 3가지를 비교합니다.",
      "section": "2차 EDA"
    },
    {
      "n": 16,
      "slug": "val-design",
      "file": "16-val-design.html",
      "title": "2차 val은 test를 닮게 설계해 1,000장으로 고정했습니다.",
      "section": "2차 EDA"
    },
    {
      "n": 17,
      "slug": "eda2-summary",
      "file": "17-eda2-summary.html",
      "title": "2차 EDA 요약: 다섯 가지 가설로 실험합니다.",
      "section": "2차 EDA"
    },
    {
      "n": 18,
      "slug": "exp2-score",
      "file": "18-exp2-score.html",
      "title": "AI Hub 보충으로 Kaggle 점수가 0.2 올랐습니다.",
      "section": "2차 실험"
    },
    {
      "n": 19,
      "slug": "exp2-rank",
      "file": "19-exp2-rank.html",
      "title": "val 순위와 Kaggle 순위가 맞지 않아 val로 run을 고를 수 없습니다.",
      "section": "2차 실험"
    },
    {
      "n": 20,
      "slug": "exp2-errors",
      "file": "20-exp2-errors.html",
      "title": "클래스 혼동은 사라지고, 남은 손실은 IoU 0.95의 박스 정밀도입니다.",
      "section": "2차 실험"
    },
    {
      "n": 21,
      "slug": "exp2-summary",
      "file": "21-exp2-summary.html",
      "title": "2차 실험 요약: val 0.99와 Kaggle 0.61의 차이가 남았습니다.",
      "section": "2차 실험"
    },
    {
      "n": 22,
      "slug": "gap-hypotheses",
      "file": "22-gap-hypotheses.html",
      "title": "점수 차이의 원인 가설 8개 중 7개를 기각했습니다.",
      "section": "3차 EDA"
    },
    {
      "n": 23,
      "slug": "gap-data",
      "file": "23-gap-data.html",
      "title": "분포·번호·box 크기·누수는 원인이 아닙니다.",
      "section": "3차 EDA"
    },
    {
      "n": 24,
      "slug": "gap-diagnostic",
      "file": "24-gap-diagnostic.html",
      "title": "진단 제출로 재학습 없이 번호·채점·크기 가설을 기각했습니다.",
      "section": "3차 EDA"
    },
    {
      "n": 25,
      "slug": "gap-proxy",
      "file": "25-gap-proxy.html",
      "title": "자체 파악 기준으로 모델은 test에서 놓친 알약이 없습니다.",
      "section": "3차 EDA"
    },
    {
      "n": 26,
      "slug": "gap-summary",
      "file": "26-gap-summary.html",
      "title": "3차 EDA 요약: 남은 후보는 대회 test 정답입니다.",
      "section": "3차 EDA"
    },
    {
      "n": 27,
      "slug": "exp3-result",
      "file": "27-exp3-result.html",
      "title": "불균형 대응으로 점수가 확실히 오른 조건은 없었습니다.",
      "section": "3차 실험"
    },
    {
      "n": 28,
      "slug": "exp3-summary",
      "file": "28-exp3-summary.html",
      "title": "3차 실험 요약: 분류는 이미 포화라 불균형 대응이 닿을 곳이 없었습니다.",
      "section": "3차 실험"
    },
    {
      "n": 29,
      "slug": "stage-table",
      "file": "29-stage-table.html",
      "title": "점수를 크게 움직인 것은 데이터뿐이었습니다.",
      "section": "결과와 최종 모델"
    },
    {
      "n": 30,
      "slug": "final-model",
      "file": "30-final-model.html",
      "title": "최종 모델은 YOLO26s(v3 e20)입니다.",
      "section": "결과와 최종 모델"
    },
    {
      "n": 31,
      "slug": "demo",
      "file": "31-demo.html",
      "title": "사진을 올리면 약 정보까지 보여 주는 데모를 배포했습니다.",
      "section": "결과와 최종 모델"
    },
    {
      "n": 32,
      "slug": "limits",
      "file": "32-limits.html",
      "title": "모델 간 비교는 할 수 없고, 정답 쪽 결론은 추정입니다.",
      "section": "한계와 개선 방안"
    },
    {
      "n": 33,
      "slug": "redo",
      "file": "33-redo.html",
      "title": "처음부터 다시 한다면 이렇게 하겠습니다.",
      "section": "한계와 개선 방안"
    },
    {
      "n": 34,
      "slug": "team",
      "file": "34-team.html",
      "title": "네 명이 데이터와 모델 3계열을 나눠 맡았습니다.",
      "section": "팀원 역할 및 회고"
    },
    {
      "n": 35,
      "slug": "kpt",
      "file": "35-kpt.html",
      "title": "회고: Keep · Problem · Try",
      "section": "팀원 역할 및 회고"
    },
    {
      "n": 36,
      "slug": "closing",
      "file": "36-closing.html",
      "title": "표지 반복",
      "section": "표지·마무리"
    },
    {
      "n": 37,
      "slug": "appx-terms",
      "file": "37-appx-terms.html",
      "title": "부록: 용어",
      "section": "부록"
    },
    {
      "n": 38,
      "slug": "appx-models",
      "file": "38-appx-models.html",
      "title": "부록: 모델 3계열과 고정한 것",
      "section": "부록"
    },
    {
      "n": 39,
      "slug": "appx-imprint",
      "file": "39-appx-imprint.html",
      "title": "부록: 입력 416은 각인이 뭉개져 640을 씁니다",
      "section": "부록"
    },
    {
      "n": 40,
      "slug": "appx-split",
      "file": "40-appx-split.html",
      "title": "부록: 사진 단위로 나누면 val의 76%가 train과 같은 조합입니다",
      "section": "부록"
    },
    {
      "n": 41,
      "slug": "appx-look",
      "file": "41-appx-look.html",
      "title": "부록: 비슷하게 생긴 알약이 많고, test는 train보다 조금 어둡습니다",
      "section": "부록"
    },
    {
      "n": 42,
      "slug": "appx-exp1-errors",
      "file": "42-appx-exp1-errors.html",
      "title": "부록: 1차 오류는 희소 약과 닮은 짝에 몰립니다",
      "section": "부록"
    },
    {
      "n": 43,
      "slug": "appx-exp1-ap",
      "file": "43-appx-exp1-ap.html",
      "title": "부록: 1차 클래스별 AP",
      "section": "부록"
    },
    {
      "n": 44,
      "slug": "appx-exp1-unstable",
      "file": "44-appx-exp1-unstable.html",
      "title": "부록: 1차에서 흔들리는 클래스는 희소 약이거나 닮은 짝이 있는 약입니다",
      "section": "부록"
    },
    {
      "n": 45,
      "slug": "appx-aihub-format",
      "file": "45-appx-aihub-format.html",
      "title": "부록: AI Hub 라벨은 그대로 쓰면 에러 없이 점수만 0이 됩니다",
      "section": "부록"
    },
    {
      "n": 46,
      "slug": "appx-aihub-dup",
      "file": "46-appx-aihub-dup.html",
      "title": "부록: AI Hub와 Kaggle 사이에 같은 파일은 없습니다",
      "section": "부록"
    },
    {
      "n": 47,
      "slug": "appx-variants",
      "file": "47-appx-variants.html",
      "title": "부록: 방법별 train, val 클래스 분포",
      "section": "부록"
    },
    {
      "n": 48,
      "slug": "appx-test-freq",
      "file": "48-appx-test-freq.html",
      "title": "부록: test 클래스 빈도와 val 비율",
      "section": "부록"
    },
    {
      "n": 49,
      "slug": "appx-exp2-runs",
      "file": "49-appx-exp2-runs.html",
      "title": "부록: 2차 run 표 (YOLO26s, RT-DETR-l)",
      "section": "부록"
    },
    {
      "n": 50,
      "slug": "appx-exp2-curves",
      "file": "50-appx-exp2-curves.html",
      "title": "부록: 2차 학습 곡선",
      "section": "부록"
    },
    {
      "n": 51,
      "slug": "appx-exp2-errors",
      "file": "51-appx-exp2-errors.html",
      "title": "부록: 2차 오류 분석",
      "section": "부록"
    },
    {
      "n": 52,
      "slug": "appx-exp2-ap",
      "file": "52-appx-exp2-ap.html",
      "title": "부록: 클래스별 AP와 train 객체 수의 관계는 없습니다 (상관 0.08)",
      "section": "부록"
    },
    {
      "n": 53,
      "slug": "appx-exp2-unstable",
      "file": "53-appx-exp2-unstable.html",
      "title": "부록: 세 모델이 같은 클래스에서 흔들립니다 (캡슐·긴 타원 정제)",
      "section": "부록"
    },
    {
      "n": 54,
      "slug": "appx-box-crops",
      "file": "54-appx-box-crops.html",
      "title": "부록: 출처별 box 차이는 알약이 놓인 방향의 차이입니다",
      "section": "부록"
    },
    {
      "n": 55,
      "slug": "appx-exp3-design",
      "file": "55-appx-exp3-design.html",
      "title": "부록: 3차 실험 설계",
      "section": "부록"
    },
    {
      "n": 56,
      "slug": "appx-exp3-curves",
      "file": "56-appx-exp3-curves.html",
      "title": "부록: 3차 학습 곡선은 네 조건이 거의 겹칩니다",
      "section": "부록"
    },
    {
      "n": 57,
      "slug": "appx-exp3-ap",
      "file": "57-appx-exp3-ap.html",
      "title": "부록: 3차 클래스별 AP 변화는 seed만 바꾼 변화와 같은 범위입니다",
      "section": "부록"
    },
    {
      "n": 58,
      "slug": "appx-refs",
      "file": "58-appx-refs.html",
      "title": "부록: 참고문헌",
      "section": "부록"
    }
  ]
};
