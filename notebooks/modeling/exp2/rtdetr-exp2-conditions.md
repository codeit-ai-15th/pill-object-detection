# RT-DETR 2차 실험(AI Hub 데이터 추가) 조건 기록

- 기록 기준: [`rtdetr-exp2.ipynb`](rtdetr-exp2.ipynb)(브랜치 `exp/rtdetr-l-exp2-data`)의 실행 출력, run 폴더의 `args.yaml`, Ultralytics 8.4.183 소스
- 1차 조건 기록: [`../exp1/rtdetr-exp1-conditions.md`](../exp1/rtdetr-exp1-conditions.md). 이 문서의 1~8번은 1차와 같은 항목 순서로 적고, 1차와 같은 값은 그 문서를 가리킨다.
- 학습 기간: 2026-10-31 ~ 2026-11-02. 학습 폴더를 만든 커밋 4fdae56(10-31) 이후에 학습했고, 11-02에 6개 run의 채점까지 끝난 것을 확인했다.
- 전체 학습 인자는 run 폴더마다 `runs/exp2/<run 이름>/args.yaml`에 있다. 1차와 달리 optimizer와 관련 값을 직접 적어서(4-2), `args.yaml`의 값이 학습에 실제로 쓰인 값이다.

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| 운영체제 | Windows, VS Code 주피터, uv 가상환경 |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| Ultralytics | 8.4.183 |
| W&B (wandb) | 0.31.0 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12GB |
| GPU 메모리 | batch 16에서 학습 중 12.7G(시간 측정 run, 본 학습도 12.7~13G). 전용 메모리 12GB를 조금 넘어 시스템 메모리를 같이 쓴다. 속도에만 영향 |
| albumentations | 학습 로그에 적용 목록 줄이 없다. 1차와 같이 설치되지 않은 것으로 보인다 |

## 2. 데이터

| 방법 | 고르는 방식 | 모델 클래스 | 그중 대회 밖 클래스 | train 사진 | train 객체 |
|---|---|---:|---:|---:|---:|
| v2 | 클래스별 200개를 목표로 고른 사진 | 118 | 44 | 7,129 | 26,734 |
| v3 | 대회 클래스가 든 사진만 | 96 | 22 | 8,974 | 33,872 |
| v4 | 사진 전부, 대회 밖 알약은 `others` 한 클래스 | 75 | `others` 1 | 11,206 | 42,800 |

| 항목 | 값 |
|---|---|
| 원본 | 대회 train 232장(라벨 수정 반영)과 AI Hub 사진. 팀 목록 CSV(`data/datasets/images.csv`, `classes.csv`)로 방법마다 고른다 |
| 학습 폴더 | 터미널에서 `uv run scripts/build_dataset.py --variant <방법> --out data/processed/exp2/<방법>`. 사진은 원본에 하드링크를 건다. 만든 커밋은 4fdae56이다 |
| 대회 클래스 | 74종 = Kaggle train에 있던 56종 + test에만 나오는 18종. 18종은 AI Hub 사진으로만 배운다 |
| v4의 `others` | 대회 밖 알약 44종을 한 클래스로 묶는다. v4 train 객체 42,800개 중 16,552개(39%) |
| val | 세 방법 모두 같은 1,000장(Kaggle 47장 + AI Hub 953장, 객체 3,790개). Kaggle 47장은 1차 val과 같다 |
| val의 채점 대상 | 대회 클래스 알약 3,053개(대회 사진 159개, AI Hub 2,894개). 나머지 737개는 대회 밖 알약이라 채점에서 무시 영역으로 쓴다(5번) |
| 사진 크기 | test 842장, 대회 train 232장, val 1,000장 모두 976 × 1280 |
| 데이터 양 | train 사진이 1차(185장)의 38~61배 |

## 3. 모델

| 항목 | 값 |
|---|---|
| 모델 | RT-DETR-L (Ultralytics `RTDETR`) |
| 시작 가중치 | `weights/rtdetr-l.pt`(COCO 80종 사전학습). 1차와 같은 파일 |
| 학습 방식 | 풀 파인튜닝(`freeze=None`). 사전학습 가중치 941개 중 926개를 가져오고, 클래스 수에 따라 크기가 달라지는 15개는 새로 초기화한다(v2 시간 측정 run의 로그). 방법마다 클래스 수(118, 96, 75)가 달라서 이 층의 크기도 다르다 |
| 크기 (v2, 클래스 118) | 학습 시 파라미터 33,048,566개, 110.4 GFLOPs. 평가 시 층을 합친 뒤 32,226,230개, 105.9 GFLOPs |

## 4. 학습 설정

- "1차"는 1차 실험에서 실제로 쓰인 값이다. "이번 실험에서"는 이 값을 쓴 이유나 결과다.

### 4-1. 직접 지정한 인자 (`model.train()`에 넘긴 값)

| 인자 | 값 | 1차 | 이번 실험에서 |
|---|---|---|---|
| model | `weights/rtdetr-l.pt` | 같음 | 3번 |
| data | `data/processed/exp2/<방법>/data.yaml` | `data/processed/yolo/data.yaml` | 2번 |
| epochs | 5, 10 | 100 | 데이터가 38~61배라 줄였다. 노트북 4번(시간 측정)에서 v2 1 epoch 학습에 14.8분이 걸려, 6개 run에 약 14.6시간으로 예측하고 정했다 |
| patience | 30 | 같음 | epoch가 10 이하라 조기 종료는 일어나지 않는다 |
| imgsz | 640 | 같음 | 976 × 1280 사진을 640 × 640으로 늘려 맞춘다 |
| batch | 16 | 같음 | GPU 메모리는 1번 |
| seed | 0 | 0, 1, 2 | 팀이 seed 0으로 통일했다. 방법과 epoch마다 한 번씩 학습했다 |
| deterministic | False | 같음 | RT-DETR의 `grid_sample`이 지원하지 않는다 |
| workers | 0 | 같음 | Windows 주피터에서 사진 읽기 오류를 피한다. 속도에만 영향 |
| cache | False | 같음(기본값) | 가이드대로 적는다 |
| device | 0 | 같음 | |
| optimizer | AdamW | auto → AdamW | 4-2 |
| lr0 | 0.000167 | auto가 고른 값과 같음 | 4-2 |
| momentum | 0.9 | auto가 정한 값과 같음 | AdamW의 beta1. 4-2 |
| warmup_bias_lr | 0.0 | auto가 정한 값과 같음 | 4-2 |
| warmup_epochs | 0.03 × epochs (0.15, 0.3) | 3 | 4-2 |
| close_mosaic | max(1, round(0.1 × epochs)) (5, 10 epoch 모두 1) | 10 | 4-2 |
| project / save_dir | `pill-object-detection` / `runs/exp2/<run 이름>` | `runs` / `rtdetr-l_e100_s{seed}_fix` | `project`는 W&B 프로젝트 이름이고, 결과 폴더는 `save_dir`다. 폴더 이름이 W&B 실행 이름이 된다 |

- 학습 전에 콜백 `require_wandb`를 붙인다(7번). 시간 측정 run은 `smoke` 태그를 붙이는 콜백과 시각을 기록하는 콜백도 붙인다.
- run 이름: `<방법>_rtdetr-l_e<epochs>` (예: `v2_rtdetr-l_e5`). 같은 이름의 run 폴더가 있으면 학습하지 않고 멈춘다(`exist_ok=False`, 기본값. 1차는 True).
- 시간 측정 run `smoke_v2_rtdetr-l_e1`은 1 epoch, `close_mosaic=0`(모자이크를 끄지 않음)이다. 본 학습과 비교하지 않는다.

### 4-2. 1차와 값이 다른 이유

- optimizer, lr0, momentum, warmup_bias_lr: `optimizer=auto`는 클래스 수와 학습량으로 optimizer와 학습률을 고른다(Ultralytics 소스).
    - 반복 수 = 올림(train 사진 수 ÷ 64) × epochs. 이번 run은 560~1,760이라 10,000 이하여서 AdamW가 된다. 10,000을 넘으면 MuSGD(학습률 0.01)가 된다.
    - 학습률 = 0.002 × 5 ÷ (4 + 클래스 수)라, auto였다면 v2 0.000082, v3 0.0001, v4 0.000127로 방법마다 달라진다(계산값). 방법끼리 비교할 수 없고 1차와도 달라진다.
    - 그래서 1차에서 auto가 실제로 고른 값 네 개를 직접 적었다. optimizer를 직접 적으면 auto가 하던 "warmup_bias_lr를 0으로 바꾸기"가 일어나지 않아서, 이 값도 함께 적는다.
- warmup_epochs, close_mosaic: 두 값은 epoch 수로 정해진다. 1차 값(3, 10)을 그대로 두면 5, 10 epoch에서는 전혀 다른 학습이 되어서, 값 대신 1차의 비율을 지켰다.
    - warmup: 1차는 100 epoch 중 3 epoch(3%)였다. 3을 그대로 두면 5 epoch run은 학습의 60%, 10 epoch run은 30%가 warmup이 된다. 그래서 `0.03 × epochs`로 3%를 지켰다.
    - 모자이크: 1차는 100 epoch 중 마지막 10 epoch(10%)를 껐다. 10을 그대로 두면 "epochs − 10"이 0 이하라 처음부터 끝까지 모자이크를 쓰지 않는다. 그래서 `max(1, round(0.1 × epochs))`로 정했다. 정수여야 해서 5, 10 epoch 모두 마지막 1 epoch만 끈다(5 epoch는 20%, 10 epoch는 10%).
    - 이 비율이 이번 데이터에서 가장 좋은 값인지는 확인하지 않았다.

### 4-3. run마다 정해진 값

| run | epoch당 batch | 반복 수 (auto 기준) | warmup batch | 모자이크 사용 / 끈 epoch | 가중치 고치는 횟수 (epoch당) |
|---|---:|---:|---:|---|---:|
| v2_rtdetr-l_e5 | 446 | 560 | 67 | 1~4 / 5 | 약 112 |
| v3_rtdetr-l_e5 | 561 | 705 | 84 | 1~4 / 5 | 약 140 |
| v4_rtdetr-l_e5 | 701 | 880 | 105 | 1~4 / 5 | 약 175 |
| v2_rtdetr-l_e10 | 446 | 1,120 | 134 | 1~9 / 10 | 약 112 |
| v3_rtdetr-l_e10 | 561 | 1,410 | 168 | 1~9 / 10 | 약 140 |
| v4_rtdetr-l_e10 | 701 | 1,760 | 210 | 1~9 / 10 | 약 175 |

- epoch당 batch = 올림(train 사진 수 ÷ 16). v2의 446은 학습 로그로 확인했고, v3, v4는 계산값이다.
- warmup batch = round(min(warmup_epochs, epochs − 1) × epoch당 batch)(Ultralytics 8.4.183 소스). 1차는 36 batch였다. 1 epoch인 시간 측정 run은 warmup이 없다.
- 가중치 고치는 횟수: batch 16이라 기울기를 4번 모아서(nbs 64 ÷ 16) 한 번 고친다. 1차와 같다.
- 학습률은 1차와 같은 직선으로 줄어든다: lr0 × ((1 − epoch ÷ epochs) × (1 − lrf) + lrf), lrf 0.01. run이 짧을수록 같은 epoch에서 학습률이 낮다.
- 실제로 만든 optimizer: `AdamW(lr=0.000167, momentum=0.9)`, 파라미터 그룹은 정규화 층 weight 143개(decay 0), 일반 weight 206개(decay 0.0005), bias 226개(decay 0)로 1차와 같다(v2 시간 측정 run의 로그).

### 4-4. Ultralytics 기본값으로 쓴 인자

- 학습 일정, 최적화, 증강의 기본값은 1차 조건 기록의 4-2와 같다. 그 표의 warmup_epochs와 close_mosaic만 이번에 직접 바꿨다(4-1, 4-2). 시간 측정 run의 학습 로그에서 확인했다.
    - 학습 일정과 최적화: lrf 0.01, cos_lr False, weight_decay 0.0005, nbs 64, amp True(AMP 확인 통과), pretrained True, freeze None, fraction 1.0, multi_scale 0.0, val True
    - 증강: hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, degrees 0.0, translate 0.1, scale 0.5, shear 0.0, perspective 0.0, flipud 0.0, fliplr 0.5, bgr 0.0, mosaic 1.0, mixup 0.0, cutmix 0.0, copy_paste 0.0
- 로그에는 있지만 이번 학습에 쓰이지 않는 인자(box, cls, dfl, warmup_momentum, dropout, rect, auto_augment, erasing, iou)도 1차 조건 기록의 4-4와 같다.

## 5. 평가 방법

| 항목 | 값 |
|---|---|
| 채점 함수 | 팀 공용 `pill_detection.evaluate.evaluate_val` (pycocotools) |
| 지표 | mAP@[0.75:0.95]. IoU 0.75, 0.80, 0.85, 0.90, 0.95의 AP를 클래스와 기준에 대해 평균한다. 대회 지표와 같다 |
| 정답 | `data/processed/exp2/<방법>/eval/val_gt.json`. val의 모든 객체가 원래 K-코드로 들어 있고, 세 방법이 같은 파일이다 |
| 채점 클래스 | 대회 클래스 74종만 |
| 버리는 예측 | 대회 클래스 밖으로 낸 예측(`others` 포함), 대회 밖 정답 박스와 IoU 0.5 이상 겹친 예측(클래스와 상관없이) |
| 점수 3개 | `val`: val 1,000장 전체. 방법과 epoch를 비교하는 기준 / `kaggle`: Kaggle 사진 47장. 1차와 비교할 때 / `test_only`: test 전용 18종 |
| 예측 조건 | best.pt, imgsz 640, 점수 하한 0.001(가이드 6단계), max_det 300(기본값) |
| 검출 수 제한 | pycocotools 기본값. 사진과 클래스마다 점수 상위 100개 |
| best.pt | Ultralytics가 epoch마다 잰 val mAP50-95가 가장 높은 epoch의 가중치. 모델의 모든 클래스를 평균하고 IoU 0.5~0.95를 보므로 대회 지표와 다르다 |
| 1차 모델 다시 채점 | 1차 best.pt(seed 0, 1, 2)를 같은 함수로 잰 `kaggle`은 0.9866, 0.9852, 0.9782다. 1차 노트북의 Ultralytics mAP75-95(0.9816, 0.9803, 0.9731)와는 계산 방식이 달라 숫자가 조금 다르다. 2차와는 이 `kaggle`로 비교한다 |

## 6. test 예측과 제출

| 항목 | 값 |
|---|---|
| 가중치 | 각 run의 best.pt |
| imgsz | 640 |
| 점수 하한 (conf) | 0.01 (가이드 8단계, 1차와 같음) |
| max_det | 300 (기본값) |
| 변환 | `predictions_from_ultralytics` → `write_submission`(팀 함수). 대회 클래스 74종 밖의 예측(v4의 `others`, v2와 v3의 AI Hub에만 있는 클래스)은 뺀다 |
| 박스 | 사진 경계로 자르지 않는다. 1차는 잘랐다. 2차 val 채점도 자르지 않아서 같은 조건으로 맞췄다. `[x, y, w, h]` 픽셀 소수 2자리, score 소수 4자리 |
| 줄 순서 | 사진 번호 순, 사진 안에서는 점수가 높은 순. `annotation_id`는 1부터 |
| 파일 | `submissions/exp2_<run 이름>.csv` (git이 무시하는 폴더) |
| 줄 수 | 90,354줄(v2_e10) ~ 176,573줄(v4_e10). 사진당 평균 107.3~209.7줄. 대회 밖 클래스가 많은 방법일수록 빠지는 예측이 많아 줄이 적다 |
| 제출 | Kaggle 웹에서 CSV를 올리고, 제출 설명에 run 이름을 적는다. 6개 run 모두 제출했다 |

## 7. 기록 (W&B)

| 항목 | 값 |
|---|---|
| 팀 / 프로젝트 | `codeit-prj1-team2` / `pill-object-detection` |
| 기록 방식 | Ultralytics에 내장된 W&B 기록(가이드 방식). `WANDB_ENTITY`로 팀을 정하고 Ultralytics 설정의 `wandb`를 켠다. 1차처럼 `wandb.init`을 직접 부르지 않는다 |
| 실행 이름 | run 폴더 이름: `v2_rtdetr-l_e5`, `v3_rtdetr-l_e5`, `v4_rtdetr-l_e5`, `v2_rtdetr-l_e10`, `v3_rtdetr-l_e10`, `v4_rtdetr-l_e10`. 시간 측정은 `smoke_v2_rtdetr-l_e1` |
| 태그 | `exp2`와 방법(`v2`, `v3`, `v4`). 시간 측정 run은 `smoke`도 붙는다 |
| 안전장치 | `require_wandb` 콜백이 첫 epoch 직전에 불린다. W&B 기록이 팀 프로젝트에 시작되지 않았으면 학습을 멈추고, 시작됐으면 태그를 붙인다 |
| Summary 지표 | 학습이 끝난 뒤 `score_run`이 `exp2/val`, `exp2/kaggle`, `exp2/test_only`를 붙인다 |
| run 기록 | [`rtdetr-exp2/runs.csv`](rtdetr-exp2/runs.csv) (6줄, 가이드 7단계 형식) |
| run 폴더 | `runs/exp2/<run 이름>/`를 공유 드라이브 `pill data/jw_runs/<run 이름>/`에 올렸다(best.pt 포함) |

## 8. 결과

| run | val | kaggle | test_only | Ultralytics best mAP50-95 (epoch) | Kaggle public | 학습 시간 (나눠서 학습한 횟수) |
|---|---:|---:|---:|---|---:|---|
| v2_rtdetr-l_e5 | 0.9800 | 0.9839 | 0.9783 | 0.9838 (5) | 0.60731 | 1.66시간 (1) |
| v3_rtdetr-l_e5 | 0.9785 | 0.9810 | 0.9722 | 0.9832 (5) | 0.60295 | 9.58시간 (2) |
| v4_rtdetr-l_e5 | 0.9858 | 0.9866 | 0.9812 | 0.9883 (5) | 0.61558 | 5.53시간 (3) |
| v2_rtdetr-l_e10 | 0.9809 | 0.9844 | 0.9783 | 0.9855 (10) | 0.61516 | 2.72시간 (2) |
| v3_rtdetr-l_e10 | 0.9882 | 0.9971 | 0.9861 | 0.9886 (10) | 0.60655 | 2.67시간 (1) |
| v4_rtdetr-l_e10 | 0.9902 | 0.9908 | 0.9875 | 0.9906 (8) | 0.61140 | 3.65시간 (1) |
| 1차 seed 0 (다시 채점) | 0.6662 | 0.9866 | 0.0000 | | 0.42152 | 1.07시간 |

- val, kaggle, test_only는 5번의 팀 채점 함수로 낸 mAP@[0.75:0.95]다. 1차 모델은 AI Hub 클래스를 배우지 않아서 1차와는 `kaggle`과 Kaggle public만 비교한다.
- val 기준으로 고른 run은 `v4_rtdetr-l_e10`이다(방법마다 5, 10 epoch 중 val이 높은 쪽을 고르면 세 방법 모두 10 epoch).
- Kaggle public이 가장 높은 run은 `v4_rtdetr-l_e5`(0.61558)로, 1차 seed 0보다 0.194 높다. 6개 run 모두 1차보다 0.181~0.194 높다.
- 6개 run의 Kaggle public 차이는 0.013 안이다. val 순위와 public 순위는 맨 아래(v3_e5)만 같았다.
- 같은 방법의 5 epoch와 10 epoch의 Kaggle public 차이(10 − 5): v2 +0.0079, v3 +0.0036(차이 없음), v4 −0.0042. 제출 전에 0.004 미만이면 차이 없음으로 정했다.
- 학습 시간 합계는 25.81시간이다. v3_e5와 v4_e5가 긴 것은 ■로 중단한 뒤 커널을 다시 시작하지 않고 이어서 학습한 4개 epoch(v3_e5의 3~5, v4_e5의 1)에 약 12시간이 들었기 때문이다(10번).

## 9. 추가 분석 요약 (노트북 9번, 10번)

val(0.98~0.99)과 Kaggle(0.60~0.62)의 차이(약 0.37)의 원인을 찾고, 재학습 없는 후처리로 Kaggle 점수를 올려 보았다.

| 노트북 | 조건 | 결과 |
|---|---|---|
| 9-2 ③ 팀원의 test 검수표와 짝짓기 | v4_e10, v4_e5의 점수 0.5 이상 test 답과 팀원의 검수표(`notebooks/eda/02_test_crops.csv`의 `final_category_id`, 3,229개)를 IoU 0.3 이상으로 짝지음 | 두 run 모두 test 알약을 모두 찾고, 99.8%(3,224개, 3,222개)를 검수표와 같은 약으로 답함 |
| 9-2 ④ 사진 크기와 좌표 기준 | test, 대회 train, val 사진 크기와 대회 라벨 JSON의 크기 | 모두 976 × 1280. 박스가 사진 밖으로 나간 것은 많아야 1.3픽셀 |
| 9-3 진단 제출 | `exp2_v4_rtdetr-l_e5.csv`에서 점수 0.5 이상 3,236줄만 남겨 제출 | 0.61545 (원래 0.61558, −0.00013). 낮은 점수 답은 Kaggle 점수에 거의 영향이 없다 |
| 9-4 val 박스 정밀도 | 6개 run, val 1,000장(점수 0.001 이상), 알약마다 점수가 가장 높은 답으로 판정 | IoU 0.75~0.90의 AP는 0.994~1.000이고, 점수를 잃는 곳은 IoU 0.95다. 박스 이탈과 못 찾음은 0개 |
| 팀원의 박스 크기 조절 제출 | `exp2_v2_rtdetr-l_e10.csv`의 박스를 ×1.02, ×0.98로 바꿔 제출 | 0.60500, 0.54908 (원래 0.61516). 둘 다 떨어짐 |
| 9-5 클래스 묶음 진단 제출 | `exp2_v4_rtdetr-l_e5.csv`에서 test 전용 18종의 줄 41,377개만 남겨 제출 | 0.15732. 계산하면 18종 평균 AP 0.647, 56종 0.606으로 두 묶음에서 고르게 잃는다 |
| 10-2 가장자리 박스 빼기 | 사진 가장자리 3픽셀 안에 닿는 박스를 지우고 val로 먼저 잼 | 6개 run 모두 val이 0.0071~0.0089 떨어짐(val 정답에 가장자리 알약 20개). 제출하지 않음 |
| 10-3 여러 run 합치기 (WBF) | `ensemble-boxes` 기본값(IoU 0.55), 조합 4가지 × 가장자리 빼기 유무 | 원래 `v4_rtdetr-l_e10`의 val(0.9902)보다 오른 조합이 없음. 제출하지 않음. 라이브러리는 실험 뒤 삭제 |

- val과 Kaggle의 차이의 원인은 모델 밖(test 정답이나 채점 쪽)에 있을 가능성이 크다고 봤다. test 정답을 볼 수 없어서 확인하지는 못했다. Kaggle 리더보드의 상위 점수도 0.6점대였다.
- 9-2 ③의 99.8%는 팀원의 검수표 기준이다. 검수표는 Kaggle 정답이 아니라 사람이 눈으로 정한 약 번호다.

## 10. 3차 실험에서 같은 조건을 유지하려면

- optimizer와 관련 값 네 개를 계속 직접 적는다: `optimizer="AdamW", lr0=0.000167, momentum=0.9, warmup_bias_lr=0.0`. auto로 두면 클래스 수에 따라 학습률이 바뀌고, 반복 수(올림(사진 수 ÷ 64) × epochs)가 10,000을 넘으면 MuSGD가 된다(4-2).
- epochs를 바꾸면 `warmup_epochs = 0.03 × epochs`, `close_mosaic = max(1, round(0.1 × epochs))`를 다시 계산한다. warmup 길이는 epoch당 batch 수(사진 수)에 비례해 늘어난다(4-3).
- 방법(데이터)을 바꾸면 클래스 수가 바뀌어 클래스 층 15개의 크기가 바뀐다. 학습률은 직접 적었으므로 바뀌지 않는다.
- val은 학습 폴더 커밋 4fdae56의 1,000장이다. val이 바뀌면 점수를 2차와 직접 비교할 수 없다.
- 제출 조건(imgsz 640, conf 0.01, max_det 300, 박스를 자르지 않는 팀 함수)을 같게 둔다. 1차와 비교할 때는 박스 자르기 차이가 조금 섞인다.
- seed 0으로 한 번씩만 학습했다. 1차 모델은 seed만 바꿔도 `kaggle`이 0.008, Kaggle public이 0.004 정도 움직였다. 이보다 작은 차이는 seed 차이일 수 있다.
- val로 run을 고르고, Kaggle 점수는 결과를 확인하는 데만 쓴다(팀 가이드). 이번에는 val 순위와 Kaggle 순위가 거의 맞지 않았다.
- 로컬 학습(Windows)
    - `workers=0`으로 둔다.
    - batch 16에서 GPU 메모리가 12.7GB로 전용 메모리를 조금 넘는다.
    - 학습 시간은 v2 1 epoch에 약 15분(batch 하나 약 2초)이었고, 10 epoch run 하나에 2.7~3.7시간이 걸렸다.
    - ■로 중단했으면 커널을 다시 시작한 뒤 이어서 학습한다. 같은 커널에서 이어서 학습하자 GPU 메모리가 22.5GB로 늘고 batch 하나가 18초 안팎으로 느려졌다.
- 재학습 없는 후처리(가장자리 박스 빼기, WBF)는 val을 올리지 못했다(9번). `ensemble-boxes`는 팀 라이브러리 목록에 없다.
