# RT-DETR 1차 실험(베이스라인) 조건 기록

- 기록 기준: [`rtdetr-exp1.ipynb`](rtdetr-exp1.ipynb)의 실행 출력, 세 seed의 학습 로그, Ultralytics 8.4.183 소스
- 학습일: 2026-10-29
- 전체 학습 인자는 run 폴더마다 `runs/rtdetr-l_e100_s{seed}_fix/args.yaml`에 저장되어 있다. 다만 `optimizer=auto`가 학습을 시작할 때 바꾼 값(4-3)은 `args.yaml`에 반영되지 않는다. optimizer, 학습률, warmup_bias_lr의 실제 값은 best.pt 안의 `train_args`에 남지만, momentum은 `train_args`에도 바뀌기 전 값(0.937)으로 남는다. 2차 실험을 설정한 뒤 이 문서의 4번과 비교하면 달라진 값을 찾을 수 있다.

## 목차

| 섹션 | 내용 |
|---|---|
| 1. 실행 환경 | 운영체제, 라이브러리 버전, GPU |
| 2. 데이터 | 원본, 라벨 수정, 분할 규칙, YOLO 변환 |
| 3. 모델 | 시작 가중치, 학습 방식, 크기 |
| 4. 학습 설정 | 직접 지정한 인자, 기본값, 자동으로 정해진 값, 쓰이지 않은 인자 |
| 5. 평가 방법 | best.pt 기준, val 평가, 대회 지표 근사 |
| 6. test 예측과 제출 | 예측 조건, 제출 파일 |
| 7. 기록 (W&B) | 팀 프로젝트, 실행 이름, 직접 넣은 지표 |
| 8. 결과 | seed별 val 점수와 Kaggle 점수 |
| 9. 2차 실험에서 같은 조건을 유지하려면 | 고정해야 할 값과 주의점 |

## 1. 실행 환경

| 항목 | 값 |
|---|---|
| 운영체제 | Windows, VS Code 주피터, uv 가상환경 |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |
| Ultralytics | 8.4.183 |
| W&B (wandb) | 0.31.0 |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12GB |
| CPU / RAM | 논리 프로세서 12개 / 31.6GB |
| albumentations | 설치되어 있으면 학습 로그에 적용 목록이 찍히는데 그 줄이 없다. 설치되지 않아 이 라이브러리의 추가 증강(흐림, 회색조 등)은 적용되지 않은 것으로 보인다 |

## 2. 데이터

| 항목 | 값 |
|---|---|
| 원본 | 대회 `train_images` 232장, `train_annotations`(JSON) bbox 763개, 클래스 56종 |
| 라벨 수정 | `data/label_fix/label_fixes.csv`(팀 공용, 21줄). add 8줄 추가, modify 13줄 중 bbox가 바뀐 9줄 수정, bbox가 같은 4줄(area만 수정)은 건너뜀. 적용 후 bbox 771개 |
| 분할 | `data/processed/split.csv`(1차 EDA에서 고정, 분할 seed 410, 조합 단위). train 185장(bbox 612개, 56종) / val 47장(bbox 159개, 30종) |
| 클래스 번호 | category_id를 작은 순으로 0~55에 대응. `data.yaml`의 클래스 이름은 약 번호(category_id) 글자 |
| YOLO 변환 | 사진 1장당 txt 1개, 한 줄에 `클래스 중심x 중심y 너비 높이`(이미지 크기로 나눈 0~1 값, 소수 6자리). 이미지 밖으로 나간 bbox는 경계에서 자르고, 완전히 밖이면 버림(버린 bbox 0개). 폴더는 `data/processed/yolo/{images,labels}/{train,val}` |

분할 규칙(1차 EDA 8-2)

- 조합 단위로 나눈다. 조합이 1개뿐인 약이 든 조합은 train에 넣는다.
- 라벨 문제가 있는 조합은 val에서 뺀다.
- val에 넣는 약은 train에도 최소 한 조합을 남긴다.
- val은 사진 기준 약 20%로 한다.
- 시드 0~499 중, val 박스에서 3351(일양하이트린정)이 차지하는 비율이 전체 박스 비율(원본 763개 기준 153/763 = 0.201)과 ±0.02 이내인 시드 가운데 val 약 종류가 가장 많은 시드를 골랐다(같으면 비율 차이가 작은 시드). 결과는 410이다.

## 3. 모델

| 항목 | 값 |
|---|---|
| 모델 | RT-DETR-L (Ultralytics `RTDETR`) |
| 시작 가중치 | `rtdetr-l.pt`(COCO 80종 사전학습, Ultralytics 배포), 레포의 `weights/rtdetr-l.pt` |
| 학습 방식 | 풀 파인튜닝. 고정한 층 없음(`freeze=None`). 사전학습 가중치 941개 중 926개를 가져왔다. 나머지 15개는 클래스 수에 따라 크기가 달라지는 층이라 새로 초기화했다(인코더의 클래스 점수 층 2개, 디코더 6단의 클래스 점수 층 12개, 디노이징용 클래스 임베딩 1개) |
| 크기 | 학습 시 파라미터 약 3,292만 개, 110.1 GFLOPs (평가 시 층을 합친 뒤 약 3,210만 개, 105.6 GFLOPs) |

## 4. 학습 설정

- "값"은 이번 실험에서 실제로 쓰인 값이다. "뜻"은 그 인자가 무엇을 정하는지, "이번 실험에서"는 이 값을 쓴 이유나 결과다.

### 4-1. 직접 지정한 인자 (`model.train()`에 넘긴 값)

| 인자 | 값 | 뜻 | 이번 실험에서 |
|---|---|---|---|
| epochs | 100 | train 사진 전체를 한 번씩 다 보는 것이 1 epoch이다. 이것을 몇 번 반복할지 정한다 | 팀 설정 |
| patience | 30 | val 성능(mAP50-95)이 이 epoch 수 동안 한 번도 최고값을 넘지 못하면 학습을 일찍 멈춘다(조기 종료) | 팀 설정. 세 seed 모두 100 epoch을 다 돌았다 |
| imgsz | 640 | 모델에 넣는 사진 크기. RT-DETR는 가로세로 비율을 유지하지 않고 사진을 640×640으로 늘이거나 줄여 맞춘다. 학습, val, 예측 모두 같다 | 팀 설정. 원본 976×1280이 640×640이 된다(976 쪽은 0.66배, 1280 쪽은 0.5배) |
| batch | 16 | 반복 한 번에 GPU에 함께 올려 손실을 계산하는 사진 수 | 팀 설정. `GPU_mem` 12.8G로 전용 GPU 메모리(12GB)를 넘어 시스템 메모리를 같이 썼다. 속도에만 영향 |
| seed | 0, 1, 2 | 난수의 시작값. 새로 만든 층의 초깃값, 사진 순서, 증강을 정한다 | seed마다 run 하나 |
| deterministic | False | True면 같은 seed에서 같은 결과가 나오도록 결정적인 연산만 쓰게 강제한다 | RT-DETR가 쓰는 `grid_sample`이 지원하지 않아 False. 같은 seed로 다시 돌려도 결과가 똑같지는 않다 |
| workers | 0 | 사진을 읽고 증강하는 보조 프로세스 수. 0이면 학습 프로세스가 직접 한다 | Windows 주피터 오류 회피. 속도에만 영향 |
| device | 0 | 학습할 장치. 0은 첫 번째 GPU, `"cpu"`는 CPU | |
| optimizer | auto | 기울기를 보고 가중치를 고치는 방법(SGD, AdamW 등). auto면 Ultralytics가 반복 수를 보고 고른다 | 실제로 정해진 값은 4-3 |
| project / name | `runs` / `rtdetr-l_e100_s{seed}_fix` | 결과를 저장할 폴더. `project` 폴더 아래에 `name` 폴더가 생긴다 | |
| exist_ok | True | 같은 이름의 폴더가 있으면 새 번호를 붙이지 않고 그 폴더에 덮어쓴다 | |

### 4-2. Ultralytics 기본값으로 쓴 인자 중 결과에 영향을 주는 것

학습 일정과 최적화

| 인자 | 값 | 뜻 | 이번 실험에서 |
|---|---|---|---|
| lrf | 0.01 | 마지막 학습률의 비율. 학습이 끝날 때 학습률은 처음 학습률 × lrf에 가까워진다 | 마지막 epoch의 학습률은 약 0.0000033 |
| cos_lr | False | 학습률을 줄이는 모양. True면 코사인 곡선, False면 직선 | 직선: 처음 학습률 × ((1 − epoch/epochs) × (1 − lrf) + lrf) |
| warmup_epochs | 3.0 | 학습 초반에 학습률을 0에서 처음 학습률까지 서서히 올리는 구간의 길이. 새로 초기화한 층이 있는데 처음부터 큰 학습률로 고치면 학습이 흔들릴 수 있어서다 | 3 epoch = 36회 반복 |
| weight_decay | 0.0005 | 가중치가 지나치게 커지지 않게 매번 일정 비율씩 줄이는 정도. 과적합을 줄인다. AdamW에서는 기울기와 따로 적용된다 | batch × 기울기 누적 횟수 / nbs로 보정하는데, batch 16이면 그대로 0.0005 |
| nbs | 64 | 기준 batch 크기. 실제 batch가 이보다 작으면 여러 batch의 기울기를 모아서(기울기 누적) 64장마다 가중치를 한 번 고친다 | batch 16이라 4번 모아서 한 번 |
| amp | True | 자동 혼합 정밀도. 일부 계산을 16비트로 해서 메모리를 덜 쓰고 빨라진다 | AMP 확인 통과, 손실 NaN 없음 |
| pretrained | True | 시작 가중치 파일의 가중치를 불러올지 | `rtdetr-l.pt`에서 시작 |
| freeze | None | 학습하지 않고 고정할 앞쪽 층 수 | 고정 없음(풀 파인튜닝) |
| fraction | 1.0 | train 데이터 중 학습에 쓸 비율 | 185장 전체 |
| cache | False | 사진을 메모리(RAM)나 디스크에 미리 올려 둘지 | 매번 파일에서 읽었다 |
| multi_scale | 0.0 | 학습 중 입력 크기를 imgsz 대비 얼마 범위에서 무작위로 바꿀지 | 0이라 항상 640 |
| val | True | 매 epoch 끝에 val로 평가할지 | 매 epoch 평가. best.pt와 patience가 이 결과로 정해진다 |

증강 (학습 사진에만 무작위로 적용된다. RT-DETR 데이터셋도 YOLO 데이터셋과 같은 증강 코드를 쓴다)

| 인자 | 값 | 뜻 |
|---|---|---|
| hsv_h | 0.015 | 색조(Hue)를 색상환의 ±1.5%(약 ±5도) 범위에서 무작위로 돌린다 |
| hsv_s | 0.7 | 채도를 ±70% 범위에서 바꾼다 |
| hsv_v | 0.4 | 밝기를 ±40% 범위에서 바꾼다 |
| degrees | 0.0 | 회전 각도 범위(±도). 0이면 회전하지 않는다 |
| translate | 0.1 | 사진을 가로, 세로로 사진 크기의 ±10% 범위에서 옮긴다 |
| scale | 0.5 | 크기를 ±50% 범위(0.5~1.5배)에서 바꾼다 |
| shear | 0.0 | 사진을 비스듬히 기울이는 각도 범위(±도) |
| perspective | 0.0 | 원근 변형 정도 |
| flipud | 0.0 | 상하 뒤집기 확률 |
| fliplr | 0.5 | 좌우 뒤집기 확률(50%) |
| bgr | 0.0 | 색 채널 순서(RGB↔BGR)를 바꾸는 확률 |
| mosaic | 1.0 | 사진 4장을 2×2로 이어 붙여 한 장으로 만드는 증강의 확률(1.0이면 항상) |
| close_mosaic | 10 | 마지막 N epoch은 모자이크를 끈다. 실제 사진과 비슷한 입력으로 학습을 마무리하기 위해서다. 이번에는 91~100 epoch(로그의 "Closing dataloader mosaic") |
| mixup | 0.0 | 두 사진을 반투명하게 겹치는 증강의 확률 |
| cutmix | 0.0 | 한 사진의 일부를 다른 사진의 조각으로 바꾸는 증강의 확률 |
| copy_paste | 0.0 | 물체를 오려 다른 사진에 붙이는 증강. 분할(segment) 라벨이 있어야 쓸 수 있다 |

### 4-3. 자동으로 정해진 값

`optimizer=auto`라서 아래 값은 Ultralytics가 학습을 시작할 때 정했다. 그래서 로그 앞부분의 설정 줄과 `args.yaml`에는 바뀌기 전 값(`optimizer=auto`, `lr0=0.01`, `momentum=0.937`, `warmup_bias_lr=0.1`)이 남아 있다. best.pt 안의 `train_args`에는 optimizer, 학습률, warmup_bias_lr의 실제 값이 남지만, momentum은 그 안에서도 0.937로 남는다. momentum은 Ultralytics가 optimizer를 만들 때만 0.9로 쓰고 설정에는 다시 적지 않기 때문이다.

| 항목 | 실제 값 | 뜻 | 정해진 방식 |
|---|---|---|---|
| optimizer | AdamW | 가중치마다 기울기의 평균과 크기를 따로 기억해서 고치는 폭을 조절하는 방법. weight decay를 기울기와 따로 적용한다 | 반복 수 = 올림(185 / max(16, 64)) × 100 = 300. 10,000 이하라 AdamW (넘으면 MuSGD) |
| 학습률(lr0) | 0.000167 | 가중치를 한 번 고칠 때 얼마나 크게 움직일지 | 0.002 × 5 / (4 + 클래스 수 56). 설정의 0.01은 쓰이지 않았다 |
| momentum | 0.9 | 이전 기울기를 얼마나 이어서 반영할지(관성). AdamW에서는 beta1 | auto가 0.9로 정했다. 설정의 0.937은 쓰이지 않았다 |
| warmup_bias_lr | 0.0 | warmup 동안 bias(편향) 학습률의 시작값. 다른 가중치는 0에서 올라가고, bias는 이 값에서 처음 학습률로 내려온다 | auto가 Adam 계열을 고르면 0.0으로 바꾼다. 로그의 0.1은 쓰이지 않았다 |
| 파라미터 그룹 | 3개 | weight decay를 적용할 가중치와 적용하지 않을 가중치를 나눠 관리한다 | 일반 weight(decay 0.0005) 206개, 정규화 층 weight(decay 0) 143개, bias(decay 0) 226개 |
| 기울기 누적 횟수 | 4 | 몇 batch의 기울기를 모아서 가중치를 한 번 고칠지 | nbs 64 / batch 16. warmup 동안은 1에서 4로 서서히 늘어난다 |
| warmup 반복 수 | 36 | warmup이 이어지는 반복 횟수 | 3 epoch × epoch당 batch 12개 |

### 4-4. 로그에는 있지만 이번 학습에 쓰이지 않은 인자

- `box` 7.5, `cls` 0.5, `dfl` 1.5: YOLO 손실의 가중치다. RT-DETR는 자체 손실 가중치(클래스 1, L1 bbox 5, GIoU 2)를 쓴다.
- `warmup_momentum` 0.8: warmup 동안 momentum을 이 값에서 올리는 설정인데, momentum 항목이 있는 optimizer(SGD 계열)에만 적용된다. AdamW에는 적용되지 않았다.
- `dropout` 0.0: 분류 모델 전용이다.
- `rect` False: RT-DETR 데이터셋은 이 값과 관계없이 항상 False로 만든다.
- `auto_augment`, `erasing`: 분류 모델용 증강이다.
- `iou` 0.7: NMS 기준값이다. RT-DETR는 NMS를 쓰지 않는다.

## 5. 평가 방법

- best.pt: val mAP50-95가 가장 높은 epoch의 가중치(Ultralytics 8.4.183의 fitness는 mAP50-95만 본다).
- 학습이 끝나면 best.pt로 val 47장을 다시 잰다. val의 신뢰도 기준은 0.001, 사진당 최대 예측 300개.
- 대회 지표 근사 mAP75-95: Ultralytics val 결과의 `metrics.box.all_ap[:, 5:].mean()`. IoU 0.75~0.95 다섯 단계 AP의 평균이고, val 정답에 있는 30종으로 평균한다.
- precision, recall: 전체 F1이 가장 높은 신뢰도 기준값 하나에서 잰 값이다.

## 6. test 예측과 제출

| 항목 | 값 |
|---|---|
| 가중치 | 각 seed의 best.pt |
| imgsz | 640 (학습과 같이 640×640으로 늘려 맞춘 뒤 예측하고, 박스는 원본 크기로 되돌린다) |
| conf | 0.01 |
| max_det | 300 (기본값) |
| 박스 | 사진 경계로 자른 뒤 `[x, y, w, h]` 픽셀, 소수 2자리. score는 소수 4자리 |
| 약 번호 | 모델의 클래스 이름(약 번호)으로 되돌림 |
| image_id / annotation_id | test 파일 이름의 숫자 / 1부터 차례로 |
| 파일 | `runs/rtdetr-l_e100_s{seed}_fix/submission_rtdetr-l_e100_s{seed}_fix.csv` |
| 제출 | Kaggle 웹에서 seed별로 업로드 (대회 `ai15-level1-project`) |

## 7. 기록 (W&B)

| 항목 | 값 |
|---|---|
| 팀 / 프로젝트 | `codeit-prj1-team2` / `pill-object-detection` |
| 실행 이름 (ID) | `jw_rtdetr-l_e100_s0_fix`(vfnm62ub), `jw_rtdetr-l_e100_s1_fix`(en7jmas5), `jw_rtdetr-l_e100_s2_fix`(b8pk5bwo) |
| 태그 | `rtdetr`, `label-fixed`, `baseline` |
| config | model, split_seed(410), data 설명, 학습 인자(경로 제외) |
| 직접 넣은 지표 | `val/map_75_95`(epoch마다), `best_val/map_75_95`(요약값, best.pt 값) |
| 기록 방식 | 학습 전에 `wandb.init`을 직접 부르고, Ultralytics의 wandb 설정은 학습하는 동안만 켠다 |
| 주의 | seed 1, 2는 학습 중 W&B 서버 오류로 요약값이 각각 89, 38단계에서 멈춰 있다. 결과는 로컬 run 폴더 기준으로 본다 |

## 8. 결과

| seed | val mAP50 | val mAP50-95 | val mAP75-95 | Kaggle (public) | 학습 시간 |
|---|---|---|---|---|---|
| 0 | 0.995 | 0.987 | 0.980 | 0.4215 | 1.07시간 |
| 1 | 0.993 | 0.985 | 0.978 | 0.4176 | 1.55시간 |
| 2 | 0.980 | 0.976 | 0.973 | 0.4201 | 2.76시간 |
| 평균 | 0.989 | 0.983 | 0.977 | 0.4198 | |

## 9. 2차 실험에서 같은 조건을 유지하려면

- optimizer와 관련 값을 직접 적는다: `optimizer="AdamW", lr0=0.000167, momentum=0.9, warmup_bias_lr=0.0`. `auto`로 두면 데이터 수, epoch, 클래스 수에 따라 optimizer와 학습률이 바뀐다. 반복 수가 10,000을 넘으면 MuSGD(학습률 0.01)가 되고, 학습률은 클래스 수로 계산하기 때문이다. 또 optimizer를 직접 적으면 auto가 하던 "warmup_bias_lr를 0으로 바꾸기"도 일어나지 않아서, 이 값도 같이 적어야 1차와 같아진다.
- epochs를 바꾸면 학습률이 줄어드는 속도와, 모자이크를 끄는 마지막 10 epoch이 전체에서 차지하는 비율이 같이 바뀐다. warmup은 3 epoch 그대로라 반복 수는 데이터 크기에 비례해 늘어난다.
- batch를 바꾸면 기울기 누적 횟수가 바뀌어 64장 단위로 가중치를 고치는 것은 같다. weight_decay 보정값도 같다. 다만 한 번에 GPU에 올리는 사진 수가 달라진다.
- `deterministic=False`라서 같은 설정, 같은 seed여도 결과가 조금 다르다. seed 0, 1, 2의 평균으로 비교한다.
- 클래스가 늘면 클래스 번호(0~N−1)가 다시 매겨진다. 제출할 때는 모델의 클래스 이름(약 번호)으로 되돌리므로 제출 형식은 그대로 쓸 수 있다.
- val 데이터가 바뀌면 val 점수는 1차와 직접 비교할 수 없다. 실험끼리는 Kaggle 점수로 비교한다.
- 제출 조건(imgsz 640, conf 0.01, max_det 300, 박스 자르기)도 같게 둔다.
