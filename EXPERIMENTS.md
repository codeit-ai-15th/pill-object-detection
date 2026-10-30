# 2차 실험 가이드
학습을 돌리는 팀원이 그대로 따라 하면 되도록 가이드를 작성했습니다. 자기 환경에 맞는 절([로컬](#로컬에서-학습) 또는 [Colab](#colab에서-학습))로 이동하셔서 1번부터 따라하시면 됩니다.

- 단계마다 **After**에 그 단계를 끝낸 뒤의 예상 폴더 상태와 화면 출력이 있습니다. 내 화면이 같으면 다음 단계로 넘어가시면 됩니다.
- After 폴더 그림에서 `+`로 시작하는 줄(초록)은 그 단계에서 새로 생긴 것이고, 나머지는 이미 있던 것입니다.
- `<...>`로 표시된 부분은 각자 상황에 맞게 바꿔서 입력합니다.
- 2차 실험은 이미지 선택 방법 4가지(v1~v4)를 같은 val로 비교합니다. 목록 CSV의 컬럼 설명은 [data/datasets/README.md](data/datasets/README.md)를 확인해주세요.

## 시작 전에
- [ ] PR #40이 main에 머지되었다. (목록 CSV, 학습 폴더 스크립트, 채점 코드)
- [ ] (Colab에서 학습하는 사람만) 데이터 담당이 Drive 공유 폴더에 zip 4개를 올렸다: `kaggle.zip`, `aihub_annotations.zip`, `aihub_images.zip`, `test_images.zip`
- [ ] 내가 맡은 **방법(v1~v4)** 과 **모델**이 정해졌다.
- [ ] 팀 공통 학습 설정값을 받았다. 방법끼리 비교하려면 데이터 말고는 모두 같아야 합니다.
    - `optimizer`, `lr0`를 직접 적습니다. `auto`로 두면 학습률이 클래스 수로 정해지는데, 클래스 수가 방법마다 달라서(118, 118, 96, 75) 비교가 깨집니다.
    - run 이름은 `<방법>_<모델>`로 작성해주세요. (예: `v2_yolo26s`)

| 방법 | 뜻 | 모델 클래스 | train 장수 | train 객체 수 |
| --- | --- | ---: | ---: | ---: |
| v1 | 사진 전부 | 118 | 11,206 | 42,800 |
| v2 | 클래스별 200개를 목표로 고른 사진 | 118 | 7,129 | 26,734 |
| v3 | 대회 클래스가 든 사진만 | 96 | 8,974 | 33,872 |
| v4 | v1과 같은 사진, 대회 클래스 밖 알약은 `others` 한 클래스 | 75 | 11,206 | 42,800 |

- val은 네 방법 모두 같은 이미지로 구성되어있으며, 1,000장(객체 3,790개)입니다.

---

## 로컬에서 학습
내 PC(NVIDIA GPU 또는 Apple Silicon)에서 학습하는 경우입니다. 명령은 모두 **레포 최상위 폴더**에서 실행해주세요.
- `uv run scripts/prepare_train_data.py`로 만든 `data/raw/`, `data/processed/`, `data/external/`이 이미 있다고 가정합니다.
- 로컬에 데이터가 없으면 3단계 끝의 "로컬에 데이터가 없을 때: zip으로 받기"를 따라해주세요.

| 언제 | 단계 |
| --- | --- |
| 처음 한 번 | 1. 레포 최신으로 → 2. 데이터 확인 → 3. "다름"일 때만 사본 다시 만들기 |
| 실험마다 | 4. 학습 폴더 만들기 → 5. 학습 → 6. val 채점 → 7. 기록 |
| val로 고른 run만 | 8. 제출 |

### 1. 레포를 최신으로 맞추기 (처음 한 번, main이 갱신되면 다시 해야합니다)
```bash
git switch main
git pull origin main
uv sync
```
- 실험 노트북을 커밋할 거라면 실험 브랜치를 만듭니다. 이름 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md#협업-방식)를 따릅니다.
    ```bash
    git switch -c exp/<모델-핵심변경>
    ```

**After**: `data/datasets/`에 파일 3개가 생깁니다. 이미지는 아직 없습니다.
```diff
  pill-object-detection/
  ├── data/
+ │   ├── datasets/
+ │   │   ├── images.csv        # 이미지 목록 12,206줄
+ │   │   ├── classes.csv       # 클래스 표 119줄
+ │   │   └── README.md         # 두 CSV 설명
  │   ├── raw/                  # 비어 있음
  │   └── processed/            # 비어 있음
  ├── scripts/
+ │   └── build_dataset.py
  └── src/pill_detection/
+     └── evaluate.py
```

### 2. 내 데이터가 팀과 같은지 확인 (처음 한 번)
```bash
uv run scripts/make_dataset_lists.py --check
```
- 파일은 쓰지 않고, 내 데이터로 데이터 목록을 다시 계산해서 커밋된 `images.csv`, `classes.csv`와 비교합니다. 몇 초 걸릴 수 있습니다.
- 입력이 같으면 결과가 항상 같습니다. 출력 마지막 줄이 "같음"이면 내 데이터가 팀과 같은 것입니다. 3단계를 건너뛰고 4단계로 가시면 됩니다.
- "다름"이면 내 사본(`data/processed/`)이 옛 정제 목록으로 만든 것입니다. 3단계로 진행해주세요.
    - 이 상태에서 `--check` 없이 돌리거나 CSV를 커밋하면 팀의 train/val 데이터 구성이 바뀝니다.

**After**: 폴더는 바뀌지 않습니다. 아래 폴더가 있고, 화면의 마지막 두 줄이 모두 "같음"이면 됩니다.
```text
pill-object-detection/data/
├── raw/
│   ├── train_images/             # PNG 232장       Kaggle 사진 (4단계가 읽음)
│   └── test_images/              # PNG 842장       test 사진 (8단계가 읽음)
├── processed/
│   ├── train_annotations/        # JSON 771개      정제한 Kaggle 라벨 (4단계가 읽음)
│   └── ai_hub/
│       ├── train_annotations/    # JSON 45,819개   정제한 AI Hub 라벨 (4단계가 읽음)
│       └── train_images/         # PNG 11,974장    AI Hub 사진 (4단계가 읽음)
└── external/aihub/raw/           # AI Hub 원본     3단계에서 사본을 다시 만들 때만 읽음
```
```text
[1] 출처별 이미지 수 (조합 수)
집합          Kaggle          AI Hub      합계
--------  --------  --------------  ------
2차 val     47 (25)       953 (320)   1,000
train v1  185 (89)  11,021 (3,677)  11,206
...                                         (보고표가 이어짐)

data/datasets/images.csv: 같음
data/datasets/classes.csv: 같음
```

### 3. "다름"일 때만: 사본 다시 만들기
```bash
uv run scripts/prepare_train_data.py
uv run scripts/make_dataset_lists.py --check
```
- 1단계에서 받은 최신 정제 목록(`data/label_fix/label_fixes.csv`)대로 사본을 다시 만듭니다. 몇 분 걸립니다.
- `data/external/aihub/raw/`가 다 풀려 있으면 AI Hub에서 받지 않으므로 API 키가 없어도 됩니다. 첫 줄에 "받을 것 0개"가 나와야 합니다.
- 사본 폴더(`data/processed/train_annotations/`, `data/processed/ai_hub/`)만 지우고 다시 만듭니다. 원본(`data/raw/`, `data/external/`)은 읽기만 합니다.
- 다시 돌린 `--check`도 "다름"이면 데이터 담당에게 알립니다.
- `git status`에 새 `data/label_fix/apply_label_fixes_v*.csv`가 보이면 내 정제 기록이 팀과 다르다는 뜻입니다. 커밋하지 말고 데이터 담당에게 알립니다.

**After**: 화면 출력
```text
[AI Hub] zip 16개 중 다 푼 것 16개, 받을 것 0개

[정제] 정제 목록대로 Kaggle 사본과 AI Hub 사본을 만듭니다. (몇 분 걸립니다)
Kaggle: JSON 763개 → 771개 (data/processed/train_annotations)
AI Hub: 이미지 11,974장, JSON 45,819개, ... (data/processed/ai_hub). ...
변경 기록: data/label_fix/apply_label_fixes_v2.csv ...

...
data/datasets/images.csv: 같음
data/datasets/classes.csv: 같음
```
**After**: 폴더 (`+` 줄은 지우고 다시 만든 사본)
```diff
  pill-object-detection/data/
  ├── raw/                          # 그대로
  ├── external/                     # 그대로
  └── processed/
+     ├── train_annotations/        # JSON 771개
+     └── ai_hub/
+         ├── train_annotations/    # JSON 45,819개
+         └── train_images/         # PNG 11,974장 (원본에 하드링크)
```

<details>
<summary>로컬에 데이터가 없을 때: zip으로 받기</summary>

`prepare_train_data.py`를 처음부터 돌리는 대신, 데이터 담당이 Drive에 올린 zip 4개를 받아서 풀어도 됩니다. 디스크는 약 50GB가 필요합니다. (zip을 지운 뒤에는 약 24GB)
- zip 4개(합계 약 23.5GB)를 **레포 밖** 폴더에 다운로드 받아주세요. 아래 예시는 `다운로드/pill_drive`입니다.
- 레포 최상위에서 zip을 풉니다. zip 안 경로가 `data/...`로 시작해서 자동으로 제자리에 들어갑니다. macOS, Linux, Windows PowerShell 모두 같은 명령입니다.
    ```bash
    uv run python -m zipfile -e $HOME/Downloads/pill_drive/kaggle.zip .
    uv run python -m zipfile -e $HOME/Downloads/pill_drive/aihub_annotations.zip .
    uv run python -m zipfile -e $HOME/Downloads/pill_drive/aihub_images.zip .
    uv run python -m zipfile -e $HOME/Downloads/pill_drive/test_images.zip .
    ```
- 다 푼 뒤 2단계로 돌아가 "같음"을 확인합니다. "같음"이면 zip 파일은 지워도 됩니다.
- zip에는 `data/external/`이 들어 있지 않습니다. 그래서 이 방법으로 받은 사람은 3단계를 할 수 없고, "다름"이면 zip을 다시 풉니다.

```diff
  pill-object-detection/data/
  ├── raw/
+ │   ├── train_images/             # PNG 232장      ← kaggle.zip
+ │   └── test_images/              # PNG 842장      ← test_images.zip
  └── processed/
+     ├── train_annotations/        # JSON 771개     ← kaggle.zip
+     └── ai_hub/
+         ├── train_annotations/    # JSON 45,819개  ← aihub_annotations.zip
+         └── train_images/         # PNG 11,974장   ← aihub_images.zip
```
</details>

### 4. 방법별 학습 폴더 만들기 (방법마다 한 번)

> 이 문서에서는 v2를 예시로 들었습니다. v1, v3, v4를 담당해도 마찬가지로 돌리면 됩니다.

```bash
uv run scripts/build_dataset.py --variant v2 --out data/processed/exp2/v2
```
- `--variant`에 맡은 방법(v1~v4)을, `--out`에 `data/processed/exp2/<방법>`을 줍니다.
- 이미지는 복사하지 않고 원본에 하드링크를 겁니다. 그래서 디스크를 수십 MB만 씁니다.
- 특정 방법으로 이 스크립트를 실행했으면, 다음에 또 그 방법으로 실험을 돌릴 때 그 폴더를 다시 쓰게 됩니다. 다시 만들려면 폴더를 지우고 이 스크립트를 다시 실행하면 됩니다.
- 다른 옵션은 [scripts/build_dataset.py](scripts/build_dataset.py) 맨 위 docstring을 확인해주세요.

**After**: 화면 출력 (v2)
```text
[v2] 학습 데이터를 읽고 data/processed/exp2/v2에 학습 폴더를 만듭니다.
클래스 118개, 이미지 train 7,129장 / val 1,000장, 객체 train 26,734개 / val 3,790개
이미지: {'hardlink': 8129}
YOLO 학습: data=<레포>/data/processed/exp2/v2/data.yaml
COCO 학습: root=<레포>/data/processed/exp2/v2/images/train, annFile=<레포>/data/processed/exp2/v2/coco/train.json (val도 같은 모양)
채점 정답: <레포>/data/processed/exp2/v2/eval/val_gt.json
```
- `'copy'`가 나오면 하드링크가 안 되는 다른 디스크입니다. 이미지가 복사되어 디스크를 최대 21GB 더 쓰므로, `--out`을 `data`와 같은 디스크로 줍니다.

**After**: 폴더
```diff
  data/processed/
+ └── exp2/v2/
+     ├── images/train/         # 7,129장 (원본에 하드링크)
+     ├── images/val/           # 1,000장
+     ├── labels/train/         # 7,129개 YOLO txt
+     ├── labels/val/           # 1,000개
+     ├── data.yaml             # nc 118, names {0: '250', ..., 117: '53384'}
+     ├── coco/                 # train.json, val.json (Faster R-CNN용)
+     ├── eval/val_gt.json      # 채점 정답 (네 방법 모두 같음)
+     ├── classes.csv           # 제출할 때 대회 클래스 74개를 읽음
+     └── dataset_info.json     # 개수, 목록 CSV 해시, git commit
```

### 5. 학습 (실험마다)
YOLO, RT-DETR: 노트북에서 실행.
```python
import os
from pathlib import Path
from ultralytics import YOLO          # RT-DETR: from ultralytics import RTDETR

# 노트북이 notebooks/ 아래에 있어도 레포 최상위로 옮긴다. 아래 경로는 모두 레포 최상위 기준
os.chdir(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file()))
DS = Path("data/processed/exp2/v2")

model = YOLO("yolo26s.pt")            # 맡은 모델. RT-DETR: RTDETR("rtdetr-l.pt")
model.train(
    data=str(DS / "data.yaml"),
    project="runs/exp2", name="v2_yolo26s",
    optimizer=..., lr0=..., epochs=..., imgsz=..., batch=..., seed=...,   # 팀 공통값
    cache=False,
    device=0,                         # NVIDIA GPU. Apple Silicon은 "mps"
)
```
- 학습 중에 Ultralytics가 출력하는 mAP50-95로는 방법끼리 비교하지 않습니다. 대회 클래스 밖 클래스까지 평균하고, IoU 기준도 대회(0.75~0.95)와 다릅니다. 비교는 6단계의 점수 3개로 합니다.
- W&B는 `uv run yolo settings wandb=True`일 때만 기록됩니다. 켜 두면 W&B 프로젝트 이름은 `project` 값의 `/`를 `-`로 바꾼 것(`runs-exp2`)이 됩니다.

<details>
<summary>Faster R-CNN(torchvision)일 때</summary>

COCO 파일의 category id는 K-코드 숫자입니다. 그대로 label로 쓰지 않고, categories의 `idx`에 1을 더해 씁니다. (0은 배경)
```python
import json
from torchvision.datasets import CocoDetection

train_set = CocoDetection(DS / "images/train", DS / "coco/train.json")
val_set = CocoDetection(DS / "images/val", DS / "coco/val.json")
cats = json.loads((DS / "coco/train.json").read_text(encoding="utf-8"))["categories"]
label_of = {c["id"]: c["idx"] + 1 for c in cats}    # K-코드 → 학습 label
category_of = {v: k for k, v in label_of.items()}   # 예측 label → K-코드 (6단계에서 씀)
num_classes = len(cats) + 1                         # 배경 포함
```
</details>

**After**
```diff
  pill-object-detection/
  └── runs/                     # git이 무시
+     └── exp2/v2_yolo26s/
+         ├── weights/best.pt
+         ├── weights/last.pt
+         ├── args.yaml         # 이번 학습 설정
+         └── results.csv       # epoch별 손실, mAP
```

### 6. val 채점 (실험마다)
5단계 셀 아래에 이어서 실행합니다.
```python
import json, shutil
from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics

RUN = Path("runs/exp2/v2_yolo26s")
model = YOLO(str(RUN / "weights/best.pt"))
results = model.predict(str(DS / "images/val"), conf=0.001, stream=True, verbose=False)
scores = evaluate_val(predictions_from_ultralytics(results), DS / "eval/val_gt.json")
print(f"val {scores['val']:.4f}  kaggle {scores['kaggle']:.4f}  test_only {scores['test_only']:.4f}")

(RUN / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
shutil.copy(DS / "dataset_info.json", RUN)
```
- 대회 클래스 밖 알약에 낸 예측과 `others` 예측은 채점에서 자동으로 빠집니다. 채점 규칙은 [src/pill_detection/evaluate.py](src/pill_detection/evaluate.py) 모듈 docstring을 봅니다.

| 점수 | 범위 | 쓰는 곳 |
| --- | --- | --- |
| `val` | val 1,000장 전체 | 방법과 모델 비교의 기준 |
| `kaggle` | Kaggle 사진 47장만 (1차 val과 같음) | 1차 실험 점수와 비교 |
| `test_only` | test에만 나오는 18개 클래스 | AI Hub로만 배운 클래스를 잘 찾는지 |

<details>
<summary>Faster R-CNN일 때 예측을 바꾸는 법</summary>

```python
from pill_detection.utils import xyxy_to_xywh

predictions = []
for file_name, out in outputs:   # out = model([image])[0], file_name은 val 이미지 파일명
    for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
        predictions.append({"file_name": file_name, "category_id": category_of[label],
                            "bbox": xyxy_to_xywh(box), "score": score})
scores = evaluate_val(predictions, DS / "eval/val_gt.json")
```
</details>

**After**: 화면에 점수 3개가 한 줄로 나오고, run 폴더에 파일 2개가 생깁니다.
```text
val 0.____  kaggle 0.____  test_only 0.____
```
```diff
  runs/exp2/v2_yolo26s/
  ├── weights/ ...
+ ├── scores.json               # 점수 3개 + 클래스별 AP
+ └── dataset_info.json         # 어떤 목록으로 학습했는지
```

### 7. 결과 기록 (실험마다)
팀 결과표에 한 줄을 추가합니다. 값은 모두 run 폴더에 있습니다.

| 항목 | 어디서 |
| --- | --- |
| 방법, 모델 | run 이름 (`v2_yolo26s`) |
| 학습 설정 (공통값과 다르게 바꾼 것) | `args.yaml` |
| val, kaggle, test_only | `scores.json` |
| git commit | `dataset_info.json`의 `git_commit` |
| best.pt 위치 | `runs/exp2/v2_yolo26s/weights/best.pt` |

**After**: 결과표 한 줄 (형식 예시. 폴더는 바뀌지 않습니다)

| run | 방법 | 모델 | val | kaggle | test_only | commit |
| --- | --- | --- | ---: | ---: | ---: | --- |
| v2_yolo26s | v2 | yolo26s | 0.____ | 0.____ | 0.____ | `<git_commit 앞 7자리>` |

### 8. Kaggle 제출 (val로 고른 run만, 하루 10회)
6단계 셀 아래에 이어서 실행합니다.
```python
from pill_detection.evaluate import write_submission, competition_from_classes

results = model.predict("data/raw/test_images", conf=0.01, stream=True, verbose=False)
n = write_submission(predictions_from_ultralytics(results), "submissions/exp2_v2_yolo26s.csv",
                     competition_from_classes(DS / "classes.csv"))
print(n, "줄")
```
- 대회 클래스 74개 밖의 예측(`others` 포함)은 CSV에서 자동으로 빠집니다. 만든 CSV를 Kaggle에 올립니다.

**After**
```diff
  pill-object-detection/
  └── submissions/              # git이 무시
+     └── exp2_v2_yolo26s.csv
```
```text
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score    ← 첫 줄 (열 이름)
```
- `image_id`는 test 파일 번호(`12.png` → 12)이고, `annotation_id`는 1부터 매깁니다.

---

## Colab에서 학습
Colab은 세션이 끝나면 `/content`가 모두 지워집니다. 그래서 2~4단계는 **세션마다** 다시 하고, 학습 결과는 처음부터 **Drive에 바로** 저장합니다.

| 언제 | 단계 |
| --- | --- |
| 처음 한 번 | 1. Drive와 런타임 준비 |
| 세션마다 | 2. 세션 시작 → 3. 데이터 풀기 → 4. 학습 폴더 만들기 |
| 실험마다 | 5. 학습 → 6. val 채점 → 7. 기록 |
| val로 고른 run만 | 8. 제출 |

### 1. Drive와 런타임 준비 (처음 한 번)
- Google Drive에서 데이터 담당이 공유한 폴더를 **내 드라이브에 바로가기로 추가**합니다. 그래야 Colab에서 경로로 열 수 있습니다. 바로가기는 내 용량을 쓰지 않습니다.
- 바로가기 이름을 기억해 둡니다. 2단계의 `ZIP_DIR`에 씁니다.
- Colab 메뉴 **런타임 → 런타임 유형 변경**에서 GPU를 고릅니다.

**After**
```diff
  내 드라이브/
+ └── <바로가기 이름>/            # → 공유 폴더
+     ├── kaggle.zip
+     ├── aihub_annotations.zip
+     ├── aihub_images.zip       # 약 21.6GB
+     └── test_images.zip        # 약 1.5GB
```

### 2. 세션 시작: Drive 연결, 레포 받기, 패키지 설치 (세션마다)
```python
from google.colab import drive
drive.mount('/content/drive')

ZIP_DIR = '/content/drive/MyDrive/<바로가기 이름>'   # zip 4개가 있는 폴더
SAVE_DIR = '/content/drive/MyDrive/pill_exp2'        # 학습 결과, 점수, 제출 CSV를 남길 내 폴더
REPO = '/content/pill-object-detection'

!test -d {REPO} || git clone -q https://github.com/codeit-ai-15th/pill-object-detection.git {REPO}
%cd {REPO}
!pip install -q ultralytics==8.4.183 pycocotools==2.0.11

import sys; sys.path.insert(0, f'{REPO}/src')
```
- 패키지 버전은 레포의 `uv.lock`과 같습니다. 레포의 공용 코드(`pill_detection`)는 마지막 줄로 불러옵니다.

**After**
```diff
  /content/
+ ├── drive/MyDrive/              # Drive 연결됨
+ └── pill-object-detection/      # main 최신
+     ├── data/datasets/          # images.csv, classes.csv
+     ├── data/raw/, processed/   # 비어 있음
+     ├── scripts/
+     └── src/pill_detection/
```

### 3. 데이터 풀기 (세션마다)
```python
for z in ['kaggle', 'aihub_annotations', 'aihub_images']:   # test_images는 8단계에서
    !unzip -q -o "{ZIP_DIR}/{z}.zip" -d {REPO}
```
- Colab 디스크를 약 23GB 씁니다. `aihub_images.zip`이 21.6GB라 시간이 걸립니다.
- **첫 세션에서만** 받은 데이터가 팀과 같은지 확인합니다. 마지막 두 줄이 "같음"이어야 합니다. "다름"이면 zip을 다시 풀어 보고, 그래도 다르면 데이터 담당에게 알립니다.
    ```python
    !PYTHONPATH=src python scripts/make_dataset_lists.py --check
    ```

**After**: 폴더
```diff
  /content/pill-object-detection/data/
  ├── datasets/                     # 그대로
  ├── raw/
+ │   └── train_images/             # PNG 232장
  └── processed/
+     ├── train_annotations/        # JSON 771개
+     └── ai_hub/
+         ├── train_annotations/    # JSON 45,819개
+         └── train_images/         # PNG 11,974장
```
**After**: 확인 출력 (끝부분)
```text
data/datasets/images.csv: 같음
data/datasets/classes.csv: 같음
```

### 4. 방법별 학습 폴더 만들기 (세션마다)
```python
!PYTHONPATH=src python scripts/build_dataset.py --variant v2 --out /content/exp2/v2
DS = '/content/exp2/v2'
```
- `--variant`에 맡은 방법을 줍니다.
- `--out`은 세션마다 **같은 경로**로 씁니다. 끊긴 학습을 이어서 할 때 이 경로가 있어야 합니다.
- 목록 CSV대로 만들기 때문에 언제 만들어도 내용이 같습니다.

**After**: 화면 출력 (v2)
```text
[v2] 학습 데이터를 읽고 /content/exp2/v2에 학습 폴더를 만듭니다.
클래스 118개, 이미지 train 7,129장 / val 1,000장, 객체 train 26,734개 / val 3,790개
이미지: {'hardlink': 8129}
YOLO 학습: data=/content/exp2/v2/data.yaml
COCO 학습: root=/content/exp2/v2/images/train, annFile=/content/exp2/v2/coco/train.json (val도 같은 모양)
채점 정답: /content/exp2/v2/eval/val_gt.json
```
**After**: 폴더
```diff
  /content/
+ └── exp2/v2/
+     ├── images/train/, val/   # 7,129장, 1,000장 (하드링크)
+     ├── labels/train/, val/   # YOLO txt
+     ├── data.yaml             # nc 118
+     ├── coco/                 # train.json, val.json
+     ├── eval/val_gt.json      # 채점 정답
+     ├── classes.csv
+     └── dataset_info.json
```

### 5. 학습 (실험마다, 결과는 Drive에 바로 저장)
YOLO, RT-DETR
```python
from ultralytics import YOLO          # RT-DETR: from ultralytics import RTDETR

model = YOLO('yolo26s.pt')            # 맡은 모델. RT-DETR: RTDETR('rtdetr-l.pt')
model.train(
    data=f'{DS}/data.yaml',
    project=SAVE_DIR, name='v2_yolo26s',   # Drive에 바로 저장 → 세션이 끊겨도 남음
    optimizer=..., lr0=..., epochs=..., imgsz=..., batch=..., seed=...,   # 팀 공통값
    cache=False,
)
```
- 세션이 끊겼으면 2~4단계를 같은 경로로 다시 한 뒤 이어서 학습합니다.
    ```python
    model = YOLO(f'{SAVE_DIR}/v2_yolo26s/weights/last.pt')
    model.train(resume=True)
    ```
- 학습 중에 Ultralytics가 출력하는 mAP50-95로는 방법끼리 비교하지 않습니다. 비교는 6단계의 점수 3개로 합니다.

<details>
<summary>Faster R-CNN(torchvision)일 때</summary>

category id(K-코드)를 그대로 label로 쓰지 않고 `idx + 1`을 씁니다. (0은 배경) 체크포인트는 `SAVE_DIR` 아래에 직접 저장합니다.
```python
import json
from torchvision.datasets import CocoDetection

train_set = CocoDetection(f'{DS}/images/train', f'{DS}/coco/train.json')
val_set = CocoDetection(f'{DS}/images/val', f'{DS}/coco/val.json')
cats = json.load(open(f'{DS}/coco/train.json', encoding='utf-8'))['categories']
label_of = {c['id']: c['idx'] + 1 for c in cats}     # K-코드 → 학습 label
category_of = {v: k for k, v in label_of.items()}    # 예측 label → K-코드 (6단계에서 씀)
num_classes = len(cats) + 1
```
</details>

**After**
```diff
  내 드라이브/
  ├── <바로가기 이름>/
+ └── pill_exp2/
+     └── v2_yolo26s/
+         ├── weights/best.pt
+         ├── weights/last.pt   # epoch마다 갱신 → 이어서 학습할 때 씀
+         ├── args.yaml
+         └── results.csv
```

### 6. val 채점 (실험마다)
```python
import json, shutil
from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics

RUN = f'{SAVE_DIR}/v2_yolo26s'
model = YOLO(f'{RUN}/weights/best.pt')
results = model.predict(f'{DS}/images/val', conf=0.001, stream=True, verbose=False)
scores = evaluate_val(predictions_from_ultralytics(results), f'{DS}/eval/val_gt.json')
print(f"val {scores['val']:.4f}  kaggle {scores['kaggle']:.4f}  test_only {scores['test_only']:.4f}")

with open(f'{RUN}/scores.json', 'w', encoding='utf-8') as f:
    json.dump(scores, f, indent=2)
shutil.copy(f'{DS}/dataset_info.json', RUN)   # /content는 지워지므로 Drive에 남김
```
- 점수 3개의 뜻은 [로컬 6단계](#6-val-채점-실험마다)의 표와 같습니다.
- Faster R-CNN은 [로컬 6단계](#6-val-채점-실험마다)의 변환 코드에서 경로만 `f'{DS}/...'`로 바꿔 씁니다.

**After**
```text
val 0.____  kaggle 0.____  test_only 0.____
```
```diff
  내 드라이브/pill_exp2/v2_yolo26s/
  ├── weights/ ...
+ ├── scores.json               # 점수 3개 + 클래스별 AP
+ └── dataset_info.json         # 어떤 목록으로 학습했는지
```

### 7. 결과 기록 (실험마다)
[로컬 7단계](#7-결과-기록-실험마다)와 같습니다. best.pt 위치는 `내 드라이브/pill_exp2/v2_yolo26s/weights/best.pt`입니다.

### 8. Kaggle 제출 (val로 고른 run만, 하루 10회)
```python
!unzip -q -o "{ZIP_DIR}/test_images.zip" -d {REPO}

from pill_detection.evaluate import write_submission, competition_from_classes

results = model.predict(f'{REPO}/data/raw/test_images', conf=0.01, stream=True, verbose=False)
n = write_submission(predictions_from_ultralytics(results), f'{SAVE_DIR}/submissions/exp2_v2_yolo26s.csv',
                     competition_from_classes(f'{DS}/classes.csv'))
print(n, '줄')
```
- 대회 클래스 74개 밖의 예측(`others` 포함)은 CSV에서 자동으로 빠집니다. Drive에서 CSV를 받아 Kaggle에 올립니다.

**After**
```diff
  /content/pill-object-detection/data/raw/
+ └── test_images/                # PNG 842장
  내 드라이브/pill_exp2/
+ └── submissions/exp2_v2_yolo26s.csv
```

---

## 1차 모델 다시 채점하기 (1차 best.pt를 가진 사람만, 한 번)
- 2차 점수와 비교하려면 1차 모델도 같은 채점 함수로 잰 점수가 있어야 합니다. 1차 best.pt를 2차 val로 채점해서 `kaggle` 점수를 1차 기준점으로 씁니다.
- 학습 폴더는 아무 방법으로 만들어도 됩니다. val은 네 방법이 같습니다.
```python
from ultralytics import RTDETR
from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics

model = RTDETR('<1차 best.pt 경로>')
print(model.names[0])      # K-코드 숫자 글자('1900' 같은 형식)여야 채점할 수 있음
results = model.predict(f'{DS}/images/val', conf=0.001, stream=True, verbose=False)
scores = evaluate_val(predictions_from_ultralytics(results), f'{DS}/eval/val_gt.json')
print(scores['kaggle'])    # 1차 기준점
```
- 1차 모델은 Kaggle 56개 클래스만 배웠으므로 `val`과 `test_only`는 낮게 나옵니다. `kaggle` 점수만 비교합니다.

## 막히면
| 화면에 나온 것 | 원인 | 해결 |
| --- | --- | --- |
| `[오류] 목록 CSV가 없습니다` | 레포가 옛 버전 | `git pull origin main` |
| `학습 데이터 annotation 폴더가 없습니다` | 사본이 없거나 zip을 다른 곳에 풂 | 로컬은 3단계, Colab은 레포 최상위에서 zip 다시 풀기 |
| `목록의 이미지 N장이 학습 데이터에 없습니다`, `PNG 파일이 없습니다` | 사본이 옛 것이거나 zip이 덜 풀림 | 로컬은 3단계, Colab은 zip 다시 풀기 |
| `--check`에서 `다름` | 옛 정제 목록으로 만든 사본이거나 zip이 덜 풀림 | 로컬은 3단계, Colab은 zip 다시 풀기. CSV는 커밋하지 않기 |
| `[AI Hub] ... 받을 것 N개` (N이 0이 아님) | `data/external/aihub/raw/`가 덜 풀림 | `.env`에 `AIHUB_APIKEY`를 넣고 그대로 진행하거나, 데이터 담당에게 묻기 |
| `출력 폴더가 비어 있지 않습니다` | 같은 `--out`에 이미 만듦 | 그 폴더를 지우거나 다른 `--out` |
| `출력 폴더가 레포 안인데 .gitignore에 걸리지 않습니다` | 커밋에 올라갈 수 있는 곳 | `--out data/processed/exp2/<방법>` 또는 레포 밖 |
| `이미지: {'copy': ...}` | 다른 디스크라 하드링크 불가 | `--out`을 `data`와 같은 디스크로 |
| `val 정답에 없는 이미지의 예측이 N장` | val이 아닌 폴더(test 등)로 예측 | `<out>/images/val`로 예측 |
| 채점 중 `invalid literal for int()` | 모델 클래스 이름이 K-코드 숫자가 아님 | 4단계 폴더의 `data.yaml`로 학습한 모델인지 확인 |

## 하지 말 것
- `data/datasets/images.csv`, `classes.csv`를 손으로 고치지 않습니다. 고치면 val이 바뀌어 다른 팀원의 실험과 비교할 수 없습니다.
- `--check`가 "다름"인 상태에서 `make_dataset_lists.py`를 `--check` 없이 돌리거나, 바뀐 CSV를 커밋하지 않습니다.
- 학습 폴더, best.pt, 제출 CSV를 `git add -f`로 올리지 않습니다.
