# 2차 실험 가이드
학습을 돌리는 팀원이 그대로 따라 하면 되도록 가이드를 작성했습니다. 자기 환경에 맞는 절([로컬](#로컬에서-학습) 또는 [Colab](#colab에서-학습))로 이동하셔서 1번부터 따라하시면 됩니다.

- 단계마다 **After**에 그 단계를 끝낸 뒤의 예상 폴더 상태와 화면 출력이 있습니다. 내 화면이 같으면 다음 단계로 넘어가시면 됩니다.
- After 폴더 그림에서 `+`로 시작하는 줄(초록)은 그 단계에서 새로 생긴 것이고, 나머지는 이미 있던 것입니다.
- `<...>`로 표시된 부분은 각자 상황에 맞게 바꿔서 입력합니다.
- 2차 실험은 이미지 선택 방법 4가지(v1~v4)를 같은 val로 비교합니다. 목록 CSV의 컬럼 설명은 [data/datasets/README.md](data/datasets/README.md)를 확인해주세요.

## 시작 전에
- [x] PR #40이 main에 머지되었다. (목록 CSV, 학습 폴더 스크립트, 채점 코드)
- [x] (Colab에서 학습하거나 로컬에 데이터가 없는 사람만) 데이터 담당이 공유 드라이브 `pill data/data/`에 zip 4개를 올렸다: `kaggle.zip`, `aihub_annotations.zip`, `aihub_images.zip`, `test_images.zip`
- [x] 내가 맡은 **방법(v1~v4)** 과 **모델**이 정해졌다.
- [x] W&B 팀 `codeit-prj1-team2`의 멤버다. 모든 학습은 팀 프로젝트 `pill-object-detection`에 기록하며, W&B 기록이 시작되지 않으면 학습이 멈춥니다.
- [x] 팀 공통 학습 설정값을 받았다. 방법끼리 비교하려면 데이터 말고는 모두 같아야 합니다.
    - `optimizer`, `lr0`를 직접 적습니다. `auto`로 두면 학습률이 클래스 수로 정해지는데, 클래스 수가 방법마다 달라서(118, 118, 96, 75) 비교가 깨집니다.
    - run 이름은 `<방법>_<모델>`로 작성해주세요. (예: `v2_yolo26s`)

| 방법 | 뜻 | 모델 클래스 | train 장수 | train 객체 수 |
| --- | --- | ---: | ---: | ---: |
| v1 | 사진 전부 | 118 | 11,206 | 42,800 |
| v2 | 클래스별 200개를 목표로 고른 사진 | 118 | 7,129 | 26,734 |
| v3 | 대회 클래스가 든 사진만 | 96 | 8,974 | 33,872 |
| v4 | v1과 같은 사진, 대회 클래스 밖 알약은 `others` 한 클래스 | 75 | 11,206 | 42,800 |
> v1은 일단 실험 후보에서 제외합니다. v2~v4만 돌려주세요
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

**After**: `data/datasets/`에 파일 3개가 생깁니다.
```diff
  pill-object-detection/
  ├── data/
+ │   ├── datasets/
+ │   │   ├── images.csv        # 이미지 목록 12,206줄
+ │   │   ├── classes.csv       # 클래스 표 119줄
+ │   │   └── README.md         # 두 CSV 설명
  │   ├── raw/                  # Kaggle 제공 원본 데이터
  │   └── processed/            # 전처리 후 가공된 데이터
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
# 프로젝트 루트에서 실행하세요
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

`prepare_train_data.py`를 처음부터 돌리는 대신, 데이터 담당이 공유 드라이브 `pill data/data/`에 올린 zip 4개를 받아서 풀어도 됩니다. 디스크는 약 50GB가 필요합니다. (zip을 지운 뒤에는 약 24GB)
- zip 4개(합계 약 23.5GB)를 **레포 밖** 폴더에 다운로드 받아주세요. 아래 예시는 `다운로드/pill_drive`입니다.
    - `data/` 폴더를 통째로 받지 말고 zip을 하나씩 받습니다. 폴더째 받으면 Drive가 zip들을 다시 압축해서 여러 조각으로 나눠 줍니다.
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

### 4. 데이터 구성 방법별 학습 폴더 만들기 (방법마다 한 번)

> 이 문서에서는 v2를 예시로 들었습니다. v1, v3, v4도 마찬가지로 돌리면 됩니다.

```bash
uv run scripts/build_dataset.py --variant v2 --out data/processed/exp2/v2
```
- `--variant`에 방법(v1~v4)을, `--out`에 `data/processed/exp2/<방법>`을 줍니다.
- 이미지는 복사하지 않고 원본에 하드링크를 겁니다.
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
YOLO, RT-DETR: 노트북에서 셀 두 개를 실행합니다. 준비 셀은 커널을 새로 띄울 때마다(이어서 학습, 6단계, 8단계 전에도) 먼저 실행합니다.

**준비 셀**: 경로와 W&B
```python
import os
from pathlib import Path
import wandb
from ultralytics import YOLO, settings          # RT-DETR: from ultralytics import RTDETR, settings

# 노트북이 notebooks/ 아래에 있어도 레포 최상위로 옮긴다. 아래 경로는 모두 레포 최상위 기준
os.chdir(next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file()))
DS = Path("data/processed/exp2/v2")
RUN = Path("runs/exp2/v2_yolo26s")              # 결과 폴더. 폴더 이름(run 이름)이 W&B 실행 이름이 된다

# W&B 팀 프로젝트 codeit-prj1-team2/pill-object-detection에 기록한다
os.environ["WANDB_ENTITY"] = "codeit-prj1-team2"
settings.update({"wandb": True})


def require_wandb(trainer):
    """첫 epoch 직전에 불린다. W&B 기록이 팀 프로젝트에 시작되지 않았으면 학습을 멈추고, 시작됐으면 2차 실험 태그를 붙인다."""
    if wandb.run is None:
        raise RuntimeError("W&B 기록이 시작되지 않았습니다. 커널을 다시 시작하고, Ultralytics를 쓰기 전에 준비 셀부터 실행하세요.")
    if wandb.run.entity != "codeit-prj1-team2":
        raise RuntimeError(f"W&B 기록이 팀이 아닌 {wandb.run.entity}에 시작됐습니다. 커널을 다시 시작하고, wandb를 쓰기 전에 준비 셀부터 실행하세요.")
    wandb.run.tags = sorted({*wandb.run.tags, "exp2", DS.name})
```

**학습 셀**
```python
assert not RUN.exists(), f"{RUN}이 이미 있습니다. 이어서 학습이면 아래 resume 셀을, 새 실험이면 run 이름을 바꿉니다."
model = YOLO("yolo26s.pt")            # 맡은 모델. RT-DETR: RTDETR("rtdetr-l.pt")
model.add_callback("on_train_start", require_wandb)
model.train(
    data=str(DS / "data.yaml"),
    project="pill-object-detection", save_dir=str(RUN),   # project = W&B 프로젝트 이름, 결과는 save_dir에 저장
    optimizer=..., lr0=..., epochs=..., imgsz=..., batch=..., seed=...,   # 팀 공통값
    cache=False,
    device=0,                         # NVIDIA GPU. Apple Silicon은 "mps"
)
```
- 학습이 시작되면 W&B 팀 프로젝트에 실행이 바로 생기고, epoch마다 손실, 학습률, mAP가 올라갑니다. 실행 이름은 run 이름(`v2_yolo26s`)이고, 태그 `exp2`와 방법(`v2`)이 붙습니다. 학습이 끝나면 best.pt도 올라갑니다.
- Ultralytics는 `project` 값을 W&B 프로젝트 이름으로 씁니다. 그래서 `project`에는 W&B 프로젝트 이름을 주고, 결과 폴더는 `save_dir`로 따로 줍니다.
- W&B 기록이 시작되지 않으면 `require_wandb`가 첫 epoch 전에 학습을 멈춥니다. 같은 커널에서 준비 셀보다 먼저 Ultralytics로 학습이나 예측을 했으면 이렇게 됩니다.
- 기록이 팀(`codeit-prj1-team2`)이 아닌 곳에서 시작돼도 멈춥니다. wandb는 커널에서 처음 쓰일 때 한 번만 `WANDB_ENTITY`를 읽기 때문에, 준비 셀보다 먼저 wandb를 쓰면(`wandb.login()` 등) 내 계정의 기본 entity에 기록됩니다.
- 학습 중에 Ultralytics가 출력하는 mAP50-95로는 방법끼리 비교하지 않습니다. 대회 클래스 밖 클래스까지 평균하고, IoU 기준도 대회(0.75~0.95)와 다릅니다. 비교는 6단계의 점수 3개로 합니다.
- 학습이 중간에 멈췄으면 준비 셀을 실행한 뒤 아래 resume 셀로 이어서 학습합니다. W&B에도 같은 실행에 이어서 기록됩니다. 동작은 [Colab의 '세션이 끊겼을 때'](#세션이-끊겼을-때-이어서-학습)와 같습니다.
    ```python
    model = YOLO(str(RUN / "weights/last.pt"))
    model.add_callback("on_train_start", require_wandb)
    model.train(resume=True)
    ```

<details>
<summary>Faster R-CNN(torchvision)일 때</summary>

학습 코드는 [notebooks/modeling/exp2_colab_smoke_test.ipynb](notebooks/modeling/exp2_colab_smoke_test.ipynb)의 Torchvision 구역(준비, 스모크 테스트 셀)을 바탕으로 짭니다. Colab에서 학습, 채점, 제출까지 돌려 확인한 코드입니다. 아래는 꼭 지킬 것입니다.
- **label**: COCO 파일의 category id(K-코드)를 그대로 쓰지 않고 categories의 `idx + 1`을 씁니다. (0은 배경) 예측 label은 `category_of`로 K-코드로 되돌립니다.
- **bbox 형식**: COCO 파일의 bbox는 xywh이고 torchvision은 xyxy를 받습니다. 학습 때 `torchvision.ops.box_convert(boxes, 'xywh', 'xyxy')`로 바꾸고, 예측은 `xyxy_to_xywh`로 되돌려 채점합니다. 형식을 섞어도 에러 없이 점수만 떨어집니다.
- **예측 점수 하한**: 모델을 만들 때 `box_score_thresh=0.001`을 줍니다. 기본값(0.05)으로 두면 YOLO 채점(`conf=0.001`)과 조건이 달라집니다.
- **W&B**: 학습 루프 앞에서 `wandb.init(entity="codeit-prj1-team2", project="pill-object-detection", name=RUN.name, tags=["exp2", DS.name], config=<학습 설정>, dir=str(RUN))`로 시작하고, `wandb.log`로 기록합니다. `RUN` 폴더를 먼저 만들어야 합니다. 폴더가 없으면 W&B가 임시 폴더에 기록해서 6단계가 같은 실행을 찾지 못합니다.
- **체크포인트**: epoch마다 `RUN/weights/last.pt`에 model, optimizer, epoch, 학습 설정을 저장합니다.
- 노트북 코드를 로컬에서 쓸 때는 Colab 전용 셀(각자 바꿀 값, 세션 시작, 데이터 풀기) 대신 위 준비 셀의 `DS`, `RUN`을 쓰고, 공용 셀의 `SAVE_DIR` 검사는 뺍니다. macOS와 Windows의 Jupyter에서는 노트북에서 정의한 Dataset을 DataLoader worker가 불러오지 못해 오류가 날 수 있으니 `num_workers=0`으로 바꿉니다.
</details>

**After**
```diff
  pill-object-detection/
  └── runs/                     # git이 무시
+     └── exp2/v2_yolo26s/
+         ├── weights/best.pt
+         ├── weights/last.pt
+         ├── wandb/            # W&B 기록. 이어서 학습하거나 점수를 붙일 때 같은 실행을 여기서 찾음
+         ├── args.yaml         # 이번 학습 설정
+         └── results.csv       # epoch별 손실, mAP
```

### 6. val 채점 (실험마다)
5단계 셀 아래에 이어서 실행합니다. 커널을 새로 띄웠으면 5단계 준비 셀을 먼저 실행합니다.
```python
import json, shutil
from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics

model = YOLO(str(RUN / "weights/best.pt"))
results = model.predict(str(DS / "images/val"), conf=0.001, stream=True, verbose=False)
scores = evaluate_val(predictions_from_ultralytics(results), DS / "eval/val_gt.json")
print(f"val {scores['val']:.4f}  kaggle {scores['kaggle']:.4f}  test_only {scores['test_only']:.4f}")

(RUN / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
shutil.copy(DS / "dataset_info.json", RUN)

# 점수와 데이터 정보를 학습 때의 W&B 실행에 붙인다
run_id = max((RUN / "wandb").glob("*run-*/run-*.wandb"), key=lambda f: f.stat().st_mtime).stem[4:]
with wandb.init(entity="codeit-prj1-team2", project="pill-object-detection", id=run_id, resume="allow", dir=str(RUN)) as run:
    run.summary.update({f"exp2/{k}": scores[k] for k in ("val", "kaggle", "test_only")})
    run.config.update({"dataset_info": json.loads((DS / "dataset_info.json").read_text(encoding="utf-8"))})
```
- 대회 클래스 밖 알약에 낸 예측과 `others` 예측은 채점에서 자동으로 빠집니다. 채점 규칙은 [src/pill_detection/evaluate.py](src/pill_detection/evaluate.py) 모듈 docstring을 봅니다.
- 점수 3개는 모두 대회 지표인 **mAP@[0.75:0.95]**(IoU 기준 0.75, 0.80, 0.85, 0.90, 0.95의 AP 평균)이고, 채점하는 범위만 다릅니다. W&B에 함께 보이는 `metrics/mAP50-95(B)`는 Ultralytics가 학습 중에 잰 다른 지표입니다(IoU 0.50~0.95, 모델 클래스 전부).
- 점수 3개는 W&B 실행의 Summary에 `exp2/val`, `exp2/kaggle`, `exp2/test_only`로 붙습니다. W&B 실행 표에서 태그 `exp2`로 거른 뒤 이 열로 방법과 모델을 비교합니다.

| 점수 | 범위 | 쓰는 곳 |
| --- | --- | --- |
| `val` | val 1,000장 전체 | 방법과 모델 비교의 기준 |
| `kaggle` | Kaggle 사진 47장만 (1차 val과 같음) | 1차 실험 점수와 비교 |
| `test_only` | test에만 나오는 18개 클래스 | AI Hub로만 배운 클래스를 잘 찾는지 |

<details>
<summary>Faster R-CNN일 때</summary>

[노트북](notebooks/modeling/exp2_colab_smoke_test.ipynb) Torchvision 구역의 val 채점 셀을 씁니다.
- `predict()`가 예측을 채점 형식(file_name, K-코드, xywh, score)으로 바꿉니다.
- `save_scores()`(공용 셀)가 위 셀과 같은 일을 합니다. `scores.json`과 `dataset_info.json`을 저장하고, W&B Summary에 점수 3개를 붙입니다.
- val 예측은 `val_predictions.json`으로 run 폴더에 남깁니다. `eval/val_gt.json`과 함께 confusion matrix를 그릴 때 씁니다.
</details>

**After**: 화면에 점수 3개가 한 줄로 나오고, run 폴더에 파일 2개가 생깁니다. W&B 실행의 Summary에도 점수 3개가 생깁니다.
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
`notebooks/modeling/{실험 이름}/` 하위에 `runs.csv` 를 만들어서 한 줄을 추가합니다. 값은 모두 run 폴더에 있습니다.

| 항목 | 어디서 |
| --- | --- |
| 방법, 모델 | run 이름 (`v2_yolo26s`) |
| 학습 설정 (공통값과 다르게 바꾼 것) | `args.yaml` (Faster R-CNN은 W&B 실행의 Config, 또는 `last.pt`의 `config`) |
| val, kaggle, test_only | `scores.json` |
| git commit | `dataset_info.json`의 `git_commit` |
| best.pt 위치 | `pill data/<cj_runs 또는 jw_runs>/v2_yolo26s/weights/best.pt` (아래에서 올린 곳). Faster R-CNN은 val로 고른 체크포인트의 위치 |

run 폴더는 공유 드라이브에 올립니다.
- `runs/`는 git이 무시하므로, 팀원이 best.pt와 점수를 보려면 Drive에 있어야 합니다.
- run 폴더 `runs/exp2/v2_yolo26s/`를 통째로 공유 드라이브 `pill data/<cj_runs 또는 jw_runs>/`에 올립니다. Drive 웹에서 그 폴더를 열고 **새로 만들기 → 폴더 업로드**로 올리면 됩니다.
- Colab에서 학습한 run은 처음부터 이 위치에 저장되므로 올리지 않아도 됩니다.

**After**: 결과표 한 줄 (형식 예시)과 Drive 폴더

| run | 방법 | 모델 | val | kaggle | test_only | commit |
| --- | --- | --- | ---: | ---: | ---: | --- |
| v2_yolo26s | v2 | yolo26s | 0.____ | 0.____ | 0.____ | `<git_commit 앞 7자리>` |

```diff
  pill data/
  ├── data/                       # zip 4개
+ └── <cj_runs 또는 jw_runs>/
+     └── v2_yolo26s/             # run 폴더 통째로
+         ├── weights/best.pt
+         ├── scores.json
+         ├── dataset_info.json
+         └── ...
```

### 8. Kaggle 제출
6단계 셀 아래에 이어서 실행합니다.
```python
from pill_detection.evaluate import write_submission, competition_from_classes

results = model.predict("data/raw/test_images", conf=0.01, stream=True, verbose=False)
n = write_submission(predictions_from_ultralytics(results), "submissions/exp2_v2_yolo26s.csv",
                     competition_from_classes(DS / "classes.csv"))
print(n, "줄")
```
- 대회 클래스 74개 밖의 예측(`others` 포함)은 CSV에서 자동으로 빠집니다. 만든 CSV를 Kaggle에 올립니다.
- Faster R-CNN은 [노트북](notebooks/modeling/exp2_colab_smoke_test.ipynb) Torchvision 구역의 Kaggle 제출 셀을 씁니다. test 폴더에는 COCO 파일이 없어서 `TestImages`로 PNG를 직접 읽습니다. 점수 0.01 미만 예측은 위 `conf=0.01`처럼 뺍니다.

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
Colab은 세션이 끝나면 `/content`가 모두 지워집니다. 그래서 2~4단계는 **세션마다** 다시 하고, 학습 결과는 매 에폭 **Drive에 바로** 저장해서 끊긴 세션 이후에 이어나갈 수 있게 합니다.

학습 과정과 정보 기록은 confusion matrix를 그릴 수 있는 정보들을 필수적으로 포함해야 합니다.

[notebooks/modeling/exp2_colab_smoke_test.ipynb](notebooks/modeling/exp2_colab_smoke_test.ipynb)에 이 절의 2~8단계가 셀로 들어 있습니다. Ultralytics와 Torchvision(Faster R-CNN) 모두 있습니다. 본 학습 전에 노트북 맨 위 **각자 바꿀 값** 셀(`ZIP_DIR`, `SAVE_DIR`, `VARIANT`)을 채우고, 스모크 테스트로 W&B 기록과 Drive 저장이 되는지 먼저 확인합니다.

| 언제 | 단계 |
| --- | --- |
| 처음 한 번 | 1. Drive와 런타임 준비 |
| 세션마다 | 2. 세션 시작 → 3. 데이터 풀기 → 4. 학습 폴더 만들기 |
| 실험마다 | 5. 학습 → 6. val 채점 → 7. 기록 |
| val로 고른 run만 | 8. 제출 |

### 1. Drive와 런타임 준비 (처음 한 번)
- Google Drive에서 데이터 담당이 팀 구글 드라이브 `pill data/data/` 하위에 데이터를 저장해놨을 겁니다. cy님이 공유해주신 `pill data`를 **내 드라이브에 바로가기로 추가**합니다. 그래야 Colab에서 경로로 열 수 있습니다.
- 바로가기 이름은 `pill data` 그대로 둡니다. 2단계의 `ZIP_DIR`, `SAVE_DIR`가 이 이름으로 경로를 씁니다. (이름을 바꿨으면 두 경로도 바꿉니다)
- Colab 메뉴 **런타임 → 런타임 유형 변경**에서 GPU를 고릅니다.

**After**
```diff
  MyDrive/
+ └── pill data/                  # → 공유 폴더 바로가기
+     ├── data/
+     │   ├── kaggle.zip
+     │   ├── aihub_annotations.zip
+     │   ├── aihub_images.zip    # 약 21.6GB
+     │   └── test_images.zip     # 약 1.5GB
+     └── <cj_runs 또는 jw_runs>/  # 학습 기록. 없으면 5단계에서 생김
```

### 2. 세션 시작: Drive 연결, 레포 받기, 패키지 설치 (세션마다)
```python
from google.colab import drive
drive.mount('/content/drive')

ZIP_DIR = '/content/drive/MyDrive/pill data/data'   # zip 4개가 있는 폴더
SAVE_DIR = '/content/drive/MyDrive/pill data/<cj_runs 또는 jw_runs>'   # 학습 기록을 저장할 폴더. 둘 중 하나로 바꿔 씁니다
REPO = '/content/pill-object-detection'
BRANCH = 'main'   # 받을 레포 브랜치. 보통 main 그대로 둡니다

!test -d {REPO} || git clone -q https://github.com/codeit-ai-15th/pill-object-detection.git {REPO}
%cd {REPO}
!git fetch -q origin {BRANCH} && git switch -q -C {BRANCH} FETCH_HEAD && git log -1 --format='%h %s'
!git branch --show-current
!pip install -q ultralytics==8.4.183 pycocotools==2.0.11 wandb==0.31.0

import sys; sys.path.insert(0, f'{REPO}/src')

import os, wandb
os.environ['WANDB_ENTITY'] = 'codeit-prj1-team2'   # W&B 팀. wandb.login()보다 먼저 정해야 적용된다
wandb.login()   # 키를 묻는 칸이 나오면 https://forge.coreweave.com/wandb/authorize 의 키를 붙여 넣습니다
```
- `BRANCH`의 GitHub 최신 커밋을 받습니다. 셀을 다시 실행하면 그 사이 push된 커밋까지 받습니다. 출력 첫 줄의 커밋이 받은 버전입니다.
    - 공용 코드(`src/`, `scripts/`, `data/datasets/`)를 고친 브랜치로 학습할 때만 `BRANCH`를 그 브랜치로 바꿉니다. 그 브랜치를 먼저 GitHub에 push해야 받을 수 있습니다.
    - 실험 노트북만 다른 실험 브랜치라면 main과 코드가 같으므로 `main` 그대로 둡니다.
- 패키지 버전은 레포의 `uv.lock`과 같습니다. 레포의 공용 코드(`pill_detection`)는 `sys.path` 줄로 불러옵니다.
- 경로(`ZIP_DIR`, `SAVE_DIR`)를 고치면 이 셀을 다시 실행해야 적용됩니다. 커널에는 마지막으로 실행한 값이 남아 있습니다.
- W&B 팀(`WANDB_ENTITY`)은 `wandb.login()`보다 **먼저** 정합니다. wandb는 처음 시작할 때 한 번만 환경 변수를 읽어서, 로그인 뒤에 정하면 무시되고 내 계정의 기본 entity에 기록됩니다.
- Colab은 세션마다 새 컴퓨터라서 W&B 로그인도 세션마다 합니다.
  - wandb.login()을 하면, 로그인을 하라는 팔레트가 뜰 겁니다. `(2) Use an existing W&B account`를 선택하시면 됩니다. 
  - 그리고 wandb 웹페이지에서 발급받은 api key를 붙여넣으면 됩니다.

**After**
```diff
  /content/
+ ├── drive/MyDrive/              # Drive 연결됨
+ └── pill-object-detection/      # BRANCH 최신
+     ├── data/datasets/          # images.csv, classes.csv
+     ├── data/raw/, processed/   # 비어 있음
+     ├── scripts/
+     └── src/pill_detection/
```

### 3. 데이터 풀기 (세션마다)
```python
for z in ['kaggle', 'aihub_annotations', 'aihub_images']:   # test_images는 8단계에서
    !unzip -q -o "{ZIP_DIR}/{z}.zip" -d {REPO}

!ls {REPO}/data/processed/ai_hub/train_images | wc -l   # 11974라면 모든 이미지가 풀린 겁니다. 만약 그렇지 않다면 !unzip -q -o "{ZIP_DIR}/aihub_images.zip" -d {REPO} 으로 다시 풀어주세요 
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
YOLO, RT-DETR: 셀 두 개를 실행합니다. 준비 셀은 세션마다(이어서 학습, 6단계, 8단계 전에도) 먼저 실행합니다.

**준비 셀**: 경로와 W&B
```python
from pathlib import Path
import wandb
from ultralytics import YOLO, settings          # RT-DETR: from ultralytics import RTDETR, settings

if '<' in SAVE_DIR or not Path(SAVE_DIR).parent.is_dir():   # 경로가 틀려도 Ultralytics는 오류 없이 새 폴더를 만든다
    raise FileNotFoundError(f'SAVE_DIR의 상위 폴더가 Drive에 없습니다: {SAVE_DIR}\n[해결] 2단계 셀의 SAVE_DIR를 내 Drive 경로로 고치고 2단계부터 다시 실행하세요.')
RUN = Path(SAVE_DIR, 'v2_yolo26s')              # Drive의 결과 폴더. 폴더 이름(run 이름)이 W&B 실행 이름이 된다

# W&B 팀 프로젝트 codeit-prj1-team2/pill-object-detection에 기록한다 (팀은 2단계에서 정했다)
settings.update({'wandb': True})


def require_wandb(trainer):
    """첫 epoch 직전에 불린다. W&B 기록이 팀 프로젝트에 시작되지 않았으면 학습을 멈추고, 시작됐으면 2차 실험 태그를 붙인다."""
    if wandb.run is None:
        raise RuntimeError('W&B 기록이 시작되지 않았습니다. 런타임을 다시 시작하고, Ultralytics를 쓰기 전에 준비 셀부터 실행하세요.')
    if wandb.run.entity != 'codeit-prj1-team2':
        raise RuntimeError(f'W&B 기록이 팀이 아닌 {wandb.run.entity}에 시작됐습니다. 런타임을 다시 시작하고 2단계부터 실행하세요.')
    wandb.run.tags = sorted({*wandb.run.tags, 'exp2', Path(DS).name})
```

**학습 셀**
```python
assert not RUN.exists(), f'{RUN}이 이미 있습니다. 이어서 학습이면 아래 "세션이 끊겼을 때"를, 새 실험이면 run 이름을 바꿉니다.'
model = YOLO('yolo26s.pt')            # 맡은 모델. RT-DETR: RTDETR('rtdetr-l.pt')
model.add_callback('on_train_start', require_wandb)
model.train(
    data=f'{DS}/data.yaml',
    project='pill-object-detection', save_dir=str(RUN),   # project = W&B 프로젝트 이름, 결과는 Drive의 RUN에 바로 저장
    optimizer=..., lr0=..., epochs=..., imgsz=..., batch=..., seed=...,   # 팀 공통값
    cache=False,
)
```
- W&B에 올라가는 것과 `project`, `save_dir`를 나눠 주는 이유는 [로컬 5단계](#5-학습-실험마다)와 같습니다.
- Ultralytics는 epoch이 끝날 때마다 `weights/last.pt`를 덮어쓰고, Ultralytics 기준 val 성적이 가장 좋은 epoch을 `weights/best.pt`로 남깁니다. `save_dir`가 Drive 경로라서 두 파일 모두 Drive에 저장됩니다.
- 세션이 끊기면 아래 [세션이 끊겼을 때](#세션이-끊겼을-때-이어서-학습)를 따릅니다.
- 학습 중에 Ultralytics가 출력하는 mAP50-95로는 방법끼리 비교하지 않습니다. 비교는 6단계의 점수 3개로 합니다.

<details>
<summary>Faster R-CNN(torchvision)일 때</summary>

[노트북](notebooks/modeling/exp2_colab_smoke_test.ipynb)의 공용 구역과 Torchvision 구역을 씁니다. 위의 준비 셀과 학습 셀(Ultralytics)은 실행하지 않습니다. 꼭 지킬 것은 [로컬 5단계](#5-학습-실험마다)의 Faster R-CNN 안내와 같습니다.
- `SAVE_DIR` 검사는 노트북 공용 셀(경로 확인과 기록 함수)에 있습니다.
- Colab의 `DS`는 문자열이라 로컬 안내의 `DS.name`을 그대로 쓰면 `AttributeError`가 납니다. 노트북처럼 방법 이름(`VARIANT`)이나 `Path(DS).name`을 씁니다.
- 체크포인트는 `RUN` 아래에 직접 저장하므로 Drive에 바로 남습니다.
</details>

**After**
```diff
  내 드라이브/pill data/<cj_runs 또는 jw_runs>/
+ └── v2_yolo26s/
+     ├── weights/best.pt
+     ├── weights/last.pt       # epoch마다 갱신 → 이어서 학습할 때 씀
+     ├── wandb/                # W&B 기록. 이어서 학습하거나 점수를 붙일 때 같은 실행을 여기서 찾음
+     ├── args.yaml
+     └── results.csv
```

#### 세션이 끊겼을 때: 이어서 학습
`last.pt`에는 가중치와 함께 끝난 epoch 번호, optimizer 상태, 학습 설정(`data` 경로, 저장 폴더)이 들어 있습니다. 그래서 `resume=True`로 주면 다음 epoch부터 같은 Drive 폴더에 이어서 학습합니다.
1. 새 세션에서 2~4단계와 5단계 준비 셀을 다시 실행합니다. 4단계 `--out`은 처음과 같은 `/content/exp2/v2`로 줍니다. `last.pt`에 이 경로의 `data.yaml`이 적혀 있습니다.
2. 위의 학습 셀은 다시 실행하지 않습니다. 실행해도 `이미 있습니다` 오류로 멈춥니다.
3. 아래 셀을 실행합니다.
```python
model = YOLO(str(RUN / 'weights/last.pt'))
model.add_callback('on_train_start', require_wandb)
model.train(resume=True)
```
- W&B에도 같은 실행에 이어서 기록됩니다. Ultralytics가 `RUN/wandb/`에 남은 기록에서 실행 id를 찾습니다.
- 끊긴 epoch은 처음부터 다시 합니다. `last.pt`가 Drive에 다 올라가기 전에 끊겼다면 한 epoch 더 앞에서 이어질 수 있습니다.
- `epochs`, `lr0` 같은 설정은 처음 학습할 때 값을 그대로 씁니다. 다른 값을 넘기면 `Resume ignores ...` 경고가 나오고 무시됩니다.
- 이미 끝난 학습이면 `nothing to resume` 오류가 납니다. 6단계로 갑니다.
- Faster R-CNN은 직접 짠 학습 루프라서, epoch마다 model, optimizer, epoch 번호를 `RUN`에 저장하고 이어서 학습할 때 그 파일을 불러오는 코드를 직접 넣습니다. W&B는 `RUN/wandb/`에서 찾은 실행 id로 `wandb.init(..., id=run_id, resume="allow", dir=str(RUN))`을 해서 이어 갑니다. (`run_id`를 찾는 줄은 6단계에 있습니다)

### 6. val 채점 (실험마다)
```python
import json, shutil
from pill_detection.evaluate import evaluate_val, predictions_from_ultralytics

model = YOLO(f'{RUN}/weights/best.pt')
results = model.predict(f'{DS}/images/val', conf=0.001, stream=True, verbose=False)
scores = evaluate_val(predictions_from_ultralytics(results), f'{DS}/eval/val_gt.json')
print(f"val {scores['val']:.4f}  kaggle {scores['kaggle']:.4f}  test_only {scores['test_only']:.4f}")

with open(f'{RUN}/scores.json', 'w', encoding='utf-8') as f:
    json.dump(scores, f, indent=2)
shutil.copy(f'{DS}/dataset_info.json', RUN)   # /content는 지워지므로 Drive에 남김

# 점수와 데이터 정보를 학습 때의 W&B 실행에 붙인다
run_id = max((RUN / 'wandb').glob('*run-*/run-*.wandb'), key=lambda f: f.stat().st_mtime).stem[4:]
with wandb.init(entity='codeit-prj1-team2', project='pill-object-detection', id=run_id, resume='allow', dir=str(RUN)) as run:
    run.summary.update({f'exp2/{k}': scores[k] for k in ('val', 'kaggle', 'test_only')})
    run.config.update({'dataset_info': json.load(open(f'{DS}/dataset_info.json', encoding='utf-8'))})
```
- 새 세션이면 2~4단계와 5단계 준비 셀을 먼저 실행합니다.
- 점수 3개의 뜻과 W&B Summary에 붙는 이름은 [로컬 6단계](#6-val-채점-실험마다)와 같습니다.
- Faster R-CNN은 [노트북](notebooks/modeling/exp2_colab_smoke_test.ipynb) Torchvision 구역의 val 채점 셀을 씁니다.

**After**
```text
val 0.____  kaggle 0.____  test_only 0.____
```
```diff
  내 드라이브/pill data/<cj_runs 또는 jw_runs>/v2_yolo26s/
  ├── weights/ ...
+ ├── scores.json               # 점수 3개 + 클래스별 AP
+ └── dataset_info.json         # 어떤 목록으로 학습했는지
```

### 7. 결과 기록 (실험마다)
[로컬 7단계](#7-결과-기록-실험마다)와 같습니다. best.pt 위치는 `pill data/<cj_runs 또는 jw_runs>/v2_yolo26s/weights/best.pt`입니다. run 폴더가 처음부터 공유 드라이브에 있으므로 따로 올리지 않습니다.

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
- Faster R-CNN은 [로컬 8단계](#8-kaggle-제출)처럼 노트북 Torchvision 구역의 Kaggle 제출 셀을 씁니다.

**After**
```diff
  /content/pill-object-detection/data/raw/
+ └── test_images/                # PNG 842장
  내 드라이브/pill data/<cj_runs 또는 jw_runs>/
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
| `[오류] 목록 CSV가 없습니다` | 레포가 옛 버전 | 로컬은 `git pull origin main`, Colab은 2단계 셀 다시 실행 |
| `학습 데이터 annotation 폴더가 없습니다` | 사본이 없거나 zip을 다른 곳에 풂 | 로컬은 3단계, Colab은 레포 최상위에서 zip 다시 풀기 |
| `목록의 이미지 N장이 학습 데이터에 없습니다`, `PNG 파일이 없습니다` | 사본이 옛 것이거나 zip이 덜 풀림 | 로컬은 3단계, Colab은 zip 다시 풀기 |
| `--check`에서 `다름` | 옛 정제 목록으로 만든 사본이거나 zip이 덜 풀림 | 로컬은 3단계, Colab은 zip 다시 풀기. CSV는 커밋하지 않기 |
| `[AI Hub] ... 받을 것 N개` (N이 0이 아님) | `data/external/aihub/raw/`가 덜 풀림 | `.env`에 `AIHUB_APIKEY`를 넣고 그대로 진행하거나, 데이터 담당에게 묻기 |
| `출력 폴더가 비어 있지 않습니다` | 같은 `--out`에 이미 만듦 | Colab에서 같은 세션에 4단계를 다시 실행했으면 이미 만든 폴더이므로 그대로 진행. 다시 만들려면 그 폴더를 지우거나 다른 `--out` |
| `출력 폴더가 레포 안인데 .gitignore에 걸리지 않습니다` | 커밋에 올라갈 수 있는 곳 | `--out data/processed/exp2/<방법>` 또는 레포 밖 |
| `이미지: {'copy': ...}` | 다른 디스크라 하드링크 불가 | `--out`을 `data`와 같은 디스크로 |
| `val 정답에 없는 이미지의 예측이 N장` | val이 아닌 폴더(test 등)로 예측 | `<out>/images/val`로 예측 |
| 채점 중 `invalid literal for int()` | 모델 클래스 이름이 K-코드 숫자가 아님 | 4단계 폴더의 `data.yaml`로 학습한 모델인지 확인 |
| 이어서 학습할 때 `Dataset '/content/exp2/v2/data.yaml' error ❌ ... does not exist` | 새 세션에서 4단계를 안 했거나 다른 `--out`에 만듦 | Colab 4단계를 `--out /content/exp2/v2`로 다시 |
| `W&B 기록이 시작되지 않았습니다` | 같은 커널에서 준비 셀보다 먼저 Ultralytics로 학습이나 예측을 함 | 커널(런타임)을 다시 시작하고 5단계 준비 셀부터. Colab은 2단계부터 |
| `W&B 기록이 팀이 아닌 ...에 시작됐습니다` | `WANDB_ENTITY`를 정하기 전에 wandb가 시작됨. Colab은 2단계 셀에서 `wandb.login()`보다 뒤에 정한 경우 | 커널(런타임)을 다시 시작하고 5단계 준비 셀부터. Colab은 2단계부터 |
| `SAVE_DIR의 상위 폴더가 Drive에 없습니다` | 2단계 `SAVE_DIR`에 `<...>`가 남았거나, `pill data` 바로가기가 다른 위치에 있음 | 2단계 셀의 `ZIP_DIR`, `SAVE_DIR`를 내 Drive 경로로 고치고 2단계부터 다시 |
| `AssertionError: ...이 이미 있습니다` | 같은 run 이름으로 학습 셀을 다시 실행함 | 이어서 학습이면 resume 셀, 새 실험이면 run 이름 바꾸기 |

## 하지 말 것
- `data/datasets/images.csv`, `classes.csv`를 손으로 고치지 않습니다. 고치면 val이 바뀌어 다른 팀원의 실험과 비교할 수 없습니다.
- `--check`가 "다름"인 상태에서 `make_dataset_lists.py`를 `--check` 없이 돌리거나, 바뀐 CSV를 커밋하지 않습니다.
- 학습 폴더, best.pt, 제출 CSV를 `git add -f`로 올리지 않습니다.
