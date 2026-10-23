# 🧪 실험 규칙
> 이 문서는 **실험을 어떻게 돌리고 기록하는지, 공용 코드를 어떻게 맞춰 쓰는지** 정합니다. git·PR·커밋 절차는 [CONTRIBUTING.md](CONTRIBUTING.md)를 따릅니다.
>
> 규칙을 바꾸고 싶으면 `docs/` 브랜치에서 이 문서를 고쳐 PR을 올리고 팀원 동의를 받습니다.

# 📊 원본 데이터 세팅 및 학습
## 1. 원본 데이터 준비
1. **로컬**
    1. 아래 위치에 원본 데이터가 준비되도록 합니다.  
        ```text
        data/                      
           └── raw/                
                ├── train_images/
                ├── train_annotations/
                └── test_images/
        ```

2. **Drive에 준비**
    1. `train_images`, `train_annotations`, `test_images`를 `raw.zip`으로 묶어서 본인 개인 드라이브에 원하는 위치에 업로드합니다. 이 경로는 추후 .env에 넣어야 하니 기억해두세요. 

## 2. 전처리된 데이터 준비
아래 스크립트는 전처리 규칙과 data split 규칙대로 전처리와 split을 진행합니다. 원본 이미지는 수정/삭제되지 않습니다.
```bash
uv run python scripts/prepare_data.py                                         # 자동으로 가장 높은 버전의 split을 사용
uv run python scripts/prepare_data.py --split configs/splits/split_v{n}.json  # 원하는 split 버전을 지정 가능
```
이 스크립트를 실행하면 다음 파일들이 갱신/생성됩니다.
- `data/processed/images/{train, val}`
- `data/processed/labels/{train, val}`
- `configs/data.yaml`

**주의** `prepare_data.py`는 버전 번호(n)이 가장 높은 것을 디폴트로 사용합니다. 특정 버전의 split 파일을 사용해서 실험하고싶다면 `--split` 인자를 넘겨주세요. 

**데이터 전처리에 영향을 주는 코드**:
- `configs/data_fixes.yaml`
    - 데이터 전처리 규칙(학습에서 뺄 이미지 및 라벨, bbox를 고칠 라벨, 새로 추가할 라벨)을 명시합니다.
    - **사람이 직접 수정하는 파일입니다.**
- `configs/splits/split_v{n}.json` 
    - train/val 분할 규칙이 나열되어있습니다. 
    - **사람이 직접 수정하는 파일이 아닙니다.**   
- `src/pill_detection/convert.py → make_group_split()`
    - `configs/splits/split_v{n}.json` 을 생성하는 함수입니다.
    - 데이터 분할 전략이 수정되었다면 이 코드를 고치면 됩니다.
- `scripts/make_split.py`
    - `make_group_split()`을 호출하는 wrapper script입니다.
    - `make_group_split()`을 수정했다면 이 스크립트를 재실행해주세요.

## 3. 환경변수 준비
1. **.env 파일 생성**
    ```bash
    cp .env.example .env        # 최초 1회. 
    ``` 
2. **.env 값 채우기**
    - WandB 값, raw.zip이 위치한 본인 구글드라이브 폴더 등 설정정보를 저장합니다.

3. **개인 Drive에 올리기**
    - `.env`를 내 드라이브에 `MyDrive/.secret/pill-object-detection.env` 이름으로 업로드하세요.
    - 반드시 드라이브에 이 지정된 경로에 업로드 해야 Colab 원격 세션에서 접근할 수 있습니다. 

4. **로컬 `.env`를 고치면 Drive의 파일도 다시 올려야 합니다.**

- **참고**
    | 실험 환경 | `.env` 값을 두는 곳 |
    | -- | -- |
    | 로컬 터미널, 로컬 노트북 | 레포 루트 `.env` |
    | 브라우저 Colab, VS Code + Colab 확장 | `.env`를 내 드라이브 `MyDrive/.secret/pill-object-detection.env` 이름으로 업로드. 공유 폴더에 두지 않기 |
    | 브라우저 Kaggle, VS Code + Kaggle Jupyter Server | Add-ons → Secrets에 같은 이름으로 `WANDB_API_KEY`, `WANDB_ENTITY` 추가 (노트북에 attach) |

## 4. 실험 설정 준비
1. main을 최신화한 뒤 실험 브랜치를 만듭니다.
    ```bash
    git switch main
    git pull origin main
    git switch -c exp/yolov8s-mosaic # 예시 브랜치
    ```

2. 비슷한 기존 YAML을 복사해 새 이름으로 저장하고, 바꿀 값만 고칩니다.
    ```bash
    cp configs/exp/yolov8s_640.yaml configs/exp/yolov8s_640_mosaic.yaml
    ```
    예시:
    ```yaml
    # configs/exp/yolov8s_640.yaml
    model: yolov8s.pt
    data: configs/data.yaml
    imgsz: 640
    epochs: 100
    batch: 16
    seed: 42
    ```

    > 쓸 수 있는 키는 [Ultralytics 학습 설정 문서](https://docs.ultralytics.com/modes/train/#train-settings)를 참고하세요.

**지킬 것**
- 한 실험에서는 한 가지만 바꿉니다. 여러 개를 한꺼번에 바꾸면 무엇 때문에 성능이 달라졌는지 알 수 없습니다.
- YAML에는 project, name을 쓰지 않습니다. 실험 이름은 파일 이름으로 정해집니다.
- 실험하다 공용 코드(src/, scripts/)를 고쳐야 하면 실험 브랜치에서 고치지 말고, feat/ 브랜치로 따로 PR을 올립니다.


## 5. 로컬 GPU로 학습할 때

1. **데이터 전처리**
    ```bash
    uv run python scripts/prepare_data.py   # data/raw → data/processed
    ```
2. **스모크 테스트**
    ```bash
    WANDB_MODE=offline uv run python scripts/train.py configs/smoke.yaml    # 동작 확인 (W&B에 올리지 않음)
    ```
3. **본 실험**
    ```bash
    uv run python scripts/train.py configs/exp/yolov8s_640.yaml     # 실험
    ```
4. **제출 파일 생성**
    ```bash
    uv run python scripts/make_submission.py --weights runs/yolov8s_640/weights/best.pt
    ```

## 6. Colab/Kaggle GPU로 학습할 때
> 원격 세션에서는 `notebooks/00-train-colab-kaggle.ipynb`으로 진행합니다.
1. **환경변수 설정**
    - 원격 세션이 읽을 수 있는 곳에 `.env` 값을 둡니다. Colab, Kaggle에서는 로컬 파일시스템에 접근할 수 없어 로컬 `.env`를 읽을 수 없습니다.
    - **Colab**: `.env`를 내 드라이브에 `MyDrive/.secret/pill-object-detection.env` 이름으로 업로드하세요. 꼭 이 안에 있는 값들이 채워져있도록 하세요.
    - **Kaggle**: Kaggle은 구글드라이브를 읽을 수 없습니다. Add-ons → Secrets에 `.env`와 같은 이름으로 `WANDB_API_KEY`, `WANDB_ENTITY`를 추가하고 노트북에 attach하세요. 원본 데이터는 Add Input으로 팀 데이터셋을 붙이므로 `DRIVE_RAW`는 필요 없습니다.
2. **실험 설정**
    - `notebooks/00-train-colab-kaggle.ipynb` 노트북 첫번째 코드 셀을 적절하게 채워줍니다.
        ```python
        BRANCH = ""                  # 실험 YAML을 push한 브랜치. 예) "exp/yolov8s-mosaic"
        EXP_YAML = ""                # 예) "configs/exp/yolov8s_640.yaml" 
        ```
3. **실험 진행**
    - `Run All`을 누르면 스모크 테스트와 본 실험이 자동으로 실행됩니다.
4. **결과 확인**
    - W&B 프로젝트에서 방금 진행한 실험을 확인할 수 있습니다. 
5. **제출 파일 생성**
    > TODO : 코랩세션에서 진행한 모델로 결과파일 생성하는 법



# 🔒 실험 규칙

## 규칙 1. 실험 1개 = YAML 1개 & `exp/` 브랜치 1개 & W&B run 1개
학습 코드는 고치지 않고, `configs/exp/`에 YAML 파일을 추가하는 것으로 실험합니다. **YAML 파일 이름이 곧 실험 이름**이고, 결과 폴더(`runs/<실험 이름>/`)와 W&B run 이름도 이 이름을 따릅니다.

### 실험 순서
1. **main을 최신화한 뒤 실험 브랜치를 만듭니다.**
    ```bash
    git switch -c exp/yolov8s-mosaic # 예시 브랜치 
    ```
2. **비슷한 기존 YAML을 복사해 새 이름으로 저장하고, 바꿀 값만 고칩니다.**
    ```bash
    cp configs/exp/yolov8s_640.yaml configs/exp/yolov8s_640_mosaic.yaml
    ```
3. **학습합니다.** 
    - 자세한 학습 방식은 바로 위 `데이터 세팅 및 학습` 절을 참고해주세요. 
    
4. **YAML을 `exp:` 타입으로 커밋합니다.** 
    - 예) `exp: yolov8s img640 mosaic=1.0`
5. **PR을 날립니다.**
    - Description에 W&B run 링크와 주요 지표(mAP)를 적습니다.

### YAML 예시
```yaml
# configs/exp/yolov8s_640.yaml
model: yolov8s.pt
data: configs/data.yaml
imgsz: 640
epochs: 100
batch: 16
seed: 42
```
`model`을 뺀 나머지 키는 Ultralytics 학습 설정 그대로입니다. 쓸 수 있는 키는 [Ultralytics 학습 설정 문서](https://docs.ultralytics.com/modes/train/#train-settings)를 참고합니다.

### 지킬 것
- 한 실험에서는 한 가지만 바꿉니다. 여러 개를 한꺼번에 바꾸면 무엇 때문에 성능이 달라졌는지 알 수 없습니다.
- YAML에는 `project`, `name`을 쓰지 않습니다. 실험 이름은 파일 이름으로 정해집니다.
- 실험하다 공용 코드(`src/`, `scripts/`)를 고쳐야 하면 실험 브랜치에서 고치지 말고, `feat/` 브랜치로 따로 PR을 올립니다.

### `scripts/train.py` 세부사항
Ultralytics 8.4 소스 기준으로 확인한 내용입니다.
- YAML의 `model`로 `YOLO(model)`을 만들고, `train(cfg=<YAML 경로>)`로 넘기면 나머지 키가 학습 설정으로 들어갑니다.
- YAML에 `data`가 없으면 Ultralytics는 예제 데이터셋 `coco8.yaml`로 학습합니다. 그래서 `train.py`는 `data`를 `train()`에 직접 넘기고, YAML에 없으면 `configs/data.yaml`을 씁니다.
- RT-DETR 가중치(`rtdetr-l.pt`)도 `YOLO()`가 알아서 RT-DETR 모델로 바꿔 줍니다. 모델 종류별로 분기할 필요가 없습니다.
- Ultralytics는 `project` 값을 W&B project 이름으로 씁니다. `project="pill-object-detection"`으로 고정하고, 결과 폴더는 `save_dir=runs/<YAML 파일 이름>`으로 따로 지정합니다.
- W&B 로깅은 `ultralytics.settings.update({"wandb": True})`로 켭니다. 이 설정은 그 컴퓨터의 Ultralytics 설정 파일에 저장됩니다.
- Ultralytics는 best.pt를 mAP50-95(`fitness`) 기준으로 고릅니다. `train.py`가 Ultralytics의 `DetMetrics`를 바꿔 기준을 대회 지표 mAP75-95로 바꾸고, 같은 값을 `metrics/mAP75-95(B)`로 results.csv와 W&B에 기록합니다. 조기 종료(`patience`)도 이 값을 따릅니다.

## 규칙 2. 클래스 매핑은 `src/pill_detection/data/classes.py` 한 곳에서만 정의
YOLO는 클래스를 0부터 N-1까지의 index로 다루지만, 원본 데이터와 제출 파일은 `category_id`를 씁니다. 둘이 섞이면 학습 지표는 멀쩡한데 제출 점수만 0에 가깝게 나오는 사고가 납니다.

- EDA가 완료된 후 `classes.py`에 `(category_id, 약 이름)` 목록을 한 번만 적고, **목록 순서를 YOLO index로 씁니다.**
- 변환은 항상 `classes.py`의 함수로 합니다. 다른 파일에 매핑을 복사하거나 번호를 직접 적지 않습니다.

    | 변환 | 쓰는 곳 |
    | -- | -- |
    | YOLO index → category_id | 제출 파일, `predict()` 결과 |
    | YOLO index → 약 이름 | 데모 |
    | category_id → YOLO index | 라벨 변환 (`convert.py`) |

- `configs/data.yaml`의 `names`도 `scripts/prepare_data.py`가 `classes.py`로 만듭니다. data.yaml을 손으로 고치지 않습니다.
- 학습을 시작한 뒤에는 목록 순서를 바꾸지 않습니다. 순서가 바뀌면 이미 학습한 가중치의 클래스 번호가 전부 어긋납니다.

## 규칙 3. `predict()` 반환 형식은 팀 합의 사항
데모(`demo/app.py`)와 제출(`scripts/make_submission.py`)은 `src/pill_detection/predict.py`의 `predict()`만 써서 검출합니다. 같은 함수를 쓰므로 데모 결과와 제출 결과가 항상 같습니다.

> 아래 형식은 초안입니다. EDA 후 회의에서 확정합니다. 

```python
predict(image, weights=None) -> list[dict]

# 리스트의 원소 하나 = 검출된 알약 하나
{
    "category_id": 1234,      # 원본 데이터의 category_id (YOLO index 아님)
    "name": "약 이름",
    "bbox": [x, y, w, h],     # 좌상단 x, y와 너비, 높이. 원본 이미지 픽셀 단위
    "score": 0.93,            # 0~1 신뢰도
}
```

- `weights`를 비우면 가짜 결과를 돌려주는 mock 모드로 동작하게 만듭니다. 그러면 데모 담당은 모델이 나오기 전부터 mock 결과로 페이지를 만들 수 있고, 모델이 나오면 가중치 경로만 넣으면 됩니다.
- 확정한 뒤 형식을 바꾸려면 팀 전체와 먼저 상의합니다. 데모와 제출 코드를 함께 고쳐야 합니다.

## 규칙 4. 이미 있는 split은 고치지 않고 다음 번호를 붙여서 새 파일로 만든다
`prepare_data.py`는 `configs/splits/`에서 번호가 가장 큰 split 파일을 기준으로 사용합니다. 이미 있는 split 파일은 이전 실험을 재현할 때 필요하므로 고치거나 지우지 않습니다.

### split을 바꾸는 순서
1. `feat/` 브랜치를 만듭니다. 나누는 규칙을 바꾼다면 `src/pill_detection/data/convert.py`의 `make_group_split()`을 고칩니다.
2. 로컬에서 새 split을 만듭니다. 파일 이름은 다음 번호(`split_v2.json`, `split_v3.json`, ...)로 자동으로 정해집니다.
    ```bash
    uv run python scripts/make_split.py                            # default: val 20%, seed 42
    uv run python scripts/make_split.py --val-ratio 0.15 --seed 7  # 비율이나 seed를 바꿀 때
    ```
3. 만들어진 split 파일을 커밋하고 PR을 올립니다. 머지되면 팀원 모두 main을 pull한 뒤 각자 로컬에서 `prepare_data.py`를 다시 실행합니다.

### 지킬 것
- mAP는 같은 split끼리만 비교합니다. val이 다르면 점수를 비교할 수 없으므로, split을 바꾸면 기준 모델(baseline)부터 다시 학습합니다. 어떤 split으로 학습했는지는 W&B의 `data_version`에 남습니다.
- 두 사람이 같은 번호를 만들면 머지할 때 충돌이 납니다. 나중에 머지하는 사람이 main을 pull하고, 자기 split 파일을 지운 뒤 `make_split.py`를 다시 실행합니다.
- 이전 split으로 다시 학습할 때만 직접 지정합니다.
    ```bash
    uv run python scripts/prepare_data.py --split configs/splits/split_v1.json
    ```
