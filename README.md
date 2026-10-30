# 💊 Pill Object Detection

## 📑 Project Overview
이 프로젝트는 경구약제 이미지 인식 기능을 이용해 사용자가 알약 사진을 올리면 그 알약이 무슨 알약인 지 검출해주는 서비스입니다. 

- 1차 마일스톤: 알약 이미지를 입력하면 약 이름을 검출하는 모델 개발
- (선택)2차 마일스톤: 알약 정보를 입력하면 [식의약 데이터 포털 공공API](https://www.data.go.kr/data/15095677/openapi.do)를 통해 복용 가이드를 제공해주는 기능

> TODO 데모 영상

## 📊 DataSet Overview
본 프로젝트는 [AI Hub에서 제공하는 공개 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 활용했습니다. 

> TODO EDA 후 데이터 개요 시각화 자료 첨부

## 📂 Project Structure

> 레포지토리 구조는 확정되지 않았습니다. EDA와 모델링을 거치면서 구조가 수정될 수 있습니다.

```text
pill-object-detection/
├── data/                      # raw·processed·external은 폴더만 git으로 관리하고 내용은 추적하지 않음
│   ├── raw/                   # Kaggle 원본 데이터. 내려받은 그대로 두고 수정하지 않음
│   ├── external/aihub/        # AI Hub 외부 데이터. 내려받은 그대로 두고 수정하지 않음
│   │   ├── raw/               # zip을 풀어 둔 원본. raw/와 같은 구조 (train_annotations, train_images)
│   │   └── sources.csv        # 파일마다 나온 zip 이름
│   ├── processed/             # 가공 결과. 학습 데이터 사본, 2차 실험 방법별 학습 폴더(exp2/)
│   │   ├── train_annotations/ # Kaggle 정제 사본 (이미지는 data/raw/train_images를 씀)
│   │   └── ai_hub/            # AI Hub 정제 사본. train_annotations, train_images(원본 PNG의 하드링크)
│   ├── datasets/              # 2차 실험 목록 CSV(images.csv, classes.csv)와 설명. git으로 관리함
│   └── label_fix/             # 정제 목록(label_fixes.csv)과 변경 기록 CSV. git으로 관리함
├── notebooks/                 # 개인 탐색용 노트북
│   ├── eda/                   # 개인 EDA 노트북
│   └── modeling/              # 각자 맡은 모델링 노트북, 2차 실험 Colab 스모크 테스트
├── src/pill_detection/        # 공용 코드 패키지 (노트북·스크립트·데모 어디서든 import)
│   ├── data.py                # 데이터 읽기, 정제 목록대로 학습 데이터 사본 만들기
│   ├── datasets.py            # 2차 실험 목록 CSV와 방법별 학습 폴더(YOLO, COCO) 만들기
│   ├── evaluate.py            # 2차 val 채점(mAP@[0.75:0.95])과 Kaggle 제출 CSV 변환
│   └── utils.py               # 공용 보일러 플레이트 코드.(bbox 변환, 시각화 등) 
├── scripts/                   # 터미널에서 실행하는 스크립트 (uv run scripts/<이름>.py)
│   ├── prepare_train_data.py  # 학습 데이터 준비: AI Hub 받기·풀기, 정제, 변경 기록을 한 번에
│   ├── download_aihub.py      # AI Hub에서 filekey로 zip 받기 (대회 금지 파일은 받지 않음)
│   ├── extract_aihub.py       # 받은 AI Hub zip을 풀고 지우기
│   ├── make_fixed_annotations.py  # AI Hub 없이 Kaggle 사본만 다시 만들기
│   ├── make_dataset_lists.py  # 2차 실험 목록 CSV 만들기. 팀원은 --check로 내 데이터가 같은지만 확인
│   └── build_dataset.py       # 목록 CSV대로 방법(v1~v4)별 학습 폴더 만들기
├── tests/                     # 공용 코드 테스트 (uv run pytest)
├── runs/                      # 학습 결과
├── submissions/               # 제출 CSV
├── hooks/                     # pre-commit hook 스크립트
├── EXPERIMENTS.md             # 2차 실험 가이드: 학습 폴더, 학습, val 채점, 제출
└── CONTRIBUTING.md            # git·PR·커밋 협업 규칙
```

## 🧑‍💻 Dependencies
이 프로젝트의 환경은 uv로 관리됩니다. 

1. **uv 설치(최초 1회)**
    - macOS / Linux
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
        ```
    - Windows PowerShell
        ```powershell
        powershell ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

2. **uv 설치 확인**
    ```bash
    uv --version
    ```

3. **레포를 받은 뒤 환경 생성**
    ```bash
    git pull origin main
    rm -rf .venv        # 기존 .venv(예: 3.14)가 있다면 삭제
    uv sync             # Python 3.12가 없으면 uv가 자동으로 받아서 설치함  
    uv run pre-commit install   # 최초 1회: 커밋 전 검사 hook 설치 (uv.lock 검사, nbstripout 설정)
    ```

4. **의존성 추가 시**
    ```bash
    uv add <패키지>
    ```
    | **NOTE**: `pip install`로 설치하지 않습니다. `uv.lock`에 기록되지 않아 다른 팀원 간 환경에 drift 발생함. 

5. **NOTE**
    1. `pyproject.toml`과 `uv.lock`은 항상 같이 커밋합니다.
    2. main을 pull한 뒤에는 uv sync를 실행합니다. 다른 팀원이 추가한 패키지가 반영됩니다.
    3. uv.lock에서 머지 충돌이 나면 직접 고치지 마세요. pyproject.toml에 충돌이 있으면 먼저 해결하고, 작업 브랜치에 그대로 있는 상태에서 아래 순서로 lock 파일을 다시 만듭니다. 충돌 표시가 남은 uv.lock으로는 `uv lock`이 실패하므로 main 버전으로 먼저 덮어씁니다.
        ```bash
        git restore --source=main --staged --worktree uv.lock   # 브랜치 이동 없이 main의 uv.lock만 가져옴
        uv lock                                                 # 합쳐진 pyproject.toml 기준으로 다시 생성
        git add pyproject.toml uv.lock
        git commit --no-edit                                    # 작업 브랜치에 머지 커밋 생성
        ```
    4. `pyproject.toml`이나 `uv.lock`을 커밋하면 pre-commit hook이 `uv lock --check`로 두 파일이 일치하는지 검사합니다. 실패하면 터미널에 원인과 해결 명령이 출력되니 그대로 따라 하면 됩니다. `--no-verify`로 우회하지 마세요.
    5. pre-commit hook을 추가할 때는 스크립트를 `hooks/`에 만들고 `.pre-commit-config.yaml`에 등록합니다.
    6. 노트북을 처음 커밋하면 pre-commit hook이 이 저장소에 출력 제거 필터(nbstripout)를 설정합니다. 따로 할 일은 없습니다.
        - 노트북을 `git add`하거나 `git status`를 실행할 때 `external filter` 오류가 나면 `uv sync`를 실행합니다. `.venv`를 지우고 다시 만들지 않은 경우에 납니다.
        - Colab의 "GitHub에 사본 저장"으로 커밋하지 않습니다. 필터를 거치지 않아 출력이 그대로 올라갑니다.

## 📦 학습 데이터 준비
학습 데이터는 Kaggle 원본(`data/raw/`)과 [AI Hub 경구약제 이미지 데이터(데이터셋 번호: 576)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 함께 씁니다. 대회 규칙상 `TL_2_조합.zip`, `TS_2_조합.zip`은 쓰지 않습니다. ([RULE.md](RULE.md))

1. **준비(최초 1회)**
    - Kaggle 대회 데이터를 `data/raw/`(`train_annotations/`, `train_images/`)에 풉니다.
    - AI Hub에서 경구약제 이미지 데이터 이용 승인을 받고 API 키를 발급받습니다. `.env.example`을 복사해 `.env`를 만들고 `AIHUB_APIKEY`에 키를 넣습니다.
    - 디스크가 약 35GB 필요합니다. AI Hub 원본이 약 28GB이고, 받는 동안 zip 하나(약 3GB)의 2배와 여유 2GB가 더 필요합니다.

2. **명령 하나로 준비**
    ```bash
    uv run scripts/prepare_train_data.py
    ```
    - AI Hub zip 16개(허용 조합 라벨 8개, 이미지 8개)를 하나씩 받고, `data/external/aihub/raw/`에 풀고, zip을 지웁니다. 이미 다 푼 zip은 건너뜁니다.
    - 정제 목록 [data/label_fix/label_fixes.csv](data/label_fix/label_fixes.csv)대로 고친 사본을 만듭니다. 몇 분 걸립니다.
        - Kaggle: `data/processed/train_annotations/` (이미지는 `data/raw/train_images/`)
        - AI Hub: `data/processed/ai_hub/train_annotations/`, `train_images/`.
    - 두 데이터의 변경 내역을 `data/label_fix/apply_label_fixes_v{n}.csv` 하나에 남깁니다. 같은 정제 목록으로 다시 실행하면 새 파일을 만들지 않습니다.
        - 기록은 원본 대비 누적입니다. 최신 파일 하나에 지금까지의 수정 전체가 있습니다.
        - 고칠 것이 생기면 정제 목록에 줄을 더하고 다시 실행하세요. 기록은 다음 번호(`_v3`, `_v4` …)로 쌓입니다.
    - 중간에 멈추면 같은 명령을 다시 실행하면 됩니다.
    - 성공하면 아래처럼 요약이 출력됩니다.
        ```text
        Kaggle: JSON 763개 → 771개 (data/processed/train_annotations)
        AI Hub: 이미지 11,974장, JSON 45,819개, 이름을 고친 이미지 13장 (data/processed/ai_hub). PNG 하드링크 11,974장, 복사 0장
        변경 기록: data/label_fix/apply_label_fixes_v2.csv aihub add 70, aihub delete 161, aihub modify 968, kaggle add 8, kaggle modify 13
        ```

3. **테스트**
    ```bash
    uv run pytest
    ```
    - 기존 Kaggle 라벨 수정 결과와 EDA 노트북의 정제 결론(이미지 수, 형식, 라벨 누락·좌표 오류 없음, 이름 변경)이 그대로 반영되는 지 확인합니다.
    - 실제 데이터가 없는 테스트는 이유를 적고 건너뜁니다. 전체 실행은 약 3분 걸립니다.

## 🛠️ Scripts
프로젝트 셋업에 필요한 명령을 쉽게 제공하기 위해 `scripts/`의 스크립트를 만들었습니다. Python을 따로 켜지 않고 터미널에서 명령 한 줄로 실행하면 됩니다. 
> 위 Dependencies의 `uv sync`를 먼저 마친 상태여야 합니다.

- **레포 최상위**(`pill-object-detection/`)에서 `uv run scripts/<이름>.py`로 실행합니다. 가상환경을 켜지 않아도 됩니다.
- 경로 인자는 지금 터미널이 있는 폴더 기준입니다. 레포 최상위에서는 `data/...`로 시작하는 경로를 넣습니다. 
- 인자 설명은 `--help`로 봅니다.
- 아래 예시의 줄 끝 `\`는 macOS/Linux 터미널에서 명령을 여러 줄로 나눠 쓰는 표시입니다. Windows PowerShell에서는 `\`를 `` ` ``(백틱)으로 바꾸거나 한 줄로 이어 씁니다.

### 1. prepare_train_data.py: 학습 데이터 준비
AI Hub를 받고 풀고, [src/pill_detection/data.py](src/pill_detection/data.py)의 `make_train_dataset()`을 불러 사본과 변경 기록을 만듭니다. 준비물과 만들어지는 결과는 위 [학습 데이터 준비](#-학습-데이터-준비)에 있습니다.
- 인자는 없습니다. (`--help`만 있습니다) 읽고 쓰는 경로는 아래 표처럼 정해져 있습니다.
- AI Hub 받기와 풀기만 따로 할 때는 `download_aihub.py`(filekey로 zip 받기), `extract_aihub.py`(zip 풀고 지우기)를 씁니다. 사용법은 `--help`와 각 파일 맨 위 설명에 있습니다.

| 경로 | 하는 일 |
| --- | --- |
| `data/raw/train_annotations`, `data/raw/train_images` | Kaggle 원본. 읽기만 함. 없으면 멈춤 |
| `.env`의 `AIHUB_APIKEY` | AI Hub API 키. 받을 zip이 있을 때만 읽음 |
| `data/external/aihub/raw/` | AI Hub 원본. zip을 받아 풀고 지움. 이미 다 푼 zip은 건너뜀 |
| `data/external/aihub/sources.csv` | zip마다 나온 파일 목록. 다 푼 zip인지 판단할 때 씀 |
| `data/label_fix/label_fixes.csv` | 정제 목록. 읽기만 함 |
| `data/processed/train_annotations/` | Kaggle 사본. **폴더 전체를 지우고 다시 만듦** |
| `data/processed/ai_hub/` | AI Hub 사본. **`train_annotations/`를 지우고 다시 만들고, `train_images/`의 PNG도 다시 만듦** |
| `data/label_fix/apply_label_fixes_v{n}.csv` | 변경 기록. 내용이 바뀌면 다음 번호로 새로 저장하고, 이전 파일은 그대로 둠 |

- **예시 1. 처음 준비하기**
    ```bash
    uv run scripts/prepare_train_data.py
    ```
    - zip 16개를 하나씩 받고 → 풀고 → 지우는 출력이 이어진 뒤, 위 [학습 데이터 준비](#-학습-데이터-준비)의 요약이 나옵니다.
    - zip 하나가 약 3GB라서 네트워크에 따라 오래 걸립니다.

- **예시 2. 정제 목록을 고친 뒤 다시 만들기**
    - [data/label_fix/label_fixes.csv](data/label_fix/label_fixes.csv)에 줄을 더하거나 고친 뒤, 같은 명령을 실행합니다.
    - 받기와 풀기는 모두 건너뛰고 정제만 다시 합니다. 첫 줄이 아래처럼 나옵니다.
        ```text
        [AI Hub] zip 16개 중 다 푼 것 16개, 받을 것 0개
        ```
    - 마지막 줄의 변경 기록 파일이 다음 번호(`apply_label_fixes_v3.csv` …)로 바뀝니다. 정제 목록이 그대로면 최신 파일 이름이 그대로 나옵니다.

- **예시 3. 중간에 멈췄을 때**
    - `Ctrl+C`나 네트워크 오류로 멈췄으면 같은 명령을 다시 실행합니다.
    - 다 푼 zip은 건너뜁니다. 받다 멈춘 zip은 처음부터 다시 받고, 풀다 멈춘 zip은 이미 푼 파일을 건너뛰고 이어서 풉니다.
    - 정제 단계에서 멈췄으면 정제를 처음부터 다시 합니다. 원본은 건드리지 않습니다.

### 2. make_fixed_annotations.py: Deprecated
> 이 스크립트는 사용하지 말아주세요. prepare_train_data.py를 사용해주세요. 동작은 똑같되, 더 안정적입니다. 

[src/pill_detection/data.py](src/pill_detection/data.py)의 `make_fixed_annotations()`를 부르기만 하며, 인자의 자세한 뜻과 규칙은 그 함수의 docstring에 있습니다.
- `prepare_train_data.py`가 `make_fixed_annotations.py`의 기능을 똑같이 수행합니다. 단, `make_fixed_annotations.py` 스크립트는 Kaggle 출처 데이터만 수정하고자 할 때 사용할 수 있습니다. 
- 변경 기록은 `prepare_train_data.py`와 같은 파일(`data/label_fix/apply_label_fixes_v{n}.csv`)에 쌓입니다. 
- `source = aihub` 줄은 최신 기록에서 그대로 이어받습니다. 
- 정제 목록의 `source = kaggle` 줄이 그대로면 새 파일을 만들지 않습니다.

| 인자 | 하는 일 | 예시 |
| --- | --- | --- |
| `--fix-csv` | 정제 목록 CSV. 읽기만 함. kaggle 줄만 씀 | `data/label_fix/label_fixes.csv` |
| `--base-ann-dir` | 기준 annotation 폴더. 읽기만 함. 원본을 넣음 | `data/raw/train_annotations` |
| `--img-dir` | 이미지 폴더. 읽기만 함 | `data/raw/train_images` |
| `--out-ann-dir` | 사본 annotation 폴더. **폴더 전체를 지우고 다시 만듦** | `data/processed/train_annotations` |
| `--change-csv` | (생략 가능) 변경 기록 묶음의 1번 파일. 내용이 바뀌면 `_v2`, `_v3` …로 새로 저장. 보통 넣지 않음 | 기본값 `data/label_fix/apply_label_fixes.csv` |

- **예시. 정제 목록의 kaggle 줄을 고친 뒤 Kaggle 사본만 다시 만들기**
    ```bash
    uv run scripts/make_fixed_annotations.py \
        --fix-csv data/label_fix/label_fixes.csv \
        --base-ann-dir data/raw/train_annotations \
        --img-dir data/raw/train_images \
        --out-ann-dir data/processed/train_annotations
    ```
    성공하면 아래처럼 요약이 출력됩니다. (정제 목록이 그대로일 때의 예)
    ```text
    기준 JSON 763개 → 사본 JSON 771개 (data/processed/train_annotations)
    수정 목록 21줄 적용: add 8, modify 13
    변경 기록: data/label_fix/apply_label_fixes_v2.csv (kaggle 21건, 이어받은 aihub 1,199건)
    ```

- **NOTE**
    1. 같은 수정 목록으로 다시 실행해도 결과는 같습니다. 사본 폴더는 다시 만들어지고, 변경 기록은 이전 기록과 내용이 같으면 새 파일을 만들지 않고 이전 파일 경로를 보여줍니다.
    2. 수정 목록이나 경로에 문제가 있으면 `[오류]` 뒤에 원인과 해결 방법이 나오고 멈춥니다. 메시지에 적힌 대로 고친 뒤 다시 실행하세요.

### 3. build_dataset.py: 방법별 학습 폴더 만들기
[data/datasets/](data/datasets/README.md)의 목록 CSV대로 2차 실험 방법(v1~v4) 하나의 학습 폴더를 만듭니다. 학습, 채점, 제출 순서는 [EXPERIMENTS.md](EXPERIMENTS.md)에 있습니다.

| 인자 | 하는 일 | 예시 |
| --- | --- | --- |
| `--variant` | 방법. v1(전부), v2(클래스별 최소 200), v3(대회 클래스가 든 사진), v4(대회 클래스 밖 → others) | `v2` |
| `--out` | 만들 폴더. 없거나 비어 있어야 함. 레포 안이면 `data/processed/` 아래 | `data/processed/exp2/v2` |
| `--format` | (생략 가능) 라벨 형식 `yolo`, `coco`, `both` | 기본값 `both` |

```bash
uv run scripts/build_dataset.py --variant v2 --out data/processed/exp2/v2
```
`<out>/data.yaml`(YOLO), `<out>/coco/`(COCO), `<out>/eval/val_gt.json`(채점 정답)이 생깁니다. 이미지는 원본에 하드링크합니다. 나머지 인자(`--link`, `--data-dir`)는 `--help`를 봅니다.

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표