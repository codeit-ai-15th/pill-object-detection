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
│   ├── processed/             # 가공 결과. 학습 데이터 사본, YOLO 포맷 변환 결과
│   │   ├── train_annotations/ # Kaggle 정제 사본 (이미지는 data/raw/train_images를 씀)
│   │   └── ai_hub/            # AI Hub 정제 사본. train_annotations, train_images(원본 PNG의 하드링크)
│   └── label_fix/             # 정제 목록(label_fixes.csv)과 변경 기록 CSV. git으로 관리함
├── notebooks/                 # 개인 탐색용 노트북
│   ├── eda/                   # 개인 EDA 노트북
│   └── modeling/              # 각자 맡은 모델링 노트북
├── src/pill_detection/        # 공용 코드 패키지 (노트북·스크립트·데모 어디서든 import)
│   ├── data.py                # 데이터 읽기, 정제 목록대로 학습 데이터 사본 만들기
│   └── utils.py               # 공용 보일러 플레이트 코드.(bbox 변환, 시각화 등) 
├── scripts/                   # 터미널에서 실행하는 스크립트 (uv run scripts/<이름>.py)
│   ├── prepare_train_data.py  # 학습 데이터 준비: AI Hub 받기·풀기, 정제, 변경 기록을 한 번에
│   ├── download_aihub.py      # AI Hub에서 filekey로 zip 받기 (대회 금지 파일은 받지 않음)
│   ├── extract_aihub.py       # 받은 AI Hub zip을 풀고 지우기
│   └── make_fixed_annotations.py  # Kaggle만 2차 가공할 때 쓰는 사본 만들기
├── tests/                     # 공용 코드 테스트 (uv run pytest)
├── runs/                      # 학습 결과
├── submissions/               # 제출 CSV
├── hooks/                     # pre-commit hook 스크립트
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
학습 데이터는 Kaggle 원본(`data/raw/`)과 [AI Hub 576 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 함께 씁니다. 대회 규칙상 `TL_2_조합.zip`, `TS_2_조합.zip`은 쓰지 않습니다. ([RULE.md](RULE.md))

1. **준비(최초 1회)**
    - Kaggle 대회 데이터를 `data/raw/`(`train_annotations/`, `train_images/`)에 풉니다.
    - AI Hub에서 576 데이터셋 이용 승인을 받고 API 키를 발급받습니다. `.env.example`을 복사해 `.env`를 만들고 `AIHUB_APIKEY`에 키를 넣습니다.
    - 디스크가 약 35GB 필요합니다. AI Hub 원본이 약 28GB이고, 받는 동안 zip 하나(약 3GB)의 2배와 여유 2GB가 더 필요합니다.

2. **명령 하나로 준비**
    ```bash
    uv run scripts/prepare_train_data.py
    ```
    - AI Hub zip 16개(허용 조합 라벨 8개, 이미지 8개)를 하나씩 받고, `data/external/aihub/raw/`에 풀고, zip을 지웁니다. 이미 다 푼 zip은 건너뜁니다.
    - 정제 목록 [data/label_fix/label_fixes.csv](data/label_fix/label_fixes.csv)대로 고친 사본을 만듭니다. 몇 분 걸립니다.
        - Kaggle: `data/processed/train_annotations/` (이미지는 `data/raw/train_images/`)
        - AI Hub: `data/processed/ai_hub/train_annotations/`, `train_images/`. 정제로 클래스나 메타데이터가 바뀐 이미지는 파일명도 라벨에 맞게 고칩니다.
    - 두 데이터의 변경 내역을 `data/label_fix/apply_label_fixes_v2.csv` 하나에 남깁니다. 같은 정제 목록으로 다시 실행하면 새 파일을 만들지 않습니다.
    - 중간에 멈추면 같은 명령을 다시 실행합니다.
    - 성공하면 아래처럼 요약이 출력됩니다.
        ```text
        Kaggle: JSON 763개 → 771개 (data/processed/train_annotations)
        AI Hub: 이미지 11,974장, JSON 45,819개, 이름을 고친 이미지 13장 (data/processed/ai_hub). PNG 하드링크 11,974장, 복사 0장
        변경 기록: data/label_fix/apply_label_fixes_v2.csv aihub add 70, aihub delete 161, aihub modify 968, kaggle add 8, kaggle modify 13
        ```

3. **정제 목록 형식**
    - 한 줄은 알약 하나를 어떻게 고칠지 적습니다. `source` 열(`kaggle`/`aihub`)로 어느 데이터의 줄인지 구분합니다.
    - 열의 뜻과 규칙은 [src/pill_detection/data.py](src/pill_detection/data.py) 모듈 docstring에 있습니다.
    - aihub 줄은 AI Hub EDA 노트북(`notebooks/eda/01_raewoo_external_aihub_eda.ipynb`) 2-4가 만듭니다. 손으로 고치지 않습니다.
    - 이 CSV를 직접 읽어 Kaggle 라벨만 고치는 코드는 `source == "kaggle"`인 줄만 씁니다.

4. **테스트**
    ```bash
    uv run pytest
    ```
    - 기존 Kaggle 라벨 수정 결과와 01 노트북의 정제 결론(이미지 수, 형식, 라벨 누락·좌표 오류 없음, 이름 변경 13장)이 그대로인지 확인합니다.
    - 실제 데이터가 없는 테스트는 이유를 적고 건너뜁니다. 전체 실행은 약 3분 걸립니다.

## 🛠️ Scripts
`scripts/`의 스크립트는 Python을 따로 켜지 않고 터미널에서 명령 한 줄로 실행합니다. 위 Dependencies의 `uv sync`를 먼저 마친 상태여야 합니다.

- **레포 최상위**(`pill-object-detection/`)에서 `uv run scripts/<이름>.py`로 실행합니다. 가상환경을 켜지 않아도 됩니다.
- 경로 인자는 지금 터미널이 있는 폴더 기준입니다. 레포 최상위에서는 `data/...`로 시작하는 경로를 넣습니다. 
- 인자 설명은 `--help`로 봅니다.
    ```bash
    uv run scripts/make_fixed_annotations.py --help
    ```
- 아래 예시의 줄 끝 `\`는 macOS/Linux 터미널에서 명령을 여러 줄로 나눠 쓰는 표시입니다. Windows PowerShell에서는 `\`를 `` ` ``(백틱)으로 바꾸거나 한 줄로 이어 씁니다.

### 1. prepare_train_data.py: 학습 데이터 준비
인자 없이 실행합니다. 하는 일은 위 [학습 데이터 준비](#-학습-데이터-준비)에 있습니다. [src/pill_detection/data.py](src/pill_detection/data.py)의 `make_train_dataset()`을 부릅니다.
- AI Hub 받기와 풀기만 따로 할 때는 `download_aihub.py`(filekey로 zip 받기), `extract_aihub.py`(zip 풀고 지우기)를 씁니다. 사용법은 `--help`와 각 파일 맨 위 설명에 있습니다.

### 2. make_fixed_annotations.py: Kaggle만 고친 사본 만들기
정제 목록 CSV의 kaggle 줄대로 고친 annotation 사본과 변경 기록 CSV를 한 번에 만듭니다. [src/pill_detection/data.py](src/pill_detection/data.py)의 `make_fixed_annotations()`를 부르기만 하며, 인자의 자세한 뜻과 규칙은 그 함수의 docstring에 있습니다.
- 1차 가공(원본 → `data/processed/train_annotations`)은 `prepare_train_data.py`가 합니다. 이 스크립트는 Kaggle 사본을 한 번 더 고치는 2차 가공에 씁니다.
- `--change-csv`에 `data/label_fix/apply_label_fixes.csv`를 넣지 않습니다. 그 이름의 최신 기록(`_v2`)은 Kaggle과 AI Hub를 함께 담고 있어서, 같은 이름을 쓰면 두 기록이 번갈아 쌓입니다.

| 인자 | 하는 일 | 예시 |
| --- | --- | --- |
| `--fix-csv` | 정제 목록 CSV. 읽기만 함. kaggle 줄만 씀 | `data/label_fix/label_fixes_round2.csv` |
| `--base-ann-dir` | 기준 annotation 폴더. 읽기만 함 | `data/processed/train_annotations` |
| `--img-dir` | 이미지 폴더. 읽기만 함. 2차 이후에도 원본 | `data/raw/train_images` |
| `--out-ann-dir` | 사본 annotation 폴더. **폴더 전체를 지우고 다시 만듦** | `data/processed/round2/train_annotations` |
| `--change-csv` | 변경 기록 CSV. 내용이 바뀌면 `_v2`, `_v3` …로 새로 저장 | `data/label_fix/apply_label_fixes_round2.csv` |

- **예시. 2차 가공: 1차 사본을 기준으로 한 번 더 고치기**
    - `--base-ann-dir`에 1차 사본을 넣고, `--out-ann-dir`는 1차 사본과 다른 폴더로 정합니다. 폴더 이름은 `train_annotations`로 같아야 합니다.
    - `--img-dir`는 그대로 원본 이미지 폴더입니다. 사본에는 이미지가 없습니다.
    - 수정 목록과 변경 기록은 1차와 다른 파일 이름을 쓰면 단계별 기록이 섞이지 않습니다. 아래 `_round2` 파일 이름은 예시입니다.
    ```bash
    uv run scripts/make_fixed_annotations.py \
        --fix-csv data/label_fix/label_fixes_round2.csv \
        --base-ann-dir data/processed/train_annotations \
        --img-dir data/raw/train_images \
        --out-ann-dir data/processed/round2/train_annotations \
        --change-csv data/label_fix/apply_label_fixes_round2.csv
    ```

- **NOTE**
    1. 같은 수정 목록으로 다시 실행해도 결과는 같습니다. 사본 폴더는 다시 만들어지고, 변경 기록은 이전 기록과 내용이 같으면 새 파일을 만들지 않고 이전 파일 경로를 보여줍니다.
    2. 수정 목록이나 경로에 문제가 있으면 `[오류]` 뒤에 원인과 해결 방법이 나오고 멈춥니다. 메시지에 적힌 대로 고친 뒤 다시 실행하세요.

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표