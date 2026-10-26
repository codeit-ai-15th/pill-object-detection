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
├── data/                      # 폴더만 git으로 관리하고 내용은 추적하지 않음
│   ├── raw/                   # 원본 데이터. 내려받은 그대로 두고 수정하지 않음
│   ├── external/              # 대회 외부 데이터. 하위 폴더는 raw/와 같은 이름(train_annotations/, train_images/)으로 둠
│   │   ├── aihub/             # AI Hub 원본 zip. AI Hub 파일 이름 그대로 둠 (이미지 zip은 푼 뒤 지워도 됨)
│   │   ├── aihub_raw/         # 라벨 zip을 그대로 푼 원본 JSON
│   │   └── aihub_converted/   # Kaggle 형식으로 바꾼 라벨과 이미지. raw/와 같은 구조
│   └── processed/             # YOLO 포맷으로 변환한 결과
├── notebooks/                 # 개인 탐색용 노트북
│   ├── eda/                   # 개인 EDA 노트북
│   └── modeling/              # 각자 맡은 모델링 노트북
├── src/pill_detection/        # 공용 코드 패키지 (노트북·스크립트·데모 어디서든 import)
│   ├── data.py                # 라벨 로딩, AI Hub 조합 라벨 읽기와 변환, 조합 단위 분할
│   └── utils.py               # 공용 보일러 플레이트 코드.(bbox 변환, 시각화 등) 
├── script/                    # 직접 실행하는 스크립트
│   ├── download_aihub.py      # AI Hub에서 filekey로 zip 받기 (대회 금지 파일은 받지 않음)
│   └── extract_aihub.py       # 받은 AI Hub zip을 data/raw와 같은 구조로 풀기
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

## 📦 AI Hub 외부 데이터 받기
대회 원본(`data/raw/`)에 더해 [AI Hub 576 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 보충 데이터로 씁니다. 대회 규칙상 `TL_2_조합.zip`, `TS_2_조합.zip`은 쓰지 않습니다. ([RULE.md](RULE.md))

1. **준비(최초 1회)**
    - AI Hub에서 576 데이터셋 이용 승인을 받고 API 키를 발급받습니다.
    - `.env.example`을 복사해 `.env`를 만들고 `AIHUB_APIKEY`에 키를 넣습니다.

2. **받기**
    ```bash
    # 허용 조합 라벨 zip 8개 (약 59MB)
    uv run python script/download_aihub.py 66065 66067 66068 66069 66070 66071 66072 66243
    ```
    - zip은 `data/external/aihub/`의 `train_annotations/`(라벨 zip)와 `train_images/`(이미지 zip)에 받습니다.
    - 금지 파일의 filekey가 있으면 아무것도 받지 않고 멈춥니다. 이미 받은 파일은 건너뜁니다.
    - filekey 목록과 폴더 규칙은 [script/download_aihub.py](script/download_aihub.py) 맨 위 설명을 봅니다.

3. **풀기**
    ```bash
    # 라벨 zip → data/external/aihub_raw/train_annotations/ (원본 JSON 그대로)
    uv run python script/extract_aihub.py data/external/aihub/train_annotations/*.zip

    # 이미지 zip(각 약 3.5GB)은 하나씩 받고, 풀고, 지운다 → data/external/aihub_converted/train_images/
    uv run python script/download_aihub.py 66154
    uv run python script/extract_aihub.py data/external/aihub/train_images/TS_1_조합.zip --delete-images
    # 66156~66161(TS_3~8), 66254(VS_1)도 같은 방법으로 받는다. 모두 풀면 약 28GB이다
    ```
    - Kaggle 형식 라벨(`aihub_converted/train_annotations/`)은 [01_raewoo_external_aihub_eda.ipynb](notebooks/eda/01_raewoo_external_aihub_eda.ipynb) 1-3이 만듭니다.

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표