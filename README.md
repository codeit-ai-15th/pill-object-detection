# 💊 Pill Object Detection

## 📑 Project Overview
이 프로젝트는 경구약제 이미지 인식 기능을 이용해 사용자가 알약 사진을 올리면 그 알약이 무슨 알약인 지 검출해주는 서비스입니다. 

- 1차 마일스톤: 알약 이미지를 입력하면 약 이름을 검출해주는 기능
- (선택)2차 마일스톤: 알약 정보를 입력하면 [식의약 데이터 포털 공공API](https://www.data.go.kr/data/15095677/openapi.do)를 통해 복용 가이드를 제공해주는 기능

> TODO 데모 영상

## 📊 DataSet Overview
본 프로젝트는 [AI Hub에서 제공하는 공개 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 활용했습니다. 

> TODO EDA 후 데이터 개요 시각화 자료 첨부

## 📂 Project Structure
학습은 [Ultralytics](https://docs.ultralytics.com/)를 엔진으로 쓰고, 폴더 규칙은 [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)에서 필요한 것만 가져왔습니다.

```text
pill-object-detection/
├── configs/
│   ├── data.yaml              # Ultralytics 데이터셋 정의. prepare_data.py가 classes.py로 생성
│   ├── data_fixes.yaml        # 원본 라벨을 고칠 목록 (원본 파일은 수정하지 않음)
│   ├── splits/                # train/val 분할 파일. 한 번 만들면 모두 같은 파일을 씀
│   ├── smoke.yaml             # 파이프라인 동작 확인용 1 epoch 설정 (실험 아님)
│   └── exp/                   # 실험 1개 = YAML 1개 (예: yolov8s_640.yaml)
├── data/                      # 폴더만 git으로 관리하고 내용은 추적하지 않음
│   ├── raw/                   # 원본 데이터. 내려받은 그대로 두고 수정하지 않음
│   └── processed/             # YOLO 포맷으로 변환한 결과
├── notebooks/                 # 개인 탐색용 노트북
│   └── 00-train-colab-kaggle.ipynb  # Colab·Kaggle 공용 학습 노트북 (하나만 두고 복사하지 않음)
├── src/pill_detection/        # 공용 코드 패키지 (노트북·스크립트·데모 어디서든 import)
│   ├── data/
│   │   ├── convert.py         # 원본 어노테이션 → YOLO 포맷
│   │   └── classes.py         # YOLO index ↔ category_id ↔ 약 이름 (유일한 정의처)
│   ├── predict.py             # 이미지 → 검출 결과. 데모와 제출이 함께 사용
│   └── utils.py               # bbox 변환, 시각화
├── scripts/                   # 실행용 스크립트. 로직은 src/에 두고 짧게 유지
│   ├── prepare_data.py        # data/raw → data/processed, configs/data.yaml 생성
│   ├── train.py               # 실험 YAML 하나로 학습
│   └── make_submission.py     # 제출 파일 생성 (Kaggle 제출이 있다면)
├── demo/
│   └── app.py                 # Streamlit 데모 페이지. predict.py만 import
├── tests/                     # bbox 변환·클래스 매핑 검사 (uv run pytest)
├── runs/                      # 학습 결과. 학습할 때 자동 생성되며 git에서 제외
├── submissions/               # 제출 CSV. make_submission.py가 만들며 git에서 제외
├── hooks/                     # pre-commit hook 스크립트
├── EXPERIMENTS.md             # 실험 규칙
└── CONTRIBUTING.md            # git·PR·커밋 협업 규칙
```

> `demo/app.py`는 아직 없습니다. 나머지 파일은 공용 뼈대로 만들어 두었습니다.

- 모든 명령은 레포 루트에서 실행합니다.
- `src/pill_detection/`은 `uv sync` 때 패키지로 설치됩니다. 노트북에서도 `from pill_detection.data.classes import ...`처럼 바로 불러오고, `sys.path`를 고치지 않습니다.
- `notebooks/`의 파일 이름은 `<번호>-<이니셜>-<설명>.ipynb`로 짓습니다. 예) `01-rk-class-distribution.ipynb`. 
    - 다른 팀원들과 공용으로 쓸 코드는 `src/pill_detection/`으로 옮깁니다.
- 실험 방법과 공용 코드 규칙은 [EXPERIMENTS.md](EXPERIMENTS.md), git·PR·커밋 절차는 [CONTRIBUTING.md](CONTRIBUTING.md)를 따릅니다.

## 🧑‍💻 How to use
이 프로젝트의 환경은 uv로 관리됩니다. 

1. uv 설치(최초 1회)
    - macOS / Linux
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
        ```
    - Windows PowerShell
        ```powershell
        powershell ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

2. uv 설치 확인
    ```bash
    uv --version
    ```

3. 레포를 받은 뒤 환경 생성
    ```bash
    git pull origin main
    rm -rf .venv        # 기존 .venv(예: 3.14)가 있다면 삭제
    uv sync             # Python 3.12가 없으면 uv가 자동으로 받아서 설치함  
    uv run pre-commit install   # 최초 1회: 커밋 전 uv.lock 검사 hook 설치
    ```

4. 의존성 추가 시
    ```bash
    uv add <패키지>
    ```
    | **NOTE**: `pip install`로 설치하지 않습니다. `uv.lock`에 기록되지 않아 다른 팀원 간 환경에 drfit 발생함. 

5. 팀 규칙
    1. `pyproject.toml`과 `uv.lock`은 항상 같이 커밋합니다.
    2. main을 pull한 뒤에는 uv sync를 실행합니다. 다른 팀원이 추가한 패키지가 반영됩니다.
    3. uv.lock에서 머지 충돌이 나면 직접 고치지 마세요. pyproject.toml에 충돌이 있으면 먼저 해결하고, 작업 브랜치에 그대로 있는 상태에서 아래 순서로 lock 파일을 다시 만듭니다. 충돌 표시가 남은 uv.lock으로는 `uv lock`이 실패하므로 main 버전으로 먼저 덮어씁니다.
        ```bash
        git restore --source=main --staged --worktree uv.lock   # 브랜치 이동 없이 main의 uv.lock만 가져옴
        uv lock                                                 # 합쳐진 pyproject.toml 기준으로 다시 생성
        git add pyproject.toml uv.lock
        git commit --no-edit                                    # 작업 브랜치에 머지 커밋 생성
        ```
    4. 스크립트는 `uv run python scripts/train.py configs/exp/yolov8s_640.yaml`처럼 실행합니다. 가상환경을 따로 켤 필요가 없습니다.
    5. `pyproject.toml`이나 `uv.lock`을 커밋하면 pre-commit hook이 `uv lock --check`로 두 파일이 일치하는지 검사합니다. 실패하면 터미널에 원인과 해결 명령이 출력되니 그대로 따라 하면 됩니다. `--no-verify`로 우회하지 마세요.
    6. pre-commit hook을 추가할 때는 스크립트를 `hooks/`에 만들고 `.pre-commit-config.yaml`에 등록합니다.

6. 데이터 준비와 학습 (로컬)
    ```bash
    cp .env.example .env        # 최초 1회. .env에 WANDB_API_KEY를 채움 (.env는 git에 올라가지 않음)
    uv run python scripts/prepare_data.py                                   # data/raw → data/processed
    WANDB_MODE=offline uv run python scripts/train.py configs/smoke.yaml    # 동작 확인 (W&B에 올리지 않음)
    uv run python scripts/train.py configs/exp/yolov8s_640.yaml             # 실험
    uv run python scripts/make_submission.py --weights runs/yolov8s_640/weights/best.pt
    ```
    GPU 학습은 Colab·Kaggle에서 `notebooks/00-train-colab-kaggle.ipynb`로 합니다. 노트북 맨 위 안내를 따릅니다.

7. API 키 위치
    레포가 공개되어 있으므로 키는 코드, YAML, 노트북 셀, 셀 출력 어디에도 쓰지 않습니다. `train.py`는 환경 변수 `WANDB_API_KEY`만 읽습니다.

    | 실행 방식 | W&B 키를 두는 곳 |
    | -- | -- |
    | 로컬 터미널 | 레포 루트 `.env` |
    | 브라우저 Colab | 🔑 보안 비밀 `WANDB_API_KEY` (노트북 액세스 켜기) |
    | VS Code + Colab 확장 | 보안 비밀을 읽지 못함. 노트북 키 셀의 입력창에 붙여 넣기 |
    | 브라우저 Kaggle, VS Code + Kaggle Jupyter Server | Add-ons → Secrets `WANDB_API_KEY` (노트북에 attach) |

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표