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
│   ├── external/              # AIhub에서 내려받은 데이터. 내려받은 그대로 두고 수정하지 않음. 
│   ├── raw/                   # 원본 데이터. 내려받은 그대로 두고 수정하지 않음
│   └── processed/             # YOLO 포맷으로 변환한 결과
├── notebooks/                 # 개인 탐색용 노트북
│   ├── eda/                   # 개인 EDA 노트북
│   └── modeling/              # 각자 맡은 모델링 노트북
│       ├── 02-cj-faster-rcnn-experiments.ipynb   # Faster·함수 내장
│       ├── 03-cj-cascade-rcnn-experiments.ipynb  # Cascade·함수 내장
│       └── 04-cj-two-stage-comparison.ipynb     # 동일 validation 재평가
├── src/pill_detection/        # 공용 코드 패키지 (노트북·스크립트·데모 어디서든 import)
│   └── utils.py               # 기존 bbox 변환·시각화
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


## 두 모델 실험과 비교

`notebooks/modeling/02-cj-faster-rcnn-experiments.ipynb`와 `03-cj-cascade-rcnn-experiments.ipynb`는 각 노트북 상단에 데이터·학습·평가·CSV 함수 구현을 포함합니다. 분리했던 `detection_data.py`, `detection_analysis.py`, `faster_experiments.py`는 사용하지 않습니다. 기존 `utils.py`는 다른 작업에서 계속 사용할 수 있도록 보존합니다.

- 공통 흐름: 환경 A → B → 설치 변경 시 현재 VM의 Python 세션만 재시작 → A → B → 셀 1 → 내장 함수 → 셀 2부터 실행합니다. 새 공용 Python 파일의 GitHub 반영·업로드는 필요하지 않습니다. GitHub는 기존 환경 기준 파일과 팀 수정표를 준비하는 데 사용합니다.
- Faster: 기존 GitHub의 `uv.lock`을 기본 Colab 커널에 적용합니다. 로컬은 `uv sync --locked` 후 해당 `.venv` 커널을 사용합니다.
- Cascade: 내장 함수를 실행한 뒤 환경 C에서 공식 MMDetection 3.3.0·MMCV 2.1.0·Torch 2.1.0+cu121·Python 3.11 전용 환경을 생성합니다. 기본 Colab 커널을 교체하지 않고 노트북에 정의한 함수를 별도 Python 프로세스에 전달합니다. 전용 환경은 Colab의 `/content`에만 만들며 프로젝트의 `pyproject.toml`·`uv.lock`을 바꾸지 않습니다. 환경 C의 GPU NMS·RoIAlign 검증 뒤 실제 batch 16·640·FP32 스모크를 수행합니다.
- 실험: 두 모델 각각 COCO 사전학습 기반 FE/PT/FT × 학습률 3개로 seed 0 탐색 9회 후 상위 두 설정에 seed 1·2를 추가합니다. 최대 100 epoch·patience 30·AdamW·FP32·추가 랜덤 증강 없음·고정 train/val을 사용합니다. 실행 스위치는 기본 False입니다.
- 데이터: 원본·고정 분할·수정표 경로를 노트북에서 지정합니다. `label_fixes.csv`의 `bbox`와 팀 수정표의 `bbox_after`를 지원합니다. 원본 데이터는 수정하지 않습니다.
- 결과: 모델별 `runs/cj_faster_experiments/`, `runs/cj_cascade_experiments/`에 체크포인트·W&B 기록·mAP·AP·혼동행렬·오류 이미지를 저장합니다. 재개는 실행 폴더를 명시할 때만 수행합니다. 제출 CSV는 `submissions/` 아래 모델별 폴더에 생성하며 Kaggle에 자동 제출하지 않습니다.
- 비교: 두 학습 노트북의 셀 17-1에서 `EXPORT_COMPARISON=True`로 최종 설정의 seed 0·1·2 validation 자료를 저장한 뒤 `04-cj-two-stage-comparison.ipynb`를 실행합니다. 비교에 필요한 함수도 상단에 내장되어 있습니다. 데이터·라벨·클래스 매핑·고정 조건·예측 파일 해시를 검사하고 동일 기본 커널에서 mAP@[0.75:0.95]를 재계산합니다. 별도 Cascade 환경과 재학습은 필요하지 않습니다. 서로 다른 공식 프레임워크·체크포인트·BN eps의 차이와 GPU 정보를 함께 기록합니다.

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표