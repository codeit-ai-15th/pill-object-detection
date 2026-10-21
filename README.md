# 💊 Pill Object Detection

## 📑 Project Overview
이 프로젝트는 경구약제 이미지 인식 기능을 이용해 사용자가 알약 사진을 올리면 그 알약이 무슨 알약인 지 검출해주는 서비스입니다. 

- 1차 마일스톤: 알약 이미지를 입력하면 약 이름을 검출해주는 기능
- (선택)2차 마일스톤: 알약 정보를 입력하면 [식의약 데이터 포털 공공API](https://data.mfds.go.kr/OPCAA01F01/search?page=1&limit=10&totalCnt=65&sort=openYmd&sortOrder=true&openApiPage=1&openApiLimit=50&openApiTotalCnt=50&openApiSort=openYmd&openApiSortOrder=true&openApiPage2=1&openApiLimit2=10&openApiTotalCnt2=0&openApiSort2=openYmd&openApiSortOrder2=true&fileDataPage=1&fileDataLimit=10&fileDataTotalCnt=15&fileDataSort=openYmd&fileDataSortOrder=true&fileDataPage2=1&fileDataLimit2=10&fileDataTotalCnt2=0&fileDataSort2=openYmd&fileDataSortOrder2=true&dataSetPage=1&dataSetLimit=10&dataSetTotalCnt=0&dataSetSort=openYmd&dataSetSortOrder=true&selectedTab=tab2&taskDivsCd=3&taskDivsDtlCd=7&allChk=&srchSrvcKorNm=&srchSrvcExplain=&srchDeptInstt=&srchInstt=00000&srchInstt=10000&srchInstt=20000&srchInstt=30000&srchInstt=40000&srchInstt=50000&srchInstt=60000&srchInstt=80000&srchInstt=A1000&srchInstt=A2000&srchInstt=A3000&srchInstt=A4000&srchInstt=A5000&srchInstt=A6000&srchInstt=A7000&srchSrvcSeCd=&srchSrvcTypeCd=)를 통해 복용 가이드를 제공해주는 기능

> TODO 데모 영상

## 📊 DataSet Overview
본 프로젝트는 [AI Hub에서 제공하는 공개 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=576)을 활용했습니다. 

## 📂 Project Structure
> TODO

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
    4. 스크립트는 `uv run python src/train.py`처럼 실행합니다. 가상환경을 따로 켤 필요가 없습니다.
    5. `pyproject.toml`이나 `uv.lock`을 커밋하면 pre-commit hook이 `uv lock --check`로 두 파일이 일치하는지 검사합니다. 실패하면 터미널에 원인과 해결 명령이 출력되니 그대로 따라 하면 됩니다. `--no-verify`로 우회하지 마세요.
    6. pre-commit hook을 추가할 때는 스크립트를 `hooks/`에 만들고 `.pre-commit-config.yaml`에 등록합니다.

## 🧠 Model Architecture 
> TODO 최종 선정 모델 아키텍처 다이어그램

## 📖 Results
> 모델별 실험 성과 비교 및 최종 모델 성능 표