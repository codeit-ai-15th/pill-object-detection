# AGENTS.md

AI 코딩 에이전트(Claude Code, Codex 등)가 이 레포에서 작업할 때 읽는 안내서입니다.
규칙과 설명의 원본은 아래에 링크한 문서입니다. 이 파일에는 어디를 보면 되는지와, 다른 문서에 없는 에이전트용 지침만 적습니다.

## 프로젝트
- 알약 사진에서 알약을 검출하는 object detection 프로젝트이며 Kaggle 대회 형식으로 진행합니다. 목표와 마일스톤은 [README.md](README.md)의 Project Overview를 봅니다.
- 초급 팀 프로젝트라서 레포 구조를 일부러 작게 유지합니다. 새 최상위 폴더, 설정 파일 체계, 프레임워크를 들이기 전에 사용자에게 먼저 묻습니다.

## 어디를 보면 되나
| 알고 싶은 것 | 원본 |
| --- | --- |
| 폴더별 용도 | [README.md](README.md) Project Structure |
| 환경 설정, uv 사용법, uv.lock 충돌 해결 | [README.md](README.md) Dependencies |
| 대회 규칙: 사용 금지 데이터, 제출 CSV 형식, 평가 지표 | [RULE.md](RULE.md) |
| 데이터 파일 구성, annotation JSON 필드, 대회 측 가공 내역 | [DATA.md](DATA.md) |
| git 명령 순서 | [CONTRIBUTING.md](CONTRIBUTING.md#git-cheat-sheet) |
| 브랜치 이름, 커밋 메시지, PR과 이슈 보드 | [CONTRIBUTING.md](CONTRIBUTING.md#협업-방식) |
| PR 본문 양식 | [.github/pull_request_template.md](.github/pull_request_template.md) |
| bbox 형식 세 가지와 변환 함수 | [src/pill_detection/utils.py](src/pill_detection/utils.py) 모듈 docstring |
| 라벨 수정 사본 만들기: 수정 목록 CSV 형식, 실행 방법 | [src/pill_detection/data.py](src/pill_detection/data.py) 모듈 docstring, [README.md](README.md) Scripts |
| 환경 변수 이름 | [.env.example](.env.example) |
| git에서 제외하는 파일 | [.gitignore](.gitignore) |
| pre-commit hook | [.pre-commit-config.yaml](.pre-commit-config.yaml), 스크립트는 [hooks/](hooks/) |

## 반드시 지킬 것
이유와 절차는 링크한 문서에 있습니다.
- main 브랜치에서 작업하지 않습니다. 작업 브랜치를 만들고 PR로 병합합니다. → [CONTRIBUTING.md](CONTRIBUTING.md#general-rule)
- 패키지는 `uv add`로만 추가하고 `pip install`은 쓰지 않습니다. `pyproject.toml`과 `uv.lock`은 함께 커밋하고 `uv.lock`은 직접 고치지 않습니다. → [README.md](README.md) Dependencies
- `git commit --no-verify`로 hook을 우회하지 않습니다. hook이 실패하면 출력된 해결 명령을 따릅니다.
- `data/raw/`의 원본 데이터는 수정하지 않습니다. → [README.md](README.md) Project Structure
- 대회에서 금지한 AI Hub 데이터셋은 코드, 노트북, 다운로드 스크립트 어디에도 쓰지 않습니다. → [RULE.md](RULE.md#1-데이터-사용-규칙)
- `.env`는 읽거나 내용을 출력하지 않습니다. 키 값을 코드, 노트북, 로그, 커밋에 넣지 않습니다. 키 이름은 `.env.example`에서 확인합니다.
- `.gitignore`가 제외하는 파일(데이터, 학습 결과, 가중치, 제출 파일)을 `git add -f`로 올리지 않습니다.

## 명령어
설치와 최초 설정은 [README.md](README.md) Dependencies를 따릅니다. 에이전트가 자주 쓸 명령만 적습니다.

```bash
uv sync                              # pull 뒤나 의존성이 바뀐 뒤 환경 맞추기
uv run python <스크립트>               # 가상환경을 켜지 않고 실행
uv add <패키지>                        # 의존성 추가. 개발용은 uv add --dev <패키지>
uv run scripts/<이름>.py --help      # 레포 최상위에서 스크립트 실행. 사용법은 README Scripts
uv run pre-commit run --all-files    # 커밋 전에 hook 미리 돌려 보기
```

## 코드 작성
- 노트북, 스크립트, 데모가 같이 쓰는 코드는 `src/pill_detection/`에 둡니다. import 이름은 `pill_detection`입니다(`from pill_detection.utils import xywh_to_yolo`). → [pyproject.toml](pyproject.toml)
- 새 함수를 만들기 전에 [utils.py](src/pill_detection/utils.py)와 [data.py](src/pill_detection/data.py)에 이미 있는지 확인합니다.
- bbox를 다룰 때는 값이 어떤 형식인지 먼저 확인하고 utils.py의 변환 함수를 씁니다. 형식을 섞어도 에러 없이 점수만 떨어집니다.
- 스타일은 기존 코드([utils.py](src/pill_detection/utils.py), [hooks/](hooks/))를 따릅니다.
  - 주석과 docstring은 한국어로 씁니다.
  - docstring은 Google 스타일(`Args:`, `Returns:`)이고, 타입은 시그니처 대신 docstring 괄호 안에 적습니다.
  - 사용자가 보는 오류 메시지에는 원인과 해결 명령을 함께 적습니다.
- 노트북은 각자의 작업물입니다. 요청받지 않은 다른 팀원의 노트북은 고치지 않습니다. 노트북 출력은 nbstripout이 커밋에서 지우고, 출력을 남길 셀에만 `keep_output` 태그를 붙입니다. → [.gitattributes](.gitattributes)

## git 작업
- 브랜치 이름, 커밋 메시지, PR 본문은 [CONTRIBUTING.md](CONTRIBUTING.md#협업-방식)와 [PR 템플릿](.github/pull_request_template.md)을 따릅니다.
- 이슈와 연결된 작업이면 이슈 번호를 브랜치 이름과 PR 본문 `관련 이슈`에 넣습니다. 프로젝트 보드 자동화가 이 형식에 의존합니다. → [CONTRIBUTING.md](CONTRIBUTING.md#pr과-이슈-상태)
- 커밋, push, PR 생성은 사용자가 요청할 때만 합니다.

## 문서 수정
- 같은 내용을 여러 문서에 복사하지 않습니다. 원본 문서 한 곳을 고치고 다른 곳에서는 링크합니다.
- 폴더나 공용 모듈을 추가하거나 지우면 [README.md](README.md) Project Structure 트리를 함께 고칩니다.
- 환경 변수를 새로 쓰면 [.env.example](.env.example)에 이름과 설명을 추가합니다.
- [RULE.md](RULE.md)와 [DATA.md](DATA.md)는 Kaggle 대회 원문이므로 고치지 않습니다.
- 이 파일에는 링크와 에이전트용 지침만 둡니다. 규칙 내용이 바뀌면 원본 문서를 고칩니다.
