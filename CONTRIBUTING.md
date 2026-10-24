# Git Cheat sheet
> `<...>`로 표시된 부분은 각자 상황에 맞게 바꿔서 입력합니다. (`<`, `>` 기호는 지우고 입력)

0. 최초 1회 설정
    - 커밋에 기록될 내 이름과 이메일을 등록합니다. GitHub 계정의 이메일을 사용하세요.
        ```bash
        git config --global user.name "<내 이름>"
        git config --global user.email "<GitHub 계정 이메일>"
        ```
    - Github 웹사이트에서 Personal Access Token을 발급받아 로컬 환경에 저장하세요.

1. 레포지토리 클론
    - 원격 레포지토리를 내 컴퓨터로 복사합니다. 클론은 처음 한 번만 하면 됩니다.
        ```bash
        git clone <원격 레포지토리 URL>
        cd <레포지토리 폴더 이름>
        ```
    - 원격 레포지토리 URL은 GitHub 레포지토리 페이지의 초록색 `<> Code` 버튼에서 복사할 수 있습니다.

2. 브랜치 생성 및 체크아웃
    - 작업을 시작하기 전에 항상 main 브랜치를 최신 상태로 맞춘 뒤, 거기서 새 브랜치를 만듭니다.
        ```bash
        git switch main          # main 브랜치로 이동
        git pull origin main     # 원격의 최신 main 내용을 받아오기
        git switch -c <브랜치 이름>   # 새 브랜치를 만들고 그 브랜치로 이동
        ```
    - 브랜치 이름은 위의 branch naming rule을 따릅니다. 예) `git switch -c eda/raewoo`, `git switch -c exp/add-dropout`
    - 이미 만들어 둔 브랜치로 이동할 때는 `-c` 없이 `git switch <브랜치 이름>`을 사용합니다.
    - 현재 내가 어느 브랜치에 있는지 확인하기
        ```bash
        git branch       # * 표시가 붙은 것이 현재 브랜치
        ```

3. 작업 단위 스테이징 및 커밋
    - 변경 사항 확인
        ```bash
        git status       # 어떤 파일이 바뀌었는지 확인
        git diff         # 바뀐 내용을 줄 단위로 확인
        ```
    - 스테이징: 이번 커밋에 포함할 파일을 고릅니다.
        ```bash
        git add <파일 경로>     # 특정 파일만 추가 (권장)
        git add .              # 현재 폴더의 모든 변경 사항 추가
        ```
    - 커밋: 위의 commit convention에 맞춰 메시지를 작성합니다. `-m`을 두 번 쓰면 첫 번째는 subject, 두 번째는 body가 됩니다.
        ```bash
        git commit -m "code: FashionMNIST 클래스별 분포 EDA 추가" -m "클래스별 샘플 수와 평균 이미지를 시각화함"
        ```
    - 커밋 기록 확인
        ```bash
        git log --oneline
        ```

4. 푸쉬
    - 내 브랜치의 커밋을 원격 레포지토리에 올립니다. **main이 아닌 내 작업 브랜치로 push 하는지 반드시 확인하세요.**
        ```bash
        git push -u origin <브랜치 이름>   # 해당 브랜치를 처음 push할 때
        git push                         # 이후부터는 이것만 입력해도 됨
        ```

5. PR
    - push 후 GitHub 레포지토리 페이지에 들어가면 `Compare & pull request` 버튼이 보입니다. 이 버튼을 누릅니다.
        - 버튼이 보이지 않으면 `Pull requests` 탭 → `New pull request` → `base: main`, `compare: <내 브랜치>`를 선택합니다.
    - PR 제목을 쓰고, 본문에 자동으로 채워진 PR 템플릿의 각 항목을 작성한 뒤 `Create pull request`를 누릅니다.
        - `관련 이슈`에 이슈 번호를 적으면 그 이슈의 보드 상태가 자동으로 바뀝니다. 아래 [PR과 이슈 상태](#pr과-이슈-상태)를 참고하세요.
        - 아직 작업 중이면 버튼 옆 화살표에서 `Create draft pull request`를 고릅니다.
    - 팀원의 리뷰를 받은 뒤 main에 병합(Merge)합니다.
    - 리뷰 반영으로 수정할 내용이 생기면, 같은 브랜치에서 3~4번(커밋, 푸쉬)을 반복하면 PR에 자동으로 반영됩니다.

6. PR 병합 후 정리
    - 병합이 끝나면 main을 최신화하고, 다 쓴 브랜치는 삭제합니다. 다음 작업은 다시 2번부터 시작합니다.
        ```bash
        git switch main
        git pull origin main
        git branch -d <브랜치 이름>   # 로컬 브랜치 삭제(선택)
        ```

7. Tips
    - 항상 main에 최근 변경사항이 있는 지 확인하는 습관을 들이세요. origin과 local의 drift가 있지 않도록 항상 로컬 환경을 최신화 해주세요. 

# 협업 방식

## general rule
- 절대로 main 브랜치에는 push하지 않습니다. 
- 각 팀원은 작업 브랜치를 만들어서 작업하고, main에 병합하기 위해서는 PR을 발행합니다. 
    
## branch naming rule
- **브랜치 이름은 반드시 영어로 작성합니다.** 
- **브랜치 이름에서 띄어쓰기는 '-' 하이픈을 사용합니다.** 
- **브랜치 전략: Github Flow를 사용합니다.**


| 브랜치 | 이름 규칙 | 예시 | 
| -- | -- | -- |
| main | 직접 푸시 금지, PR로만 머지	| - |
| 기능 | feat/이슈번호-설명	| feat/12-wandb-logging |
| 버그 | fix/이슈번호-설명 | fix/15-bbox-normalize |
| 실험 | exp/모델-핵심변경 | exp/rtdetr-l-640 |
| EDA | eda/설명 | eda/TODO | 
| 문서 | docs/설명 | docs/readme-results | 
| 잡업 | chore/설명 | chore/uv-setup | 

> 추후 브랜치 유형 추가가 필요하면 여기에 추가됩니다. 

- 이슈와 연결된 작업이면 유형과 상관없이 `<유형>/<이슈번호>-설명`으로 만듭니다. 예) `eda/5-raewoo`, `chore/11-pipeline`
    - 이 브랜치를 처음 push하면 프로젝트 보드에서 그 이슈가 자동으로 `In progress`로 옮겨집니다. 이후 상태 변화는 아래 [PR과 이슈 상태](#pr과-이슈-상태)를 참고하세요.

## PR과 이슈 상태
- PR을 만들면 `.github/pull_request_template.md` 양식이 본문에 자동으로 채워집니다. 각 항목을 채우고, `관련 이슈`에는 `- #12`처럼 이슈 번호를 적습니다.
- 브랜치 이름의 이슈 번호와 `관련 이슈`에 적은 번호를 보고, 프로젝트 보드의 이슈 상태를 아래처럼 자동으로 옮깁니다. 보드에서 손으로 옮길 필요가 없습니다.

    | 언제 | 이슈 상태 |
    | -- | -- |
    | 이슈 번호가 든 브랜치를 처음 push | In progress |
    | PR을 Draft로 열거나, 열린 PR을 Draft로 되돌림 | In progress |
    | PR을 열거나 다시 엶, Draft에서 Ready for review로 바꿈 | In review |
    | PR을 merge하지 않고 닫음 | In progress |
    | PR을 merge함 | Done (이슈도 자동으로 닫힘) |

- 커밋을 추가로 push하는 것만으로는 상태가 바뀌지 않습니다. 리뷰를 반영해 push해도 In review가 유지됩니다.
- 자동화 설정은 `.github/workflows/project-status.yml`에 있습니다. 상태가 안 바뀌면 GitHub 레포의 `Actions` 탭에서 `이슈 상태 자동 이동` 실행 로그를 확인하세요.

## commit convention
- 일정 작업 단위가 완성되면 커밋을 진행합니다.
- **커밋 메시지는 한국어로 작성해도 좋습니다.**
- 커밋 메시지에는 작업 내용을 포괄할 수 있는 묘사(description)을 작성합니다. 팀원이 커밋 메시지만 보고 무슨 일을 했는 지 확인할 수 있어야 합니다. 
- 커밋 메시지의 구성
    ```text
    <type>: <subject>

    <body>
    ```

    | **타입** | **용도** | **예시** |
    | --- | --- | --- |
    | `feat` | 새 기능 | `feat: add wandb bbox logging` |
    | `fix` | 버그 수정 | `fix: convert xywh to xyxy before NMS` |
    | `exp` | 실험 설정 변경 (팀 자체 규칙) | `exp: yolov8s img640 mosaic=1.0` |
    | `refactor` | 동작 변화 없는 구조 개선 | `refactor: split dataset class` |
    | `docs` | 문서 | `docs: add results table to README` |
    | `test` | 테스트 | `test: add bbox conversion test` |
    | `chore` | 설정, 패키지 | `chore: pin ultralytics version` |

    > 추후 타입 추가가 필요하면 여기에 추가됩니다. 
