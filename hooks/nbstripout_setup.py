"""pre-commit hook: 노트북을 커밋할 때 nbstripout git 필터가 설정되어 있는지 확인한다.

.gitattributes가 노트북마다 nbstripout 필터를 쓰라고 지정하지만, 필터를 실제로 실행하는 설정은
각자의 .git/config에 저장되어 clone할 때 따라오지 않는다. 설정이 없으면 git은 필터 없이 출력까지 커밋한다.
그래서 설정이 없으면 여기서 설정하고, 스테이징된 노트북을 다시 스테이징해 이번 커밋부터 출력이 빠지게 한다.
"""

import shutil
import subprocess
import sys

INSTALL_FAILED = """\
[오류] 노트북 출력을 지우는 nbstripout 필터를 설정하지 못했습니다. 위 메시지를 확인한 뒤
아래 명령을 실행하고 다시 커밋하세요.
  uv sync
  uv run nbstripout --install --attributes .gitattributes"""


def filter_configured():
    result = subprocess.run(
        ["git", "config", "--get", "filter.nbstripout.clean"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() != ""


def main(notebooks):
    if filter_configured():
        return 0

    if shutil.which("uv") is None:
        print("[오류] uv 명령을 찾을 수 없습니다. README의 'uv 설치' 단계를 따라 uv를 설치하세요.")
        return 1

    # .gitattributes에 이미 *.ipynb 줄이 있으므로 git 설정만 추가된다. 필터는 레포 .venv의 python으로 실행된다.
    result = subprocess.run(["uv", "run", "nbstripout", "--install", "--attributes", ".gitattributes"])
    if result.returncode != 0 or not filter_configured():
        print(INSTALL_FAILED)
        return 1

    # 다시 스테이징하면 필터를 거쳐 출력이 빠진 내용이 스테이징된다
    subprocess.run(["git", "add", "--", *notebooks], check=True)
    print("[nbstripout] 이 저장소에 노트북 출력 제거 필터를 설정했습니다. 이번 커밋부터 출력이 빠집니다.\n"
          "  작업 중인 노트북 파일의 출력은 그대로 남습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
