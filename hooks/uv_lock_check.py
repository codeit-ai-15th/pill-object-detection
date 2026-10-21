"""pre-commit hook: uv.lock이 pyproject.toml과 일치하는지 검사한다.

어긋나 있으면 원인과 해결 명령을 출력하고 커밋을 막는다.
"""

import shutil
import subprocess
import sys

# uv lock --check 종료 코드: 0 일치, 1 lock 갱신 필요, 2 그 외 오류
LOCK_OUTDATED = 1

FIX_STEPS = """\
[해결] 아래 명령을 순서대로 실행한 뒤 다시 커밋하세요.
  uv lock
  git add pyproject.toml uv.lock
  git commit -m "<type>: <subject>" -m "<body>"

git commit --no-verify로 우회하지 마세요. main의 uv.lock이 어긋나면
팀원 모두 uv sync를 할 때마다 uv.lock이 바뀌어 브랜치를 옮길 수 없게 됩니다."""


def staged_files():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    )
    return set(result.stdout.splitlines())


def explain_cause(has_pyproject, has_lock):
    if has_pyproject and not has_lock:
        return ("pyproject.toml은 바뀌었는데 uv.lock이 이번 커밋에 없습니다.\n"
                "  uv lock을 실행하지 않았거나, 실행한 뒤 uv.lock을 git add하지 않았습니다.")
    if has_lock and not has_pyproject:
        return ("uv.lock은 바뀌었는데 pyproject.toml 변경이 이번 커밋에 없습니다.\n"
                "  바뀐 pyproject.toml을 git add하지 않았습니다.")
    if has_pyproject and has_lock:
        return ("두 파일 모두 커밋에 있지만 내용이 서로 맞지 않습니다.\n"
                "  pyproject.toml을 고친 뒤 uv lock을 다시 실행하지 않았거나,\n"
                "  uv.lock 머지 충돌을 직접 고쳤을 수 있습니다.")
    return "pyproject.toml과 uv.lock이 서로 맞지 않습니다."


def main():
    if shutil.which("uv") is None:
        print("[오류] uv 명령을 찾을 수 없습니다.\n"
              "  README의 'uv 설치' 단계를 따라 uv를 설치하세요.\n"
              "  설치했는데도 이 메시지가 나오면 터미널에서 커밋해 보세요"
              "(GUI git 도구는 PATH가 다를 수 있습니다).")
        return 1

    result = subprocess.run(["uv", "lock", "--check"], capture_output=True, text=True)
    if result.returncode == 0:
        return 0

    if result.returncode != LOCK_OUTDATED:
        print(result.stderr.strip())
        print("\n[오류] uv lock --check 실행 중 문제가 생겼습니다. 위 uv 메시지를 확인하세요.\n"
              "  pyproject.toml 문법 오류나 네트워크 문제일 수 있습니다.")
        return 1

    staged = staged_files()
    has_pyproject = "pyproject.toml" in staged
    has_lock = "uv.lock" in staged

    print("uv.lock이 pyproject.toml과 맞지 않아 커밋을 중단했습니다.\n")
    print("[이번 커밋에 포함된 파일]")
    print(f"  pyproject.toml : {'포함' if has_pyproject else '빠짐'}")
    print(f"  uv.lock        : {'포함' if has_lock else '빠짐'}\n")
    print("[원인]")
    print(f"  {explain_cause(has_pyproject, has_lock)}\n")
    print(FIX_STEPS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
