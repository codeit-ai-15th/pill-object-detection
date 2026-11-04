"""데모를 Hugging Face Space(Gradio SDK)에 올린다. 코드·캐시·글꼴과 .env가 가리키는 가중치를 한 커밋으로 올린다.

준비
- https://huggingface.co/settings/tokens 에서 write 권한 토큰을 만들어 .env의 HF_TOKEN에 넣는다(또는 `uv run hf auth login`).
- .env의 PILL_DEMO_YOLO / PILL_DEMO_RTDETR / PILL_DEMO_FRCNN이 올릴 가중치를 가리킨다. 비어 있는 계열은 올리지 않는다.

실행 (레포 최상위에서)
    uv run demo/deploy_space.py --repo <계정>/<Space 이름>              # 없으면 Gradio Space를 만든다. Streamlit·Docker SDK는 무료가 아니다
    uv run demo/deploy_space.py --repo <계정>/<이름> --only yolo,frcnn   # 가중치는 일부만
    uv run demo/deploy_space.py --repo <계정>/<이름> --no-weights        # 코드만
    uv run demo/deploy_space.py --repo <계정>/<이름> --delete-missing    # 로컬 변수가 빈 계열은 Space에서도 지운다

무엇을 올리나 (Space 레포 기준 경로)
- demo/            ← 이 폴더 (space/, deploy_space.py, __pycache__ 제외. 글꼴과 drug_info.json 포함)
- src/pill_detection/, data/datasets/classes.csv
- README.md ← demo/space/README.md (Space 설정 YAML, sdk: gradio), requirements.txt ← demo/space/requirements.txt (CPU torch),
  packages.txt ← demo/space/packages.txt (opencv가 쓰는 apt 패키지)
- Space 하드웨어는 ZeroGPU(zero-a10g)를 전제로 한다(무료 플랜에서 CPU basic을 못 씀). torch는 ZeroGPU가 허용하는 2.13.0으로 고정한다.
- weights/<원래 파일 이름> ← 각 계열의 .pt. 계열당 하나만 남도록, Space 변수 PILL_DEMO_<계열>이 가리키던 옛 파일은 지운다.
- Space 변수 PILL_DEMO_YOLO 등을 weights/<파일 이름>으로 맞춘다. 앱은 로컬과 같은 코드로 이 변수를 읽는다.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi  # noqa: E402

from predict import ENV_VARS, FAMILIES, DemoError, configured_weights, detect_family  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"__pycache__", "space", ".pytest_cache"}
EXCLUDE_FILES = {"deploy_space.py", ".DS_Store"}
WEIGHTS_DIR = "weights"


def code_operations():
    """코드·자료 파일의 업로드 operation 목록."""
    ops = []
    for path in sorted(DEMO.rglob("*")):
        rel = path.relative_to(DEMO)
        if path.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts) or path.name in EXCLUDE_FILES or path.suffix == ".pyc":
            continue
        ops.append(CommitOperationAdd(path_in_repo=f"demo/{rel.as_posix()}", path_or_fileobj=str(path)))
    for path in sorted((ROOT / "src" / "pill_detection").glob("*.py")):
        ops.append(CommitOperationAdd(path_in_repo=f"src/pill_detection/{path.name}", path_or_fileobj=str(path)))
    ops.append(CommitOperationAdd(path_in_repo="data/datasets/classes.csv", path_or_fileobj=str(ROOT / "data/datasets/classes.csv")))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(DEMO / "space" / "README.md")))
    ops.append(CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=str(DEMO / "space" / "requirements.txt")))
    ops.append(CommitOperationAdd(path_in_repo="packages.txt", path_or_fileobj=str(DEMO / "space" / "packages.txt")))
    return ops


def weight_operations(slots, families, current_vars, existing_files, delete_missing):
    """가중치 업로드·삭제 operation과 바꿀 Space 변수.

    Args:
        slots (dict): configured_weights()의 결과.
        families (list[str]): 올릴 계열.
        current_vars (dict[str, str]): Space 변수 이름 → 값.
        existing_files (set[str]): Space에 있는 파일 경로.
        delete_missing (bool): 로컬 변수가 빈 계열의 Space 파일·변수도 지울지.

    Returns:
        tuple: (operations, new_vars(dict), removed_vars(list), log(list[str]))
    """
    ops, new_vars, removed_vars, log = [], {}, [], []
    for family in families:
        slot = slots[family]
        var = ENV_VARS[family]
        old = current_vars.get(var, "")
        if slot.path is None:
            if delete_missing and old:
                if old in existing_files:
                    ops.append(CommitOperationDelete(path_in_repo=old))
                removed_vars.append(var)
                log.append(f"{FAMILIES[family]}: 로컬 변수가 비어 있어 Space의 {old}와 변수 {var}를 지움")
            continue
        if slot.error:
            raise DemoError(slot.error)
        actual = detect_family(slot.path)
        if actual != family:
            raise DemoError(f"{var}가 가리키는 {slot.path.name}은 {FAMILIES[actual]} 가중치입니다.\n[해결] .env의 {ENV_VARS[actual]}에 적으세요.")
        new = f"{WEIGHTS_DIR}/{slot.path.name}"
        if old and old != new and old in existing_files:
            ops.append(CommitOperationDelete(path_in_repo=old))
        ops.append(CommitOperationAdd(path_in_repo=new, path_or_fileobj=str(slot.path)))
        new_vars[var] = new
        log.append(f"{FAMILIES[family]}: {old or '(없음)'} → {new}")
    return ops, new_vars, removed_vars, log


def main(argv=None):
    parser = argparse.ArgumentParser(description="데모를 HF Space에 올린다")
    parser.add_argument("--repo", required=True, help="Space 레포 id. 예) raewoo/pill-detection-demo")
    parser.add_argument("--only", default="", help="올릴 가중치 계열. 쉼표 구분 (yolo,rtdetr,frcnn). 기본 전부")
    parser.add_argument("--no-weights", action="store_true", help="코드만 올린다")
    parser.add_argument("--delete-missing", action="store_true", help="로컬 변수가 빈 계열은 Space에서도 지운다")
    parser.add_argument("--private", action="store_true", help="새로 만들 때 비공개 Space로")
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception as e:  # noqa: BLE001
        print(f"[오류] Hugging Face 인증 실패: {e}\n[해결] write 권한 토큰을 .env의 HF_TOKEN에 넣거나 `uv run hf auth login`을 하세요.", file=sys.stderr)
        return 1
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    existing = set(api.list_repo_files(args.repo, repo_type="space"))
    current_vars = {k: v.value for k, v in api.get_space_variables(args.repo).items()}

    ops = code_operations()
    new_vars, removed_vars, log = {}, [], []
    if not args.no_weights:
        try:
            slots = configured_weights()
            families = [f.strip() for f in args.only.split(",") if f.strip()] or list(FAMILIES)
            unknown = [f for f in families if f not in FAMILIES]
            if unknown:
                raise DemoError(f"--only에 모르는 계열: {unknown}. yolo, rtdetr, frcnn 중에서 고르세요.")
            weight_ops, new_vars, removed_vars, log = weight_operations(slots, families, current_vars, existing, args.delete_missing)
            ops.extend(weight_ops)
        except DemoError as e:
            print(f"[오류] {e}", file=sys.stderr)
            return 1
    stale = [f for f in existing if f.startswith(f"{WEIGHTS_DIR}/") and f.endswith(".pt")
             and f not in current_vars.values() and f not in new_vars.values()]

    print(f"Space {args.repo}에 파일 {len(ops)}개를 올립니다 (가중치 {len(new_vars)}개)…")
    api.create_commit(args.repo, repo_type="space", operations=ops, commit_message="demo: 코드·캐시·가중치 갱신")
    for var, value in new_vars.items():
        api.add_space_variable(args.repo, var, value)
    for var in removed_vars:
        api.delete_space_variable(args.repo, var)
    for line in log:
        print("  " + line)
    if stale:
        print(f"[경고] 어느 변수도 가리키지 않는 가중치가 Space에 남아 있습니다 (지우지 않음): {', '.join(stale)}")
    print(f"완료: https://huggingface.co/spaces/{args.repo}  (빌드에 몇 분 걸립니다. 로그는 Space 페이지의 Logs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
