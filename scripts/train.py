"""실험 YAML 하나로 학습한다. (EXPERIMENTS.md 규칙 1)

    uv run python scripts/train.py configs/exp/yolov8s_640.yaml
    WANDB_MODE=offline uv run python scripts/train.py configs/smoke.yaml   # W&B에 올리지 않고 동작만 확인

YAML 파일 이름이 실험 이름이다. 결과는 runs/<실험 이름>/, W&B run 이름도 같다.
W&B 키는 환경 변수 WANDB_API_KEY로 받는다. 로컬에서는 레포 루트의 .env에 적어 두면 여기서 읽는다.
"""

import argparse
import netrc
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO, settings

WANDB_PROJECT = "pill-object-detection"
RESERVED_KEYS = ("project", "name", "save_dir")  # 실험 이름은 파일 이름으로 정한다

NO_WANDB_KEY = """\
[오류] W&B API 키를 찾지 못했습니다. 아래 중 하나로 키를 넣은 뒤 다시 실행하세요.
  로컬: 레포 루트 .env 파일에 WANDB_API_KEY=<키> 한 줄 추가 (.env는 git에 올라가지 않음)
  Colab/Kaggle: 공용 노트북의 키 셀을 먼저 실행
  W&B에 올리지 않고 동작만 확인: WANDB_MODE=offline 을 앞에 붙여 실행
키는 https://wandb.ai/authorize 에서 확인할 수 있습니다."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exp_yaml", help="configs/exp/<실험 이름>.yaml")
    return parser.parse_args()


def has_wandb_key():
    if os.environ.get("WANDB_MODE") in ("offline", "disabled") or os.environ.get("WANDB_API_KEY"):
        return True
    try:  # 예전에 wandb login을 해 둔 컴퓨터
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (OSError, netrc.NetrcParseError):
        return False


def run_env():
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


def git_commit():
    """지금 코드의 커밋. 커밋하지 않은 수정이 있으면 -dirty를 붙인다."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def add_callbacks(model, run_info):
    """Ultralytics W&B 로깅에 우리 정보를 더한다. model.add_callback으로 넣은 콜백은 W&B 콜백보다 먼저 실행된다."""

    def log_run_info(trainer):
        import wandb

        if wandb.run is not None:
            wandb.run.config.update(run_info, allow_val_change=True)

    def add_competition_metric(trainer):
        # 대회 지표 mAP@[0.75:0.95]. all_ap는 (클래스 수, IoU 0.50~0.95의 10단계) 배열이라 뒤쪽 5칸이 0.75~0.95
        # trainer.metrics에 넣어 두면 바로 다음에 실행되는 W&B 콜백이 epoch별 그래프로 올린다.
        # 마지막에는 best.pt로 다시 평가한 값이 한 번 더 올라가므로 W&B summary에는 best.pt 기준 값이 남는다.
        all_ap = trainer.validator.metrics.box.all_ap
        if len(all_ap):
            trainer.metrics["metrics/mAP75-95(B)"] = float(all_ap[:, 5:].mean())

    model.add_callback("on_train_start", log_run_info)
    model.add_callback("on_fit_epoch_end", add_competition_metric)


def main():
    args = parse_args()
    exp_path = Path(args.exp_yaml)
    cfg = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}

    reserved = [k for k in RESERVED_KEYS if k in cfg]
    if reserved:
        print(f"[오류] 실험 YAML에 {reserved}를 쓰지 않습니다. 실험 이름은 파일 이름으로 정해집니다.")
        return 1
    if "model" not in cfg:
        print("[오류] 실험 YAML에 model이 없습니다. 예) model: yolov8s.pt")
        return 1
    data_yaml = Path(cfg.get("data", "configs/data.yaml"))
    if not data_yaml.exists():
        print(f"[오류] {data_yaml}이 없습니다. 먼저 uv run python scripts/prepare_data.py를 실행하세요.")
        return 1

    load_dotenv()  # 이미 있는 환경 변수는 덮어쓰지 않는다
    if not os.environ.get("WANDB_ENTITY"):  # .env에 빈 값으로 남아 있으면 지워서 개인 계정에 기록되게 한다
        os.environ.pop("WANDB_ENTITY", None)
    if not has_wandb_key():
        print(NO_WANDB_KEY)
        return 1
    settings.update({"wandb": True})  # Colab/Kaggle은 세션마다 설정이 초기화되므로 매번 켠다

    run_info = {
        "data_version": yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("data_version", "unknown"),
        "git_commit": git_commit(),
        "run_env": run_env(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    print(f"[실험] {exp_path.stem}  {run_info}")

    model = YOLO(cfg["model"])
    add_callbacks(model, run_info)
    model.train(cfg=str(exp_path), project=WANDB_PROJECT, name=exp_path.stem,
                save_dir=str(Path("runs") / exp_path.stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
