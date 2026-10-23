"""실험 YAML 하나로 학습한다. (EXPERIMENTS.md 규칙 1)

    uv run python scripts/train.py configs/exp/yolov8s_640.yaml
    WANDB_MODE=offline uv run python scripts/train.py configs/smoke.yaml   # W&B에 올리지 않고 동작만 확인

YAML 파일 이름이 실험 이름이다. 결과는 runs/<실험 이름>/, W&B run 이름도 같다.
best.pt는 대회 지표 mAP75-95가 가장 높은 epoch의 가중치다.
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
from ultralytics.utils.metrics import DetMetrics

WANDB_PROJECT = "pill-object-detection"
RESERVED_KEYS = ("project", "name", "save_dir")  # 실험 이름은 파일 이름으로 정한다
COMPETITION_METRIC = "metrics/mAP75-95(B)"

NO_WANDB_KEY = """\
[오류] W&B API 키를 찾지 못했습니다. 아래 중 하나로 키를 넣은 뒤 다시 실행하세요.
  로컬: 레포 루트 .env 파일에 WANDB_API_KEY=<키> 한 줄 추가 (.env는 git에 올라가지 않음)
  Colab/Kaggle: 공용 노트북을 Run All (설정 셀이 Drive의 .env나 Kaggle Secrets에서 키를 넣음)
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


def map75_95(metrics):
    """대회 지표 mAP@[0.75:0.95]. box.all_ap는 (클래스 수, IoU 0.50~0.95의 10단계) 배열이라 뒤쪽 5칸이 0.75~0.95"""
    all_ap = metrics.box.all_ap
    return float(all_ap[:, 5:].mean()) if len(all_ap) else 0.0


def use_competition_metric():
    """Ultralytics 검증 결과에 대회 지표를 넣고, best.pt를 고르는 기준(fitness)을 mAP50-95에서 대회 지표로 바꾼다.

    검증 결과에 들어가므로 epoch마다 results.csv와 W&B에 기록되고, 마지막 best.pt 재평가 값도 W&B summary에 남는다.
    fitness가 바뀌므로 best.pt 저장과 조기 종료(patience)도 대회 지표를 따른다. RT-DETR도 같은 DetMetrics를 쓴다.
    """
    base_results = DetMetrics.results_dict.fget  # 안에서 self.fitness를 읽으므로 "fitness" 값도 대회 지표가 된다
    DetMetrics.fitness = property(map75_95)
    DetMetrics.results_dict = property(lambda self: {**base_results(self), COMPETITION_METRIC: map75_95(self)})


def add_callbacks(model, run_info):
    """W&B run에 우리 정보를 더하고, 콘솔에 대회 지표를 보여 준다."""

    def log_run_info(trainer):
        import wandb

        if wandb.run is not None:
            wandb.run.config.update(run_info, allow_val_change=True)

    def print_competition_metric(trainer):
        # Ultralytics 검증 표에는 이 열이 없어서 따로 출력한다
        value = trainer.metrics.get(COMPETITION_METRIC)
        if value is None:
            return
        if trainer.validator.training:  # epoch마다 하는 검증
            print(f"[대회 지표] mAP75-95 {value:.4f} | 지금까지 최고 {trainer.best_fitness:.4f} (best.pt)")
        else:  # 학습이 끝난 뒤 best.pt를 다시 불러와 평가. 계산 방식이 조금 달라 위 최고값과 소수점 끝자리가 다를 수 있다
            print(f"[대회 지표] best.pt 최종 평가 mAP75-95 {value:.4f}")

    model.add_callback("on_train_start", log_run_info)
    model.add_callback("on_fit_epoch_end", print_competition_metric)


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
    use_competition_metric()

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
    # data를 직접 넘긴다. YAML에 data가 없으면 Ultralytics가 예제 데이터셋 coco8.yaml로 학습한다
    model.train(cfg=str(exp_path), data=str(data_yaml), project=WANDB_PROJECT, name=exp_path.stem,
                save_dir=str(Path("runs") / exp_path.stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
