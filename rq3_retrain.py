import argparse
import subprocess
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Batch retrain RQ3 models.")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/zyc/code/V2XGen/rq2/rq2_select",
        help="RQ2 selected dataset directory."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable path."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="rq3_retrain_logs",
        help="Directory to save training logs."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands without executing."
    )

    args = parser.parse_args()

    model_dirs = [
        # "model/early_fusion",
        # "model/late_fusion",
        # "model/PointPillar_V2VNet",
        # "model/PointPillar_V2XViT",
        "model/attfuse",
        # "model/PointPillar_Fcooper",
    ]

    methods = [
        "v2x_gen",
        "coo_test",
        "random",
    ]

    scales = [
        # "0.1",
        "0.15",
        # "0.2",
    ]

    os.makedirs(args.log_dir, exist_ok=True)

    total = len(model_dirs) * len(methods) * len(scales)
    current = 1

    failed_commands = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)

        for method in methods:
            for scale in scales:
                cmd = [
                    args.python,
                    "opencood/rq_eval/rq3_train.py",
                    "--dataset_dir",
                    args.dataset_dir,
                    "--model_dir",
                    model_dir,
                    "--method",
                    method,
                    "--scale",
                    scale,
                ]

                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_name = f"{model_name}_{method}_{scale}_{time_str}.log"
                log_path = os.path.join(args.log_dir, log_name)

                print("=" * 80)
                print(f"[{current}/{total}] Start retraining")
                print(f"model_dir : {model_dir}")
                print(f"method    : {method}")
                print(f"scale     : {scale}")
                print(f"log_path  : {log_path}")
                print("command   :")
                print(" ".join(cmd))
                print("=" * 80)

                if args.dry_run:
                    current += 1
                    continue

                with open(log_path, "w") as log_file:
                    process = subprocess.run(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True
                    )

                if process.returncode != 0:
                    print(f"[FAILED] {model_dir}, {method}, {scale}")
                    failed_commands.append((cmd, log_path))
                else:
                    print(f"[DONE] {model_dir}, {method}, {scale}")

                current += 1

    print("=" * 80)
    print("Batch retraining finished.")
    print(f"Total commands: {total}")
    print(f"Failed commands: {len(failed_commands)}")

    if failed_commands:
        print("\nFailed command list:")
        for cmd, log_path in failed_commands:
            print("-" * 80)
            print("Command:")
            print(" ".join(cmd))
            print(f"Log: {log_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()