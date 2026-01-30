#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU monitor logger (nvidia-smi)")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--log", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log) if args.log else Path("logs") / "gpu_monitor.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    query = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as f:
            f.write(
                "timestamp,index,name,utilization_gpu,utilization_memory,memory_used_mb,memory_total_mb,temperature_gpu\n"
            )

    print(f"[INFO] Logging GPU stats to {log_path}")

    while True:
        try:
            output = subprocess.check_output(query, text=True)
            lines = output.strip().splitlines()
            with log_path.open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        except subprocess.CalledProcessError:
            now = datetime.now().isoformat()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{now},-1,unavailable,0,0,0,0,0\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
