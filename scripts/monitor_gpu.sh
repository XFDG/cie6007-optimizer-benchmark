#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
training_pid=$(cat logs/experiments.pid)
echo 'timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu' > logs/gpu_telemetry.csv
while kill -0 "$training_pid" 2>/dev/null; do
 nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits >> logs/gpu_telemetry.csv
 sleep 2
done
