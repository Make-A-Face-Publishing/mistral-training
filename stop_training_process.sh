#!/bin/bash
LOG_DIR="./logs"

echo "Stopping training processes..."

if [ -f "$LOG_DIR/train.pid" ]; then
    TRAIN_PID=$(cat "$LOG_DIR/train.pid")
    kill $TRAIN_PID 2>/dev/null
    echo "Stopped training process (PID: $TRAIN_PID)"
    rm "$LOG_DIR/train.pid"
fi

if [ -f "$LOG_DIR/gpu_monitor.pid" ]; then
    GPU_PID=$(cat "$LOG_DIR/gpu_monitor.pid")
    kill $GPU_PID 2>/dev/null
    echo "Stopped GPU monitor (PID: $GPU_PID)"
    rm "$LOG_DIR/gpu_monitor.pid"
fi
