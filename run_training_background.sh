#!/bin/bash
# Run fine-tuning as a background process with logging

# Configuration
LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment
source mistral_env/bin/activate

# Function to monitor GPU usage
monitor_gpu() {
    while true; do
        echo "=== GPU Status at $(date) ===" >> "$LOG_DIR/gpu_monitor_$TIMESTAMP.log"
        nvidia-smi >> "$LOG_DIR/gpu_monitor_$TIMESTAMP.log" 2>&1
        echo "" >> "$LOG_DIR/gpu_monitor_$TIMESTAMP.log"
        sleep 60  # Check every minute
    done
}

# Start GPU monitoring in background
monitor_gpu &
GPU_MONITOR_PID=$!

# Run training with nohup
echo "Starting fine-tuning process..."
echo "Logs will be written to: $LOG_FILE"
echo "GPU monitoring logs: $LOG_DIR/gpu_monitor_$TIMESTAMP.log"

nohup ./mistral_env/bin/python -u train_mistral_instruct.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "Training started with PID: $TRAIN_PID"
echo "GPU monitor PID: $GPU_MONITOR_PID"

# Save PIDs for later management
echo "$TRAIN_PID" > "$LOG_DIR/train.pid"
echo "$GPU_MONITOR_PID" > "$LOG_DIR/gpu_monitor.pid"

# Create status checking script
cat > check_status.sh << 'EOF'
#!/bin/bash
LOG_DIR="./logs"

if [ -f "$LOG_DIR/train.pid" ]; then
    TRAIN_PID=$(cat "$LOG_DIR/train.pid")
    if ps -p $TRAIN_PID > /dev/null; then
        echo "Training is running (PID: $TRAIN_PID)"
        echo "Recent log output:"
        tail -n 20 $(ls -t $LOG_DIR/training_*.log | head -1)
    else
        echo "Training process is not running"
    fi
else
    echo "No training PID file found"
fi
EOF

chmod +x check_status.sh

# Create stop script
cat > stop_training.sh << 'EOF'
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
EOF

chmod +x stop_training.sh

echo ""
echo "Training is running in the background!"
echo "Commands:"
echo "  - Check status: ./check_status.sh"
echo "  - Stop training: ./stop_training.sh"
echo "  - View logs: tail -f $LOG_FILE"
echo ""
