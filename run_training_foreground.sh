#!/bin/bash
# Run full FP16 training with monitoring

# Create logs directory
mkdir -p logs

# Set timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/full_training_${TIMESTAMP}.log"

echo "Starting full FP16 Mistral-7B training..."
echo "Log file: $LOG_FILE"
echo "GPU: NVIDIA L4 (24GB)"
echo ""

# Activate virtual environment
source mistral_env/bin/activate

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# Start training
echo "Training started at $(date)" | tee $LOG_FILE
echo "Running: python train_full_fp16.py" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run training with output to both console and log
python train_full_fp16.py 2>&1 | tee -a $LOG_FILE &

# Get the PID
TRAIN_PID=$!
echo "Training process PID: $TRAIN_PID"

# Function to monitor GPU
monitor_gpu() {
    while kill -0 $TRAIN_PID 2>/dev/null; do
        echo -e "\n=== GPU Status at $(date) ===" >> $LOG_FILE
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> $LOG_FILE
        sleep 60
    done
}

# Start GPU monitoring in background
monitor_gpu &
MONITOR_PID=$!

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo -e "\nTraining completed at $(date)" | tee -a $LOG_FILE
echo "Exit code: $TRAIN_EXIT_CODE" | tee -a $LOG_FILE

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "\nTraining successful!" | tee -a $LOG_FILE
    echo "Next steps:" | tee -a $LOG_FILE
    echo "1. Check the model in ./full_finetuned_model/" | tee -a $LOG_FILE
    echo "2. Run quantization: python quantize_model.py ./full_finetuned_model mistral-full-trained" | tee -a $LOG_FILE
    echo "3. Register with Ollama using the generated Modelfile" | tee -a $LOG_FILE
else
    echo -e "\nTraining failed! Check the log for errors." | tee -a $LOG_FILE
fi