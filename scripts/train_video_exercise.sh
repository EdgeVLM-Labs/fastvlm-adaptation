#!/bin/bash
# FastVLM Video Fine-tuning Script for Exercise Form Analysis
# Uses the Qwen-based training pipeline with video support

set -e

# Configuration
MODEL_PATH="checkpoints/llava-fastvithd_0.5b_stage3"
DATA_PATH="dataset/fastvlm_train.json"
IMAGE_FOLDER="dataset"
OUTPUT_DIR="models/exercise-video-finetuned"
NUM_VIDEO_FRAMES=4  # Reduced for memory efficiency on 8GB GPU

# Training hyperparameters (optimized for 8GB VRAM)
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MODEL_MAX_LENGTH=2048

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="models/training_log_${TIMESTAMP}.log"
HPARAMS_FILE="models/hyperparameters_${TIMESTAMP}.json"

# Save hyperparameters to JSON
cat > "$HPARAMS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "model_path": "$MODEL_PATH",
  "data_path": "$DATA_PATH",
  "image_folder": "$IMAGE_FOLDER",
  "output_dir": "$OUTPUT_DIR",
  "num_video_frames": $NUM_VIDEO_FRAMES,
  "batch_size": $BATCH_SIZE,
  "gradient_accumulation_steps": $GRAD_ACCUM_STEPS,
  "effective_batch_size": $((BATCH_SIZE * GRAD_ACCUM_STEPS)),
  "learning_rate": "$LEARNING_RATE",
  "num_epochs": $NUM_EPOCHS,
  "model_max_length": $MODEL_MAX_LENGTH,
  "vision_tower": "mobileclip_l_1024",
  "mm_projector_type": "mlp2x_gelu",
  "mm_vision_select_layer": -2,
  "image_aspect_ratio": "pad",
  "bf16": true,
  "tf32": true,
  "gradient_checkpointing": true,
  "weight_decay": 0.0,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine",
  "dataloader_num_workers": 2
}
EOF

echo "=============================================="
echo "FastVLM Video Fine-tuning for Exercise Analysis"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Video frames: $NUM_VIDEO_FRAMES"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Log file: $LOG_FILE"
echo "Hyperparameters: $HPARAMS_FILE"
echo "=============================================="

# Set environment variables for better logging
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

# Run training with output to both console and log file
python -u llava/train/train_mem.py \
    --model_name_or_path "$MODEL_PATH" \
    --version qwen_v2 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_video_frames $NUM_VIDEO_FRAMES \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none 2>&1 | tee "$LOG_FILE"

echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "Hyperparameters saved to: $HPARAMS_FILE"
echo "=============================================="
