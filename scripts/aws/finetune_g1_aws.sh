#!/usr/bin/env bash
# G1 fine-tuning スクリプト (g5.xlarge: 1x A10G 24GB 向け)
# 前提: setup_ec2.sh 実行済み、~/Isaac-GR00T/ 内で実行
# 使い方: bash scripts/aws/finetune_g1_aws.sh
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-/home/ubuntu/g1_finetune}
DATASET_PATH="examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC"

echo "=== G1 Fine-tuning (g5.xlarge: 1x A10G 24GB) ==="
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# global_batch_size=8 + gradient_accumulation_steps=8 → effective batch = 64
# paged_adamw_8bit: 8-bit quantized optimizer states (~4x less GPU memory)
# gradient_checkpointing: recompute activations instead of storing them
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag UNITREE_G1 \
    --num_gpus 1 \
    --output_dir "$OUTPUT_DIR" \
    --save_total_limit 5 \
    --save_steps 1000 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --optim paged_adamw_8bit \
    --dataloader_num_workers 1 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo ""
echo "=== 完了: $OUTPUT_DIR ==="
