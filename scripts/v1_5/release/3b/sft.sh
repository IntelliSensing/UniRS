#!/bin/bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


deepspeed --master_port=$((RANDOM + 10000)) --include localhost:0 ./llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /path/to/the/pretrained/model \
    --version v1 \
    --data_mixture geochat_instruct+levir_cc+levir_cc+levir_cc+levir_cc+levir_cc+era_class+era_class+era_class+era_class+era_class+era_class+era_class+era_class+era_class+era_class \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_cc_projector False \
    --tune_single_projector False \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --chg True \
    --chg_type Chg2Cap \
    --from_origin True \
    --cc_n_layers 0 \
    --cc_head 8 \
    --cc_dropout 0.1 \
    --bf16 True \
    --output_dir /checkpoints/save/path/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.5 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
