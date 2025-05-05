# if using Using RTX 4000 series
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

echo "Running LoRA finetuning on CPU for Qwen3-0.6B model"
# This script is for finetuning the Qwen3-0.6B model using LoRA on CPU.


MODEL="Qwen/Qwen3-0.6B" 

DATA="data/alpaca-cleaned"

python finetune_cpu.py \
        --model_name_or_path $MODEL \
        --output_dir output/qwen3_lora_adapter \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --target_modules "q_proj,k_proj,v_proj,up_proj,down_proj" \
        --data_path $DATA \
        --dataset_split "train[:1000]" \
        --dataset_field instruction output \
        --model_max_length 2048 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --save_strategy "steps" \
        --save_steps 50 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "linear" \
        --logging_steps 1 \
        --optim "adamw_torch" \
        --fp16 False