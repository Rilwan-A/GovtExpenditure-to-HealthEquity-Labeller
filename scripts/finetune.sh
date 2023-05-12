#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export HF_HOME="/mnt/Data1/akann1w0w1ck/AlanTuring/.cache"
export TRANSFORMERS_CACHE="/mnt/Data1/akann1w0w1ck/.cache/transformers"

# run the finetuning script for EleutherAI/gpt-j-6B
python3 ./prompt_engineering/finetune/finetune.py --exp_name gptj --nn_name EleutherAI/gpt-j-6B \
        --dir_ckpt prompt_engineering/finetune/ckpt "--dir_data", "./data/finetune/preprocessed" \
        --max_epochs 10 --batch_size 6 --batch_size_inf 8 --accumulate_grad_batches 6 \
        --num_workers 2 --devices 2 --strategy deepspeed_stage_3_offload \
        --optimizer DeepSpeedCPUAdam --val_task spot_alignment --k_shot 0 \
        --prompt_style yes_no --parse_output_method language_model_perplexity \
        --directly_or_indirectly indirectly \
        --lr 1e-5 --freeze_layers 21
        
        