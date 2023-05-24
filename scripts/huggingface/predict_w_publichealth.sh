#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export HF_HOME="/mnt/Data1/akann1w0w1ck/AlanTuring/.cache"
export TRANSFORMERS_CACHE="/mnt/Data1/akann1w0w1ck/.cache/transformers"

# Runs experiments which Test Various Prompt Engineering Methods of comparing the ability of GPT-J to guess which local government budget items affect which 
# socio-economic indicators

# Variants where model aims to output Yes / No, with rule-based parsing of output into Yes/No/Nan
# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'rule_based' \
#     --k_shot "0" --ensemble_size "1" \
#     --batch_size "2" &&

# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'rule_based' \
#     --k_shot "2" --ensemble_size "2" \
#     --batch_size "1" &&

# # Yes / No output variants with Language Model Output Parsing (perplexity style) to convert output to Yes/No/Nan
# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'perplexity' \
#     --k_shot "0" --ensemble_size "1" \
#     --batch_size "2" &&

# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'perplexity' \
#     --k_shot "2" --ensemble_size "2" \
#     --batch_size "1" &&


# # Yes / No Variants with Language Model Output Parsing (generation style)
# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'generation' \
#     --k_shot "0" --ensemble_size "1" \
#     --batch_size "2" &&


# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'yes_no' --parse_output_method 'generation' \
#     --k_shot "2" --ensemble_size "1" \
#     --batch_size "1" &&


# # Open-ended Variants with Language Model Output Parsing (perplexity style)
# python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
#     --prompt_style 'open' --parse_output_method 'perplexity' \
#     --k_shot "0" --ensemble_size "1" \
#     --batch_size "1" &&


python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
    --prompt_style 'open' --parse_output_method 'perplexity' \
    --k_shot "2" --ensemble_size "1" \
    --batch_size "1" &&


# Open-ended Variants with Language Model Output Parsing (perplexity style) and PileStackOverflow formatting
python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
    --prompt_style 'pilestackoverflow_open' --parse_output_method 'perplexity' \
    --k_shot "0" --ensemble_size "1" \
    --batch_size "1" &&

python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
    --prompt_style 'pilestackoverflow_open' --parse_output_method 'perplexity' \
    --k_shot "2" --ensemble_size "1" \
    --batch_size "1" &&

# Open-ended Variants with Language Model Output Parsing (generation) and PileStackOverflow formatting
python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
    --prompt_style 'pilestackoverflow_open' --parse_output_method 'generation' \
    --k_shot "0" --ensemble_size "1" \
    --batch_size "1" &&

python3 ./prompt_engineering/huggingface/predict.py --nn_name EleutherAI/gpt-neox-20b --dset_name 'spot' \
    --prompt_style 'pilestackoverflow_open' --parse_output_method 'generation' \
    --k_shot "2" --ensemble_size "1" \
    --batch_size "1"