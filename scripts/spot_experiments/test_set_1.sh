#!/bin/bash

export HF_HOME="/mnt/Data1/akann1w0w1ck/AlanTuring/.cache"
export TRANSFORMERS_CACHE="/mnt/Data1/akann1w0w1ck/.cache/transformers"

# Parsing the CUDA_VISIBLE_DEVICES argument
devices=(${1//,/ })
num_devices=${#devices[@]}

if [[ $num_devices -gt 2 || $num_devices -eq 0 ]]; then
  echo "Please provide one or two GPU IDs separated by a comma without spaces"
  exit 1
fi

# Testing models with yes_no prompt style w/ rule based parsing and binary weight edge value
if [[ $num_devices -eq 2 ]]; then
  # If two numbers are entered
  (
    export CUDA_VISIBLE_DEVICES=${devices[0]}
    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --prompt_style yes_no \
        --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
        --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
        --batch_size 24 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/guanaco-65B-HF --prompt_style yes_no \
        --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
        --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
        --batch_size 2 --save_output
  ) &

  (
    export CUDA_VISIBLE_DEVICES=${devices[1]}
    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/wizard-vicuna-13B-HF --prompt_style yes_no \
        --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
        --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
        --batch_size 16 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --prompt_style yes_no \
        --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
        --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
        --batch_size 5 --save_output
  ) &
else
  # If one number is entered
  export CUDA_VISIBLE_DEVICES=${devices[0]}
  python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --prompt_style yes_no \
      --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
      --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
      --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
      --batch_size 15 --save_output

  python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/wizard-vicuna-13B-HF --prompt_style yes_no \
      --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
      --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
      --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
      --batch_size 10 --save_output

  python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --prompt_style yes_no \
      --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
      --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
      --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
      --batch_size 5 --save_output

  python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/guanaco-65B-HF --prompt_style yes_no \
      --parse_style rule_based --ensemble_size 1 --effect_type directly --edge_value binary_weight \
      --input_file ./data/spot/spot_indicator_mapping_table_test.csv --k_shot_b2i 0 --k_shot_i2i 0 \
      --k_shot_example_dset_name_b2i spot --k_shot_example_dset_name_i2i None --local_or_remote local \
      --batch_size 2 --save_output
fi
