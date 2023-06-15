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
# &
# Testing models with open-ended prompt style w/ categories_rules and categories_perplexity based parsing and binary weight edge value


if [[ $num_devices -eq 2 ]]; then
  # If two numbers are entered
  (
    export CUDA_VISIBLE_DEVICES=${devices[0]}

    # python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style yes_no \
    #     --parse_style rules --effect_type directly --edge_value binary_weight \
    #     --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
    #     --k_shot_example_dset_name_b2i spot --local_or_remote local \
    #     --batch_size 6 --save_output

    # python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style open \
    #     --parse_style categories_perplexity --effect_type directly --edge_value distribution \
    #     --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
    #     --k_shot_example_dset_name_b2i spot --local_or_remote local \
    #     --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style yes_no \
        --parse_style rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style yes_no \
        --parse_style rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

  ) &

  (
    export CUDA_VISIBLE_DEVICES=${devices[1]}
    
    sleep 10

    # python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style open \
    #     --parse_style categories_rules --effect_type directly --edge_value binary_weight \
    #     --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
    #     --k_shot_example_dset_name_b2i spot --local_or_remote local \
    #     --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style open \
        --parse_style categories_perplexity --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style open \
        --parse_style categories_rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output
  ) &
  
  wait
  export CUDA_VISIBLE_DEVICES=${devices[0]},${devices[1]}
  sleep 10
  python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

  python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style open \
        --parse_style categories_rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

  
  python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style open \
        --parse_style categories_perplexity --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

  python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

else
    # If one number is entered
    export CUDA_VISIBLE_DEVICES=${devices[0]}

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style yes_no \
        --parse_style rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style open \
        --parse_style categories_rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style open \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/vicuna-7B-1.1-HF --exp_name vic7b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output
    

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style open \
        --parse_style categories_perplexity --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style yes_no \
        --parse_style rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style open \
        --parse_style categories_rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name TheBloke/Wizard-Vicuna-13B-Uncensored-HF --exp_name wizvic13b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style open \
        --parse_style categories_rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style open \
        --parse_style categories_perplexity --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style yes_no \
        --parse_style rules --effect_type directly --edge_value binary_weight \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 4 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style categorise \
        --parse_style perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

    python3 ./prompt_engineering/langchain/predict.py --llm_name timdettmers/guanaco-33b-merged --exp_name guanco33b --prompt_style cot \
        --parse_style categories_perplexity --effect_type directly --edge_value distribution \
        --input_file ./data/spot/spot_indicator_mapping_table_test.csv \
        --k_shot_example_dset_name_b2i spot --local_or_remote local \
        --batch_size 6 --save_output

fi
