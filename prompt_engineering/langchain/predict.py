"""
    This file provides a LangChain based approach to using a LLM to determine the weights of edges in a graph, 
        where the graph nodes represent government budget items and socioeconomic/health indicators.

"""

import os,sys
# This experiment produces predictions for a given test set
import torch
import os, sys
## Add path to parent directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
import math
import transformers
from transformers import GenerationConfig
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import random

# import utils_prompteng
from prompt_engineering import utils_prompteng
# import utils_prompteng
import copy 
from functools import reduce
import operator
from collections import Counter
import ujson as json
import yaml
import numpy as np

from .utils import HUGGINGFACE_MODELS
# NOTE: when edge value is 0/1 then use majority_vote, when edge_value is float use average for aggregation_method


def main(
        lm_name:str,
        finetuned:bool,
        prompt_style:str,
        parse_style:str,
        k_shot:int,
        ensemble_size:int,
        effect_order:str, 

        input_csv:str,
        ):
    

    # Create Prompt Builder & Prediction Generator
    # prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset_records, directly_or_indirectly,  tokenizer )
    # prediction_generator = PredictionGenerator(model, tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method, deepspeed_compat=False)

    # Create prediction set

def load_model( lm_name:str, finetuned:bool):
    

    if finetuned:
        assert lm_name in HUGGINGFACE_MODELS, f"Loading of a finetuned model is only available for models provided by HUGGINGFACE_MODELS, this includes {HUGGINGFACE_MODELS}"

        model = transformers.AutoModelForCausalLM.from_pretrained( nn_name, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
    
    pass

def load_data():
    pass

def prepare_data():
    pass

def prepare_prediction_set():
    pass

def generate_predictions():
    pass

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--lm_name', type=str, default='EleutherAI/gpt-j-6B', choices=['gpt-3.5-turbo-030','EleutherAI/gpt-j-6B'] )
    parser.add_argument('--finetuned', action='store_true', default=False, help='Indicates whether a finetuned version of nn_name should be used' )
    parser.add_argument('--prompt_style',type=str, choices=['yes_no','open' ], default='open', help='Style of prompt' )


    parser.add_argument('--parse_style', type=str, choices=['rule_based','perplexity', 'generation' ], default='perplexity', help='How to convert the output of the model to a Yes/No Output' )

    parser.add_argument('--k_shot', type=int, default=2, help='Number of examples to use for each prompt. Note this number must respect the maximum length allowed by the language model used' )
    parser.add_argument('--ensemble_size', type=int, default=2 )
    parser.add_argument('--effect_order', type=str, default='arbitrary', choices=['arbitrary', '1st', '2nd'], help='The degree of orders which we require from our model' )
    

    parser.add_argument('--aggregation_method', type=str, default='majority_vote', choices=['majority_vote', 'aggregate '], help='The method used to aggregate the results of the ensemble.' )
    parser.add_argument('--dset_name',type=str, default='spot', choices=['spot','england'] )
    parser.add_argument('--batch_size', type=int, default=1 )

    parser.add_argument('--debugging', action='store_true', default=False, help='Indicates whether the script is being run in debugging mode')

    
    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))