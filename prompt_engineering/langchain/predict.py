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

import langchain
from langchain import LLMChain, PromptTemplate

from prompt_engineering.langchain.utils import HUGGINGFACE_MODELS, OPENAI_MODELS
from prompt_engineering.utils_prompteng import PromptBuilder

# NOTE: when edge value is 0/1 then use majority_vote, when edge_value is float use average for aggregation_method


def main(
        lm_name:str,
        finetuned:bool,
        prompt_style:str,
        parse_style:str,
        ensemble_size:int,
        effect_order:str, 

        input_csv:str,

        k_shot_b2i:int=2,
        k_shot_i2i:int=0,

        k_shot_example_dset_name_b2i:str = 'spot',
        k_shot_example_dset_name_i2i:str = None,

        api_key:str = None,
        ):
    

    # Load LLM
    llm =  load_llm(lm_name, finetuned)

    # Load Annotated Examples to use in K-Shot context for Prompt
    annotated_examples_b2i = load_annotated_examples(k_shot_example_dset_name_b2i, relationship_type='budget_item_to_indicator')
    annotated_examples_i2i = None if effect_order != '2nd' else load_annotated_examples(k_shot_example_dset_name_i2i, relationship_type='indicator_to_indicator')

    # Create Prompt Builder & Prediction Generator
    
    prompt_builder_b2i = PromptBuilder(prompt_style, k_shot_b2i, ensemble_size, annotated_examples_b2i,  effect_order )
    
    # when modelling second order effects we include indicator to indicator weights
    prompt_builder_i2i = None if effect_order != '2nd' else PromptBuilder(prompt_style, k_shot_i2i, ensemble_size, annotated_examples_i2i,  effect_order )


    # prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset_records, directly_or_indirectly,  tokenizer )


    # prediction_generator = PredictionGenerator(model, tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method, deepspeed_compat=False)


    # Create prediction set

def load_llm( lm_name:str, finetuned:bool):
    

    if finetuned:
        assert lm_name in HUGGINGFACE_MODELS, f"Loading of a finetuned model is only available for models provided by HUGGINGFACE_MODELS, this includes {HUGGINGFACE_MODELS}"

        from langchain import HuggingFacePipeline
        llm = HuggingFacePipeline(lm_name, device='cuda')
    
    else:
        if lm_name in OPENAI_MODELS:
            from langchain import OpenAIPipeline
            llm = OpenAIPipeline(lm_name, device='cuda')

        elif lm_name in HUGGINGFACE_MODELS:
            from langchain import HuggingFacePipeline
            llm = HuggingFacePipeline(lm_name, device='cuda', load_in_8bit=True)
    
    return llm

def load_annotated_examples(k_shot_example_dset_name) -> list[dict]:
    # Load Annotated Examples to use in K-Shot context for Prompt
    if k_shot_example_dset_name == 'spot':

    elif k_shot_example_dset_name == 'england':
        pass

    elif k_shot_example_dset_name == None:
        annotated_examples = None
    
    return annotated_examples

def load_annotated_examples(k_shot_example_dset_name:str, random_state_seed:int=10, relationship_type:str='budget_item_to_indicator') -> list[dict]:
    
    if k_shot_example_dset_name == 'spot' and relationship_type == 'budget_item_to_indicator':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./data/spot/spot_indicator_mapping_table.csv')
        
        # Remove all rows where 'type' is not 'Outcome'
        dset = dset[dset['type'] == 'Outcome']

        # Creating target field
        dset['label'] = 'Yes'

        # Rename columns to match the format of the other datasets
        dset = dset.rename( columns={'category': 'budget_item', 'name':'indicator' } )

        # Create negative examples
        random_state = np.random.RandomState(random_state_seed)

        # Too vague a budget item and there are only 4 examples of it, we remove it
        dset = dset[ dset['budget_item'] != 'Central' ]

        # create negative examples
        dset = utils_prompteng.create_negative_examples(dset, random_state=random_state )

        # Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
        dset = dset.groupby(['budget_item','label']).filter(lambda x: len(x) > 1)

        li_records = dset.to_dict('records')
    
    elif k_shot_example_dset_name == 'spot' and relationship_type == 'indicator_to_indicator':
        logging.log.warning('Currently, there does not exist any annotated examples for indicator to indicator relationships. Therefore we can not use K-Shot templates for indicator to indicator edge determination. This will be added in the future')        
        li_records = None

    elif k_shot_example_dset_name == 'england':
        logging.log.warning('Currently, the relationshps for England dataset have not yet been distilled. This will be added in the future')        
        li_records = None
    
    else:
        raise ValueError('Invalid dset_name: ' + k_shot_example_dset_name)

    return li_records


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