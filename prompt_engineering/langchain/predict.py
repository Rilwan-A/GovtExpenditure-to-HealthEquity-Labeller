"""
    This file provides a LangChain based approach to using a LLM to determine the weights of edges in a graph, 
        where the graph nodes represent government budget items and socioeconomic/health indicators.

"""

import os,sys
# This experiment produces predictions for a given test set

## Add path to parent directory to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
import math

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
import numpy as np

import langchain
from langchain import LLMChain, PromptTemplate
from importlib.lazy import lazy_import
from .utils import HUGGINGFACE_MODELS, OPENAI_MODELS

from prompt_engineering.langchain.utils import load_annotated_examples
# TODO: convert to lazy loading later - check for

# # Lazy imports of modules depending on the model to be used
# #region chatOpenAI imports
# ChatOpenAI = lazy_import("langchain.chat_models", "ChatOpenAI")
# ChatPromptTemplate = lazy_import("langchain.prompts.chat", "ChatPromptTemplate")
# SystemMessagePromptTemplate = lazy_import("langchain.prompts.chat", "SystemMessagePromptTemplate")
# AIMessagePromptTemplate = lazy_import("langchain.prompts.chat", "AIMessagePromptTemplate")
# HumanMessagePromptTemplate = lazy_import("langchain.prompts.chat", "HumanMessagePromptTemplate")
# AIMessage = lazy_import("langchain.schema", "AIMessage")
# HumanMessage = lazy_import("langchain.schema", "HumanMessage")
# SystemMessage = lazy_import("langchain.schema", "SystemMessage")
# #endregion

#region HuggingFace imports
# HuggingFaceHub = lazy_import("langchain.llms", "HuggingFaceHub")
from  langchain.chat_models import ChatOpenAI
from  langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from  langchain.schema import AIMessage, HumanMessage, SystemMessage
from  langchain.llms import HuggingFaceHub

# endregion


from prompt_engineering.utils_prompteng import PromptBuilder
from .utils import PredictionGeneratorRemoteLM
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

        local_or_remote:str='remote',
        api_key:str = None):
    
    # Load LLM
    llm =  load_llm(lm_name, finetuned, local_or_remote, api_key)

    # Load Annotated Examples to use in K-Shot context for Prompt
    annotated_examples_b2i = load_annotated_examples(k_shot_example_dset_name_b2i, relationship_type='budget_item_to_indicator')
    annotated_examples_i2i = None if effect_order != '2nd' else load_annotated_examples(k_shot_example_dset_name_i2i, relationship_type='indicator_to_indicator')

    # Create Prompt Builder & Prediction Generator
    # when modelling second order effects we include indicator to indicator weights    
    prompt_builder_b2i = PromptBuilder(prompt_style, k_shot_b2i,
                                        ensemble_size, annotated_examples_b2i, 
                                        effect_order,
                                        relationship='budget_item_to_indicator')
    prompt_builder_i2i = None if effect_order != '2nd' else PromptBuilder(prompt_style, k_shot_i2i,
                                                                           ensemble_size, annotated_examples_i2i, 
                                                                           effect_order,
                                                                           relationship='indicator_to_indicator')

    if local_or_remote == 'remote':
        prediction_generator = PredictionGeneratorRemoteLM(llm, prompt_style, ensemble_size,
                                                aggregation_method, parse_output_method, deepspeed_compat=False)
    else:
        raise NotImplementedError("Local LMs not yet supported")

    # run predictions
    batch_prompt_ensembles_b2i, batch_pred_ensembles_b2i, batch_pred_ensembles_parsed_b2i, batch_pred_agg_b21 = predict_batches(llm, prompt_builder_b2i, prompt_builder_i2i, prediction_generator, input_csv, parse_style, api_key)
    if effect_order == '2nd':
        batch_prompt_ensembles_i2i, batch_pred_ensembles_i2i, batch_pred_ensembles_parsed_i2i, batch_pred_agg_i2i = predict_batches(llm, prompt_builder_b2i, prompt_builder_i2i, prediction_generator, input_csv, parse_style, api_key)
    
    # format predictions

    # email to user
       

def load_llm( lm_name:str, finetuned:bool, local_or_remote:str='remote', api_key:str = None, prompt_style:str = 'yes_no'):
    
    assert local_or_remote in ['local', 'remote'], f"local_or_remote must be either 'local' or 'remote', not {local_or_remote}"
    if local_or_remote == 'remote': assert api_key is not None, f"api_key must be provided if local_or_remote is 'remote'"
    
    # TODO: All models used done on Huggingface Hub
    # TODO: if user wants to run it faster they can download the model from Huggingface Hub and load it locally and use the predict step with different --local_or_remote set to local
    if local_or_remote == 'local':
        raise NotImplementedError
            
    elif local_or_remote == 'remote':
        #NOTE: max_tokens is currently dependent on prompt_style, it maybe should be depndent on parse_style

        if lm_name in OPENAI_MODELS:
            
            llm = ChatOpenAI(
                model_name=lm_name if not finetuned else f"rilwanade/{lm_name}",
                openai_api_key=api_key,
                max_tokens = 5 if prompt_style == 'yes_no' else 100 )            
        
        if lm_name in HUGGINGFACE_MODELS:
            llm = HuggingFaceHub(repo_id=lm_name, huggingfacehub_api_token=api_key, model_kwargs={ 'max_new_tokens': 5 if prompt_style == 'yes_no' else 100, 'do_sample':False } )


    return llm 

def prepare_data():
    pass

def prepare_prediction_set():
    pass

def generate_predictions():
    pass

def predict_batches(batch, prompt_builder:PromptBuilder, prediction_generator:PredictionGenerator ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[str]]:

    # Creating Predictions for each row in the test set
    li_prompt_ensemble = []
    li_pred_ensemble = []
    li_pred_ensemble_parsed = []
    preds_agg = []

    for idx, batch in enumerate(test_dset_records):

        # Create prompts
        batch_prompt_ensembles = prompt_builder(batch)
        
        # Generate predictions
        batch_pred_ensembles, batch_pred_ensembles_parsed = prediction_generator.predict(batch_prompt_ensembles)

        # Aggregate ensembles into predictions
        batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles_parsed)


        # Extract predictions from the generated text
        li_prompt_ensemble.append(batch_prompt_ensembles)  # type: ignore
        li_pred_ensemble.append( batch_pred_ensembles ) # type: ignore
        li_pred_ensemble_parsed.append( batch_pred_ensembles_parsed ) # type: ignore
        preds_agg.append(batch_pred_agg) # type: ignore


    # Create prompts
    batch_prompt_ensembles = prompt_builder(batch)
    
    # Generate predictions
    batch_pred_ensembles, batch_pred_ensembles_parsed = prediction_generator.predict(batch_prompt_ensembles)

    # Aggregate ensembles into predictions
    batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles_parsed)

    return batch_prompt_ensembles, batch_pred_ensembles, batch_pred_ensembles_parsed, batch_pred_agg

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