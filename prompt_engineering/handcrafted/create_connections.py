#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
from argparse import ArgumentParser
from random import choice,choices
from typing import List, Iterable, Dict
import itertools
import pandas as pd

from prompt_engineering.gpt import OpenAICompletion
from datamodels.healthindicators import HealthIndicator
from datamodels.governmentbudget import BudgetItem, GovernmentBudget
from datasets.utils_datasets import load_budget_dataset, load_health_indicators_dset
import yaml
import random
import math
"""
    Creates the connections between a list of governmemt budget items and a list of health inidicators
        uses one of the prespecified methods 

    User must pass in two csv files:
        File 1 information on Government Spending: 
            columns aligned to governmentbudget.BudgetItem attributes
            - columns: name, code, final_date, period, description, group_code, group_name

        File 1 information on Health Indicators: 
            columns aligned to healthindicators.HealthIndicator attributes
            - columns: id, name, final_date, period, definition_short, definition_long, group_id, group_name
"""

map_code_prompt = {
        
    # 'short1': 'Give me a yes or no answer to whether #budget_name# affect #health_indicator_name#?',
    'short2': 'Give me a yes or no answer to whether #budget_name# (#budget_group_name#) affect #health_indicator_name#?',
    'short3': 'Give me a yes or no answer to whether #budget_name# (#budget_group_name#) affect #health_indicator_name# (#health_indicator_group_name#)?',

    'short4': 'Give me a yes or no answer to the following question: Does #budget_name# affect #health_indicator_name#?',
    'short5': 'Give me a yes or no answer to the following question: Does #budget_name# (#budget_group_name#) affect #health_indicator_name#?',
    'short6': 'Give me a yes or no answer to the following question: Does #budget_name# (#budget_group_name#) affect #health_indicator_name# (#health_indicator_group_name#)?',

    'short4': 'Give me a yes or no answer to the following question: Does government spending on #budget_name# affect #health_indicator_name#?',
    'short7': 'Give me a yes or no answer to the following question: Does government spending on #budget_name# (#budget_group_name#) affect #health_indicator_name#?',
    'short8': 'Give me a yes or no answer to the following question: Does government spending on #budget_name# (#budget_group_name#) affect #health_indicator_name# (#health_indicator_group_name#)?',    

    'short9': 'Answer my following question with a yes or no: Does #budget_name# affect #health_indicator_name#?',
    'short10': 'Answer my following question with a yes or no: Does #budget_name# (#budget_group_name#) affect #health_indicator_name#?',
    'short11': 'Answer my following question with a yes or no: Does #budget_name# (#budget_group_name#) affect #health_indicator_name# (#health_indicator_group_name#)?',

    'short12': 'Answer my following question with a yes or no: Does government spending on #budget_name# affect #health_indicator_name#?',
    'short13': 'Answer my following question with a yes or no: Does government spending on #budget_name# (#budget_group_name#) affect #health_indicator_name#?',
    'short14': 'Answer my following question with a yes or no: Does government spending on #budget_name# (#budget_group_name#) affect #health_indicator_name# (#health_indicator_group_name#)?' 
}

def parse_args(parent_parser):
    if parent_parser != None:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
    else:
        parser = ArgumentParser()

    parser.add_argument('--dataset_name', type=str, choices=['england'], help='Name of dataset to create connection between budget items and health indicators')
    
    parser.add_argument('--prompt_codes', nargs='+',
        help='The list of templates to experiment with', required=True,
        type=str)
    
    parser.add_argument('--sample', default=1.0, type=float)

    parser.add_argument('--debugging', action='store_true')

    
    args = parser.parse_known_args()[0]

    return args


def main(dataset_name, prompt_codes:List[str], completion_model:OpenAICompletion, debugging=False, sample=1.0):

    # Load budget dataset and health indicator dataset dataframes
    df_budget_items = load_budget_dataset(dataset_name)
    df_health_indicators = load_health_indicators_dset(dataset_name)

    # Choose the templates
    if 'all' in prompt_codes:
        prompt_codes = list( map_code_prompt.keys() )
        li_prompts = map_code_prompt.values()
    else:
        li_prompts =  [map_code_prompt[code] for code in prompt_codes]
    dict_prompt_template = { code: PromptTemplate(prompt) for code,prompt in zip(prompt_codes, li_prompts) }

    # For each budget_item predict whether it affects the related Health Item
    for code, prompt_template in dict_prompt_template.items():

        # Some budget_items and health indicators are identical with respect to the attributes used by specific PromptTemplate
        li_budget_items_unique = prompt_template.select_unique_budget_items( df_budget_items ).to_dict('records')
        li_health_indicators_unique = prompt_template.select_unique_health_indicators( df_health_indicators ).to_dict('records')
        
        if sample != 1.0:
            random.seed( 24 )
            li_budget_items_unique = random.sample(li_budget_items_unique, int(len(li_budget_items_unique)*math.sqrt(sample) ) )
            li_health_indicators_unique = random.sample(li_health_indicators_unique, int(len(li_health_indicators_unique)*math.sqrt(sample)) )


        # Creating The Prompts
        li_prompts = [None]*len(li_budget_items_unique)*len(li_health_indicators_unique)
        for idx, (dict_budget_item, dict_health_indicator) in enumerate(itertools.product( li_budget_items_unique, li_health_indicators_unique )):
                        
            li_prompts[idx] = prompt_template.generate_prompt( 
                budget_name=dict_budget_item['name'],
                health_indicator_name = dict_health_indicator['name'],
                budget_group_name=dict_budget_item.get('group_name',''),
                health_indicator_group_name=dict_health_indicator.get('group_name','')
                )

        #TODO: Add code which checks whether search has been done before and if there is an answer already in save csv file / dataset class

        # Get completions
        if debugging:
            li_completions = choices(['yes','no'], k=len(li_prompts))
        else:
            li_completions =  completion_model.get_completions_batched(li_prompts)

        
        # Creating a GovernemntBudget Object
        li_budget_item = []*len(li_budget_items_unique)
        li_health_indicators = [ HealthIndicator(**health_indicator) for health_indicator in li_health_indicators_unique  ]
        count_health_indicator = len(li_health_indicators_unique)
        
        for idx, budget_item_dict in enumerate(li_budget_items_unique):
            
            # Slice for preds of a specific budget item affecting each health indicators
            s_idx = idx*count_health_indicator
            e_idx = (idx+1)*count_health_indicator
            
            # Filtering for the health indicators which budget_item_dict was predicted to have an effect
            li_budget_item_preds = li_completions[s_idx:e_idx] 
            budget_item_connected_health_indicators = [ hi for hi, pred in  zip(li_health_indicators, li_budget_item_preds) if 'yes' in pred.lower() ]

            budget_item_obj = BudgetItem(
                **budget_item_dict,
                connetcted_health_indicators = budget_item_connected_health_indicators
            )
            li_budget_item.append(budget_item_obj)

        # Creating Government Budget Object - Represents a NoSql data model
        government_budget = GovernmentBudget(li_budget_item = li_budget_item )

        government_budget.info = {
            'prompt_code':code,
            'prompt_template': prompt_template.template_text,
            'dataset_name':dataset_name,
        }
        government_budget.info.update(completion_model.__dict__)

        government_budget.to_file( os.path.join('methods','2_0','matchings', f"{dataset_name}_promptcode_{code}.pkl" ) ) 
        government_budget.to_csv( os.path.join('methods','2_0','matchings', f"{dataset_name}_promptcode_{code}.csv" ) )
        government_budget.to_yaml( os.path.join('methods','2_0','params', f"{dataset_name}_promptcode_{code}.csv" ) )
        # save args

    return True

class PromptTemplate():

    def __init__(self, template_text ):
        self.template_text = template_text

        self.bool_budget_group_name = '#budget_group_name#' in self.template_text
        self.bool_health_indicator_group_name = '#health_indicator_group_name#' in self.template_text

    def generate_prompt(self, budget_name:str, health_indicator_name:str, budget_group_name:str='', health_indicator_group_name:str='' ) -> str:
        
        prompt = self.template_text.replace('#budget_name#',budget_name).replace('#health_indicator_name#',health_indicator_name)
        
        if self.bool_budget_group_name:
            if budget_group_name != '':
                prompt = prompt.replace('#budget_group_name#', budget_group_name)
            else:
                prompt = prompt.replace(' #budget_group_name# ', '')

        if self.bool_health_indicator_group_name:
            if health_indicator_group_name != '':
                prompt = prompt.replace('#health_indicator_group_name#',health_indicator_group_name)
            else:
                prompt = prompt.replace(' #health_indicator_group_name# ', '')

        return prompt
        
    def generate_prompt_batch(self, li_budget_item:List[Dict], li_health_indicators:List[Dict]) -> List[str]:
        
        raise NotImplementedError
        
        li_prompt = [None]*len(li_budget_name)*len(li_health_indicator_name)

        for idx,(bn,hi) in enumerate( itertools.product( li_budget_name, li_health_indicator_name) ):
            prompt = self.template_text.replace('#budget_name#',bn).replace('#health_indicator',hi)        
            li_prompt[idx] = prompt
        
        return li_prompt
    
    def select_unique_budget_items(self, budget_items: pd.DataFrame  ) -> pd.DataFrame:
        """
            Check which budget
        """
        attrs_to_filter_on = ['name'] + ['group_name']*self.bool_budget_group_name

        budget_items = budget_items.drop_duplicates(attrs_to_filter_on,  keep='first', inplace=False, ignore_index=True )

        return budget_items

    def select_unique_health_indicators(self, health_indicators: pd.DataFrame ) -> pd.DataFrame:
        """
            Check which health indicators are unique w.r.t the prompt keys this prompt uses
        """
        attrs_to_filter_on = ['name'] + ['group_name']*self.bool_health_indicator_group_name

        health_indicators = health_indicators.drop_duplicates(subset=attrs_to_filter_on,  keep='first', inplace=False, ignore_index=True )

        return health_indicators
        

if __name__ == '__main__':

    parent_parser = ArgumentParser(add_help=False, allow_abbrev=False)

    args = parse_args(parent_parser)
    args_openai_model = OpenAICompletion.parse_args(parent_parser)
    
    completion_model = OpenAICompletion(**vars(args_openai_model))
    
    main(**vars(args), completion_model=completion_model)
