import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from prompt_engineering.utils_prompteng import map_relationship_system_prompt, map_relationship_promptsmap, map_relationship_sppywlg, create_negative_examples, perplexity
import random

import copy
import numpy as np
import pandas as pd

HUGGINGFACE_MODELS = [ 'mosaicml/mpt-7b-instruct' ]
OPENAI_MODELS = ['gpt-3.5-turbo-030', 'gpt-4']
from collections import Counter
import logging

class PredictionGenerator():
    """
        NOTE: This prediction generator currently only designed for models tuned on instruct datasets that are remote
    """
    def __init__(self, llm,  
                 prompt_style:str,
                  ensemble_size:int,
                  edge_value:str="binary weight", # binary weight or float weight or float pair
                  parse_style:str='rule_based',
                  relationship:str='budget_item_to_indicator',
                  local_or_remote='local',
                  deepspeed_compat:bool=False ):
        
        # Can not get model logits scores from ChatOpenAI
        if isinstance(llm, langchain.chat_models.ChatOpenAI) and parse_style == 'language_model_perplexity': raise ValueError("Can not get model logits scores from ChatOpenAI")
        if parse_style == 'language_model_perplexity': assert local_or_remote == 'local', "Can not get model logits scores from remote models"

        # Restrictions on combinations of parse style and edge value
        if edge_value in ['float weight','distribution'] and parse_style != 'language_model_perplexity': 
            if ensemble_size == 1: raise ValueError(f"Can not get a float edge value with ensemble size 1 and parse_style:{parse_style}.\
                                                         To use ensemble size 1, please use parse_style='language_model_perplexity'.\
                                                         Alternatively use ensemble size > 1, ")
            
        self.llm = llm

        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_style = parse_style
        self.relationship = relationship
        self.edge_value   = edge_value
        self.local_or_remote = local_or_remote


    def predict(self, li_li_prompts:list[list[str]])->tuple[list[list[str]], list[list[str]]]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        
        # Generate predictions
        li_li_preds = []
        for li_prompts in li_li_prompts:
            
            if isinstance(self.llm, langchain.chat_models.base.BaseChatModel):
                batch_messages = [
                    [ SystemMessage(content=map_relationship_system_prompt[self.relationship]),
                        HumanMessage(content=prompt) ]
                        for prompt in li_prompts]
                
                outputs = self.llm.generate(batch_messages)
                li_preds = [ chatgen.text for chatgen in outputs.generations ]

            elif isinstance(self.llm, langchain.llms.base.LLM):
            
               outputs = self.llm.generate( prompts= [ map_relationship_system_prompt[self.relationship] + '\n' + prompt for prompt in li_prompts ] )
               li_preds = [ chatgen.text for chatgen in outputs.generations ]

            li_li_preds.append(li_preds)

                
        # Parse {'Yes':prob_yes, 'No':prob_no, 'Nan':prob_nan } from the predictions
        if self.parse_style == 'rule_based':
            li_li_preds_parsed = [ self.parse_yesno_with_rules(li_predictions) for li_predictions in li_li_preds]

        elif self.parse_style == 'language_model_perplexity':
            li_li_preds_parsed = [ self.parse_yesno_with_lm_perplexity(li_predictions) for li_predictions in li_li_preds]
        
        elif self.parse_style == 'language_model_generation':
            li_li_preds_parsed = [ self.parse_yesno_with_lm_generation(li_predictions) for li_predictions in li_li_preds]
        
        else:
            raise ValueError(f"parse_style {self.parse_style} not recognized")

        return li_li_preds, li_li_preds_parsed

    def parse_yesno_with_rules(self, li_predictions:list[str])->list[dict]:
        
        li_preds = ['NA']*len(li_predictions)

        # Parse yes/no from falsetrue
        for idx in range(len(li_predictions)):
            
            prediction = li_predictions[idx].lower()

            if 'yes' in prediction:
                prediction = {'Yes':1, 'No':0, 'NA':0}
            
            elif 'no' in prediction:
                prediction = {'Yes':0, 'No':1, 'NA':0}
            
            elif any( ( neg_phrase in prediction for neg_phrase in ['not true','false', 'is not', 'not correct', 'does not', 'can not', 'not'])):
                prediction = {'Yes':0, 'No':1, 'NA':0}

            else:
                prediction = {'Yes':0, 'No':0, 'NA':0 }
        
            li_preds[idx] = prediction
                   
        return li_predictions
               
    def parse_yesno_with_lm_generation(self, li_predictions:list[str])->list[dict]:
        
        # Template to prompt language llm to simplify the answer to a Yes/No output
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_parse_yesno_from_answer']
        template = copy.deepcopy( random.choice(li_prompts) )
            

        # Create filled versions of the template with each of the predictions
        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]
        
        # Generate prediction
        if isinstance(self.llm, langchain.chat_models.base.BaseChatModel):
            batch_messages = [
                [ SystemMessage(content=map_relationship_system_prompt[self.relationship] ),
                    HumanMessage(content=prompt) ]
                    for prompt in li_filledtemplate]
            
            outputs = self.llm.generate(batch_messages)
            li_preds = [ chatgen.text for chatgen in outputs.generations ]


        elif isinstance(self.llm, langchain.llms.base.LLM):
        
            outputs = self.llm.generate( prompts= li_prompts )
            li_preds = [ chatgen.text for chatgen in outputs.generations ]


        # Parse Yes/No/Na from the prediction
        li_preds = [ {'Yes':1,'No':0, 'NA':0 } if 'affirm' in pred.lower() else {'Yes':0,'No':1, 'NA':0 } if 'negat' in pred.lower() else {'Yes':0,'No':0, 'NA':1 } for pred in li_preds]

        return li_preds
    
    def parse_yesno_with_lm_perplexity(self, li_predictions:list[str])->list[str]:
        # Get average perplexity of text when sentence is labelled Negation vs when it is labelled Affirmation.
        # NOTE: the perpleixty is calculated as an average on the whole text, not just the answer. Therefore, we rely on the
        #       the fact that 'Negation". and "Affirmation". both contain the same number of tokens

        # Template to prompt language llm to simplify the answer to a Yes/No output
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_parse_yesno_from_answer']
        template = copy.deepcopy( random.choice(li_prompts) )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]

        # For each fill template, create 3 filled versions with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = ['Negation', 'Affirmation']
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answers ] for filledtemplate in li_filledtemplate ]
        li_filledtemplates_with_answers = sum(li_li_filledtemplates_with_answers,[])

        # Get the perplexity of each of the filled templates
        # return a flattened set of perplexities
        li_perplexity = perplexity(
            li_filledtemplates_with_answers, 
            self.llm.pipeline.model , 
            self.llm.pipeline.tokenizer,
            batch_size=6, 
            deepspeed_compat = self.deepspeed_compat ) 

        # # For each set of filltemplates get the index of the answer with the lowest perplexity
        # li_idx = [ np.argmin(li_perplexity[idx:idx+len(answers)]) for idx in range(0,len(li_perplexity),len(answers)) ]
        # li_predictions = [ 'No' if idx==0 else 'Yes' for idx in li_idx ]
        
        li_preds = [ {'Yes': li_perplexity[ idx + answers.index('Affirmation') ] , 'No': li_perplexity[ idx + answers.index('Negation') ] , } for idx in range(0,len(li_perplexity),len(answers)) ]

        return li_preds
        
    def aggregate_predictions(self, li_li_predictions:list[list[dict]])->list[int|float|dict]:
        
        """            
            
            For Each prediction we have a list of samples.
                Each sample is a dictionary with keys {'Yes':pred_yes, 'No':pred_no, 'NA':pred_na}"

            edge_value: 'binary weight': for a prediction p with n samples, returns 1 if Yes is the modal prediction

            edge_value: 'float weight': for a prediction p with n samples, returns the average relative weight of Yes for each sample
            
            edge_value: 'distribution': for a prediction p with n samples, returns the average relative weights of Yes/No/NA for each sample

            output format: list of dictionaries with keys {'Yes':pred_yes, 'No':pred_no, 'NA':pred_na}. Reduced to 1 sample per prediction
        """
        
        if self.edge_value == 'binary weight':
            
            # For each prediction (list of samples), return 1 if Yes is the model prediction
            def is_yes_mode(li_dict_pred) -> int:
                li_argmax_pred = [ max(d, key=d.get) for d in li_dict_pred ]
                most_common_pred = Counter(li_argmax_pred).most_common(1)[0][0]                
                return 1 if most_common_pred == 'Yes' else 0
           
            li_pred_agg = [ is_yes_mode(li_dict_pred) for li_dict_pred in li_li_predictions ]
        
        elif self.edge_value == 'float weight':
             
            def avg_relative_y(li_dict_pred):
                li_relative_yes = [ d['Yes'] / sum(d.values()) for d in li_dict_pred ]
                avg_relative_yes = sum(li_relative_yes) / len(li_relative_yes)
                return avg_relative_yes
            
            li_pred_agg = [ avg_relative_y(li_dict_pred) for li_dict_pred in li_li_predictions ]
        
        elif self.edge_value == 'distribution':

            def distribution(li_dict_pred):
                li_relative_yes = [ d['Yes'] / sum(d.values()) for d in li_dict_pred ]
                li_relative_no = [ d['No'] / sum(d.values()) for d in li_dict_pred ]
                li_relative_na = [ d['NA'] / sum(d.values()) for d in li_dict_pred ]
                avg_relative_yes = sum(li_relative_yes) / len(li_relative_yes)
                avg_relative_no = sum(li_relative_no) / len(li_relative_no)
                avg_relative_na = sum(li_relative_na) / len(li_relative_na)
                return {'Yes':avg_relative_yes, 'No':avg_relative_no, 'NA':avg_relative_na}

            li_pred_agg = [ distribution(li_dict_pred) for li_dict_pred in li_li_predictions ]

        else:
            raise NotImplementedError(f"Aggregation method {self.aggregation_method} not implemented")
        
        return li_pred_agg

def load_annotated_examples(k_shot_example_dset_name:str, 
                            random_state_seed:int=10, 
                            relationship_type:str='budget_item_to_indicator') -> list[dict]:
    
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
        dset = create_negative_examples(dset, random_state=random_state )

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
