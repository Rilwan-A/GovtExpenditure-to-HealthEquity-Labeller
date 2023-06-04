import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from prompt_engineering.utils_prompteng import map_relationship_system_prompt, map_relationship_promptsmap, map_relationship_sppywlg, create_negative_examples, perplexity, map_llmname_input_format
import random

import copy
import numpy as np
import pandas as pd



#https://old.reddit.com/r/LocalLLaMA/wiki/models#wiki_current_best_choices
HUGGINGFACE_MODELS = [ 
    

    'mosaicml/mpt-7b-chat', 'TheBloke/vicuna-7B-1.1-HF', 'TheBloke/wizard-vicuna-13B-HF',  'timdettmers/guanaco-33b-merged', 'TheBloke/guanaco-65B-HF',

    'TheBloke/wizard-vicuna-13B-GPTQ', 'TheBloke/wizard-vicuna-13B-GPTQ', 'TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g'  ,'TheBloke/guanaco-65B-GPTQ'
    ]

MAP_LOAD_IN_NBIT = {
    
    'eachadea/vicuna-7b-1.1':4,
    'TheBloke/gpt4-x-vicuna-13B-HF': 4,
    'timdettmers/guanaco-33b-merged':4,

    'mosaicml/mpt-7b-chat': 4,
    'TheBloke/wizard-vicuna-13B-HF':4,

}

OPENAI_MODELS = ['gpt-3.5-turbo-030', 'gpt-4']
ALL_MODELS = HUGGINGFACE_MODELS + OPENAI_MODELS
from collections import Counter
import logging

class PredictionGenerator():
    """
        NOTE: This prediction generator currently only designed for models tuned on instruct datasets that are remote
    """
    def __init__(self, llm,  
                 llm_name,
                 prompt_style:str,
                  ensemble_size:int,
                  edge_value:str="binary_weight", # binary_weight or float_weight or float pair
                  parse_style:str='rule_based',
                  relationship:str='budgetitem_to_indicator',
                  local_or_remote='local',
                  deepspeed_compat:bool=False,
                  effect_type:str='directly' ):
        
        # Can not get model logits scores from ChatOpenAI
        if isinstance(llm, langchain.chat_models.ChatOpenAI) and parse_style == 'perplexity': raise ValueError("Can not get model logits scores from ChatOpenAI") #type: ignore
        if parse_style == 'perplexity': assert local_or_remote == 'local', "Can not get model logits scores from remote models"

        # Restrictions on combinations of parse style and edge value
        if edge_value in ['float_weight','distribution'] and parse_style != 'perplexity': 
            if ensemble_size == 1: raise ValueError(f"Can not get a float edge value with ensemble size 1 and parse_style:{parse_style}.\
                                                         To use ensemble size 1, please use parse_style='perplexity'.\
                                                         Alternatively use ensemble size > 1, ")
            
        self.llm = llm
        self.llm_name = llm_name

        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_style = parse_style
        self.relationship = relationship
        self.edge_value   = edge_value
        self.local_or_remote = local_or_remote
        self.deepspeed_compat = deepspeed_compat

        self.generation_kwargs = {}
        self.generation_parse_kwargs = {}
        k = isinstance(llm, langchain.llms.huggingface_pipeline.HuggingFacePipeline )*'max_new_tokens' + isinstance(llm, langchain.chat_models.ChatOpenAI)*'max_length'
        self.generation_kwargs[k]= 10 if prompt_style == 'yes_no' else 50 if prompt_style == 'open' else None
        self.generation_parse_kwargs[k]= 6
        self.effect_type = effect_type


    def predict(self, li_li_prompts:list[list[str]])->tuple[ list[list[str]], list[list[dict[str,int|float]]] ]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        
        # Generate predictions
        li_li_preds : list[list[str]] = []
            
        if isinstance(self.llm, langchain.chat_models.base.BaseChatModel): #type: ignore
            for li_prompts in li_li_prompts:
                batch_messages = [
                    [ SystemMessage(content=map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style]  ),
                        HumanMessage(content=prompt) ]
                        for prompt in li_prompts]
                
                outputs = self.llm.generate(batch_messages, **self.generation_kwargs)
                li_preds: list[str] = [ chatgen.text for chatgen in outputs.generations ]
                li_li_preds.append(li_preds)
        
        elif isinstance(self.llm, langchain.llms.base.LLM): #type: ignore
            # Set the generation kwargs - Langchain equivalent method to allow variable generation kwargs
            for k,v in self.generation_kwargs.items():
                self.llm.pipeline._forward_params[k] = v

            for li_prompts in li_li_prompts:
                
                li_prompts_fmtd = [
                    map_llmname_input_format(self.llm_name).format( 
                        system_message = map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style],
                        user_message = prompt) for prompt in li_prompts ]

                # prompts_fmtd = [ map_relationship_system_prompt[self.relationship][self.effect_type] + '\n' + prompt for prompt in li_prompts ]

                outputs = self.llm.generate( 
                    prompts=li_prompts_fmtd  )
                
                li_preds : list[str] = [ chatgen.text for chatgen in sum(outputs.generations,[]) ]
            
                li_li_preds.append(li_preds)
        else:
            raise ValueError(f"llm type {type(self.llm)} not recognized")


        # Parse {'Yes':prob_yes, 'No':prob_no, 'Nan':prob_nan } from the predictions
        if self.parse_style == 'rule_based':
            li_li_preds_parsed = [ self.parse_yesno_with_rules(li_predictions) for li_predictions in li_li_preds]

        elif self.parse_style == 'perplexity':
            li_li_preds_parsed = [ self.parse_yesno_with_lm_perplexity(li_predictions) for li_predictions in li_li_preds]
        
        elif self.parse_style == 'generation':
            li_li_preds_parsed = [ self.parse_yesno_with_lm_generation(li_predictions) for li_predictions in li_li_preds]
        
        else:
            raise ValueError(f"parse_style {self.parse_style} not recognized")

        return li_li_preds, li_li_preds_parsed

    def parse_yesno_with_rules(self, li_predictions:list[str]) -> list[dict[str,float]] :
        
        li_preds = [{}]*len(li_predictions)

        # Parse yes/no from falsetrue
        for idx, prediction in enumerate(li_predictions):
            
            prediction = copy.deepcopy(prediction).lower()

            if 'yes' in prediction:
                dict_pred = {'Yes':1.0, 'No':0.0, 'NA':0.0}
            
            elif 'no' in prediction:
                dict_pred = {'Yes':0.0, 'No':1.0, 'NA':0.0}
                        
            elif any( ( neg_phrase in prediction for neg_phrase in ['not true','false', 'is not', 'not correct', 'does not', 'can not', 'not'])):
                dict_pred = {'Yes':0.0, 'No':1.0, 'NA':0.0}

            elif any( ( neg_phrase in prediction for neg_phrase in ['true','correct'])):
                dict_pred = {'Yes':1.0, 'No':0.0, 'NA':0.0}

            else:
                dict_pred = {'Yes':0.0, 'No':0.0, 'NA':1.0 }
        
            li_preds[idx] = dict_pred 
                   
        return li_preds
               
    def parse_yesno_with_lm_generation(self, li_predictions:list[str])-> list[dict[str, float]] :
        
        """This method generates an output categorising the response to either affirmation or negation, then uses rules to  parse the output"""

        # Template to prompt language llm to simplify the answer to a Yes/No output
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_categorise_answer_affirm_negat']
        template = copy.deepcopy( random.choice(li_prompts) )
            

        # Create filled versions of the template with each of the predictions
        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]
        
        # Generate prediction
        if isinstance(self.llm, langchain.chat_models.base.BaseChatModel): #type: ignore
            batch_messages = [
                [ SystemMessage(content=map_relationship_sppywlg[self.relationship] ),
                    HumanMessage(content=prompt) ]
                    for prompt in li_filledtemplate]
            
            outputs = self.llm.generate(batch_messages, **self.generation_parse_kwargs)
            li_preds_str:list[str] = [ chatgen.text for chatgen in outputs.generations ]


        elif isinstance(self.llm, langchain.llms.base.LLM): #type: ignore
            for k,v in self.generation_parse_kwargs.items():
                self.llm.pipeline._forward_params[k] =  v

            li_prompts_fmtd = [
                map_llmname_input_format(self.llm_name).format( 
                    system_message = map_relationship_sppywlg[self.relationship],
                    user_message = prompt) for prompt in li_prompts ]
            
            outputs = self.llm.generate( prompts= li_prompts_fmtd )

            li_preds_str: list[str] = [ chatgen.text for chatgen in sum(outputs.generations,[]) ]

        else:
            raise ValueError(f"llm type {type(self.llm)} not recognized")

        # Parse Yes/No/Na from the prediction
        li_preds:list[dict[str,float]] = [ {'Yes':1.0,'No':0.0, 'NA':0.0 } if 'affirm' in pred.lower() else {'Yes':0.0,'No':1.0, 'NA':0.0 } if 'negat' in pred.lower() else {'Yes':0.0,'No':0.0, 'NA':1.0 } for pred in li_preds_str]

        return li_preds
    
    def parse_yesno_with_lm_perplexity(self, li_predictions:list[str])->  list[dict[str,float]] :
        # Get average perplexity of text when sentence is labelled Negation vs when it is labelled Affirmation.
        # NOTE: the perpleixty is calculated as an average on the whole text, not just the answer. Therefore, we rely on the
        #       the fact that 'Negation". and "Affirmation". both contain the same number of tokens

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_categorise_answer_affirm_negat']
        template = copy.deepcopy( random.choice(li_prompts) )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]

        li_filledtemplate = [
                map_llmname_input_format(self.llm_name).format( 
                    system_message = map_relationship_sppywlg[self.relationship],
                    user_message = prompt) for prompt in li_prompts ] #Added some base model formatting

        # For each fill template, create 3 filled versions with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = ['Negation', 'Affirmation']
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ans for ans in answers ] for filledtemplate in li_filledtemplate ]
        li_filledtemplates_with_answers = sum(li_li_filledtemplates_with_answers,[])

        # Get the perplexity of each of the filled templates
        # return a flattened set of perplexities
        li_perplexity = perplexity(
            li_filledtemplates_with_answers, 
            self.llm.pipeline.model, 
            self.llm.pipeline.tokenizer,
            batch_size=6, 
            deepspeed_compat = self.deepspeed_compat ) 
        
        li_preds: list[dict[str,float]] = [ {'Yes': li_perplexity[ idx + answers.index('Affirmation') ] , 'No': li_perplexity[ idx + answers.index('Negation') ] , } for idx in range(0,len(li_perplexity),len(answers)) ]

        return li_preds
        
    def aggregate_predictions(self, li_li_predictions:list[list[dict[str,int|float]]] )->  list[float | dict[str,float] ] :
        
        """            
            
            For Each prediction we have a list of samples.
                Each sample is a dictionary with keys {'Yes':pred_yes, 'No':pred_no, 'NA':pred_na}"

            edge_value: 'binary_weight': for a prediction p with n samples, returns 1 if Yes is the modal prediction

            edge_value: 'float_weight': for a prediction p with n samples, returns the average relative weight of Yes for each sample
            
            edge_value: 'distribution': for a prediction p with n samples, returns the average relative weights of Yes/No/NA for each sample

            output format: list of dictionaries with keys {'Yes':pred_yes, 'No':pred_no, 'NA':pred_na}. Reduced to 1 sample per prediction
        """
        
        if self.edge_value == 'binary_weight':
            
            # For each prediction (list of samples), return 1 if Yes is the model prediction
            def is_yes_mode(li_dict_pred) -> float:
                li_argmax_pred = [ max(d, key=d.get) for d in li_dict_pred ]
                most_common_pred = Counter(li_argmax_pred).most_common(1)[0][0]                
                y_val = 1.0 if most_common_pred == 'Yes' else 0.0
                n_val = 1.0 if most_common_pred == 'No' else 0.0
                na_val =1 - y_val - n_val

                d = {'Yes':y_val,'No':n_val,'NA':na_val}
                return d
           
            li_pred_agg = [ is_yes_mode(li_dict_pred) for li_dict_pred in li_li_predictions ]
  
        elif self.edge_value == 'distribution':
             
            def avg_relative_y(li_dict_pred):
                li_relative_yes = [  ]
                li_relative_no = [  ]
                li_relative_na = [  ]

                for d in li_dict_pred:
                    sum_vals = sum(d.values())
                    relative_yes = d['Yes'] / sum_vals 
                    relative_no = d['No'] / sum_vals
                    relative_na = d['NA'] / sum_vals

                    li_relative_yes.append(relative_yes)
                    li_relative_no.append(relative_no)
                    li_relative_na.append(relative_na)

                avg_relative_yes = sum(li_relative_yes) / len(li_relative_yes)
                avg_relative_no = sum(li_relative_no) / len(li_relative_no)
                avg_relative_na = sum(li_relative_na) / len(li_relative_na)

                return {'Yes':avg_relative_yes, 'No':avg_relative_no, 'NA':avg_relative_na}
            
            li_pred_agg = [ avg_relative_y(li_dict_pred) for li_dict_pred in li_li_predictions ]
        
        else:
            raise NotImplementedError(f"Aggregation method {self.edge_value} not implemented")
        
        return li_pred_agg #type: ignore

def load_annotated_examples(k_shot_example_dset_name:str|None, 
                            relationship_type:str='budgetitem_to_indicator') -> list[dict[str,str]] | None:
    
    li_records: list[dict[str,str]] | None = None

    if k_shot_example_dset_name == 'spot' and relationship_type == 'budgetitem_to_indicator':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./data/spot/spot_indicator_mapping_table_train.csv')
        
        li_records = dset.to_dict('records') #type: ignore
    
    elif k_shot_example_dset_name == 'spot' and relationship_type == 'indicator_to_indicator':
        logging.warning('Currently, there does not exist any annotated examples for indicator to indicator relationships. Therefore we can not use K-Shot templates for indicator to indicator edge determination. This will be added in the future')        
        li_records = None

    elif k_shot_example_dset_name == 'england':
        logging.warning('Currently, the relationshps for England dataset have not yet been distilled. This will be added in the future')        
        li_records = None
    
    else:
        raise ValueError('Invalid dset_name: ' + str(k_shot_example_dset_name))

    return li_records
