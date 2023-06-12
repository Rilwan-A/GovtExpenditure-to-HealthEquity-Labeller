from time import sleep
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from prompt_engineering.utils_prompteng import (map_relationship_system_prompt, map_relationship_promptsmap, 
                                                map_relationship_sysprompt_categoriesanswer, perplexity_for_category,
                                                map_llmname_input_format, perplexity_to_normalised_probability,
                                                  open_response_cats, joint_probabilities_for_category, nomalized_probabilities,
                                                   open_response_labels, categorise_cats )
import random

import copy
import numpy as np
import pandas as pd
from functools import lru_cache


#https://old.reddit.com/r/LocalLLaMA/wiki/models#wiki_current_best_choices
HUGGINGFACE_MODELS = [ 

    'mosaicml/mpt-7b-chat', 'TheBloke/vicuna-7B-1.1-HF', 'TheBloke/wizard-vicuna-13B-HF', 'TheBloke/Wizard-Vicuna-13B-Uncensored-HF', 'timdettmers/guanaco-33b-merged',

    'TheBloke/wizard-vicuna-13B-GPTQ', 'TheBloke/wizard-vicuna-13B-GPTQ', 'TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g'  ,'TheBloke/guanaco-65B-GPTQ'
    ]

MAP_LOAD_IN_NBIT = {
    
    'mosaicml/mpt-7b-chat': 8,

    'TheBloke/vicuna-7B-1.1-HF':8,
    'TheBloke/wizard-vicuna-13B-HF':4,
    'TheBloke/Wizard-Vicuna-13B-Uncensored-HF':4,

    'timdettmers/guanaco-33b-merged':4,
    'TheBloke/guanaco-65B-HF':4,

}

OPENAI_MODELS = ['gpt-3.5-turbo', 'gpt-4']
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
                  parse_style:str='rules',
                  relationship:str='budgetitem_to_indicator',
                  local_or_remote='local',
                  deepspeed_compat:bool=False,
                  effect_type:str='directly' ):
        
        # Can not get model logits scores from ChatOpenAI
        if isinstance(llm, langchain.chat_models.ChatOpenAI) and parse_style == 'categories_perplexity': 
            raise ValueError("Can not get model logits scores from ChatOpenAI") #type: ignore
        
        if parse_style in ['categories_perplexity', 'perplexity']: 
            assert local_or_remote == 'local', "Can not get model logits scores from remote models"

        # Restrictions on combinations of parse style and edge value
        if edge_value in ['distribution']:
            if parse_style not in ['categories_perplexity', 'perplexity'] :
                if ensemble_size == 1: raise ValueError(f"Can not get a float edge value with ensemble size 1 and parse_style:{parse_style}.\
                                                         To use ensemble size 1, please use parse_style='categories_perplexity'.\
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
      
        self.effect_type = effect_type

    @lru_cache(maxsize=2)
    def get_generation_params(self, prompt_style:str):
        generation_params = {}
        k = isinstance(self.llm, langchain.llms.huggingface_pipeline.HuggingFacePipeline )*'max_new_tokens' + isinstance(self.llm, langchain.chat_models.ChatOpenAI)*'max_tokens'
        if prompt_style == 'yes_no':
            generation_params[k] = 10
        elif prompt_style == 'open':
            generation_params[k] = 100
        elif prompt_style == 'categorise':
            generation_params[k] = 10
        elif prompt_style == 'cot':
            generation_params[k] = 210
        
        return generation_params

    def predict(self, li_li_prompts:list[list[str]])->tuple[ list[list[str]], list[list[str]], list[list[dict[str,int|float]]] ]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        
        # Generate predictions
        li_li_preds : list[list[str]] = []
        li_li_prompts_fmtd : list[list[str]] = []

        if self.parse_style == 'perplexity':
            # Case: prompt_style == 'cateogrise' so we do not need to generate predictions just insert answers and compare perplexities
            li_li_preds = li_li_prompts 

        elif isinstance(self.llm, langchain.chat_models.ChatOpenAI): #type: ignore
            
            generation_params = self.get_generation_params(self.prompt_style)
            
            for k,v in generation_params.items():
                setattr(self.llm, k, v)

            for li_prompts in li_li_prompts:
                sleep(20)
                batch_messages = [
                    [ SystemMessage(content=map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style]),
                        HumanMessage(content=prompt) ]
                        for prompt in li_prompts]
                
                
                outputs = self.llm.generate(batch_messages )
                li_preds: list[str] = [ li_chatgen[0].text for li_chatgen in outputs.generations ]
                li_li_preds.append(li_preds)
        
        elif isinstance(self.llm, langchain.llms.base.LLM): #type: ignore
            # Set the generation kwargs - Langchain equivalent method to allow variable generation kwargs            
            
            self.llm.pipeline._forward_params =  self.get_generation_params(self.prompt_style)

            for li_prompts in li_li_prompts:
                
                # Formatting prompts to adhere to format required by Base Language Model
                li_prompts_fmtd = [
                    map_llmname_input_format(self.llm_name,
                        user_message = prompt,
                        system_message = map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style]
                        ) for prompt in li_prompts ]

                outputs = self.llm.generate(
                    prompts=li_prompts_fmtd,
                    stop = ['\n\n'] )
                
                li_preds : list[str] = [ chatgen.text.strip(' ') for chatgen in sum(outputs.generations,[]) ]
            
                li_li_prompts_fmtd.append(li_prompts_fmtd)
                li_li_preds.append(li_preds)
        else:
            raise ValueError(f"llm type {type(self.llm)} not recognized")


        # Parse {'Yes':prob_yes, 'No':prob_no, 'Nan':prob_nan } from the predictions
        if self.parse_style == 'rules':
            li_li_preds_parsed = [ self.parse_outp_rules(li_predictions) for li_predictions in li_li_preds]

        elif self.parse_style == 'categories_perplexity':
            li_li_preds_parsed = [ self.parse_outp_categories_perplexity(li_predictions) for li_predictions in li_li_preds]
        
        elif self.parse_style == 'categories_rules':
            li_li_preds_parsed = [ self.parse_outp_categories_rules(li_predictions) for li_predictions in li_li_preds]
        
        elif self.parse_style == 'perplexity':
            li_li_preds_parsed = [ self.parse_outp_perplexity(li_predictions) for li_predictions in li_li_preds]

        else:
            raise ValueError(f"parse_style {self.parse_style} not recognized")

        return li_li_prompts_fmtd, li_li_preds, li_li_preds_parsed

    def parse_outp_rules(self, li_predictions:list[str]) -> list[dict[str,float]] :
        
        li_preds = [{}]*len(li_predictions)

        # Parse yes/no from falsetrue
        for idx, prediction in enumerate(li_predictions):
            
            prediction = copy.deepcopy(prediction).lower()

            
            if any( ( neg_phrase in prediction for neg_phrase in ['cannot answer','can\'t answer', 'not sure', 'not certain',  ])):
                dict_pred = {'Yes':0.0, 'No':0.0, 'NA':1.0}

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
               
    def parse_outp_categories_rules(self, li_predictions:list[str])-> list[dict[str, float]] :
        
        # Template to prompt language llm to simplify the answer to a Yes/No output
        li_template = map_relationship_promptsmap[self.relationship]['li_prompts_categories_answer_v2']
        template = copy.deepcopy( random.choice(li_template) )     

        
        # Create filled versions of the template with each of the predictions
        # NOTE: Currently the statement isn't inserted
        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]
        
        # Generate prediction
        if isinstance(self.llm, langchain.chat_models.ChatOpenAI): #type: ignore
            generation_params = self.get_generation_params(self.prompt_style)
            system_message = map_relationship_sysprompt_categoriesanswer[self.relationship]
            system_message = system_message if system_message is not None else ''

            for k,v in generation_params.items():
                setattr(self.llm, k, v)
            
            batch_messages = [
                [ SystemMessage(content=system_message),
                    HumanMessage(content=prompt) ]
                    for prompt in li_filledtemplate]
            outputs = self.llm.generate(batch_messages)
            li_preds_str:list[str] = [ li_chatgen[0].text for li_chatgen in outputs.generations ]

            # NOTE: sleep added to avoid API rate limits
            sleep(20)

        elif isinstance(self.llm, langchain.llms.base.LLM): #type: ignore
            
            generation_params = {
                # 'num_beams':3,
                'num_return_sequences':1,
                'early_stopping':True,
                'max_new_tokens': 60,
            }

            self.llm.pipeline._forward_params =  generation_params

            # Formatting prompts to adhere to format required by Base Language Model
            li_prompts_fmtd = [
                map_llmname_input_format(self.llm_name,
                    user_message = prompt,
                    system_message = map_relationship_sysprompt_categoriesanswer[self.relationship],
                    ) for prompt in li_filledtemplate ]
            
            #TODO: responses such as 'there is no clear relationship between local .. and  ...'  are classified as 'Nan' instead of No 
            #NOTE: Model keeps outputting the whole category name not just the first letter, how to fix
            outputs = self.llm.generate( prompts=li_prompts_fmtd )

            li_preds_str: list[str] = [ chatgen.text.strip(' ') for chatgen in sum(outputs.generations,[]) ]

        else:
            raise ValueError(f"llm type {type(self.llm)} not recognized")


        def parse_outp_from_catpred(pred:str)->dict[str,float]:
            # Prompt tries to coerce model to answer with category letter, so first we check if model instead output category name

            pred_lower = pred.lower()

            outp = {'Yes':0.0,'No':0.0, 'NA':1.0 }

            # Cycle through categories and check if any is included in the answer. If so, set that category to 1.0 and NA to 0.0
            for cat in open_response_cats.keys():
                if open_response_cats[cat].lower() in pred_lower:
                    label = open_response_labels[cat]
                    outp[label] = 1.0
                    outp['NA'] = 0.0 
                    break

            return outp
            
        li_preds:list[dict[str,float]] = [ parse_outp_from_catpred(pred) for pred in li_preds_str]


        return li_preds
    
    def parse_outp_categories_perplexity(self, li_predictions:list[str])->  list[dict[str,float]] :

        # This function outputs a distribution of probabilities over the Yes, No, NA categories and they are calcaulated as follows:
        #   - First select a prompt which contains the initial statement to verify and a space for potential answers
        #   - Then fill the prompt with each of the possible answers (Yes/No/NA) - creating 3 filled prompts
        #   - Then calculate the perplexity of each of the filled prompts - we only get perplexity of the fill text, not the whole prompt
        #   - Then calculate the probability of each of the answers as the ratio of the perplexities
        #   - Finally, return the probabilities as a distribution over the Yes/No/NA categories


        li_templates = map_relationship_promptsmap[self.relationship]['li_prompts_categories_answer_v1']
        template = copy.deepcopy( random.choice(li_templates) )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions ]

        # Formatting prompts to adhere to format required by Base Language Model
        li_filledtemplate = [
                map_llmname_input_format(self.llm_name,
                                        user_message = prompt, 
                                        system_message = map_relationship_sysprompt_categoriesanswer[self.relationship] )
                                    for prompt in li_filledtemplate ] #Added some base model formatting

        # For each template, create a set of 3 filled templates with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = list(  open_response_cats.keys() )
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answers ] for filledtemplate in li_filledtemplate ]


        # For each filled template set calcualte the relative probability of each answer
        li_li_probability = []
        for li_filledtemplates_with_answers in li_li_filledtemplates_with_answers:
            li_probability = joint_probabilities_for_category(
                li_filledtemplates_with_answers, 
                self.llm.pipeline.model, 
                self.llm.pipeline.tokenizer,
                batch_size=len(open_response_cats.keys()), 
                deepspeed_compat = self.deepspeed_compat,
                category_token_len=1 ) 
            li_li_probability.append(li_probability)
        
        # Convert set of perplexities into a list of list with each sublist having N perplexities for each category of answer
        li_map_probability = [ { open_response_labels[answer]:li_probability[idx] for idx, answer in enumerate(answers) } for li_probability in li_li_probability ]

        # Convert this to normalised probabilities
        li_map_probability = [  nomalized_probabilities(map_perplexities) for map_perplexities in li_map_probability ]

        return li_map_probability
    
    def parse_outp_perplexity(self, li_predictions:list[str])->  list[dict[str,float]] :

        # Formatting prompts to adhere to format required by Base Language Model
        li_filledtemplate = [
                map_llmname_input_format(self.llm_name,
                                        user_message = prompt, 
                                        system_message = (map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style] ).replace('  ',' ') )
                                    for prompt in li_predictions ] #Added some base model formatting

        # For each template, create a set of 3 filled templates with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = list(  categorise_cats.keys() )
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answers ] for filledtemplate in li_filledtemplate ]


        # For each filled template set calcualte the relative probability of each answer
        li_li_probability = []
        for li_filledtemplates_with_answers in li_li_filledtemplates_with_answers:
            li_probability = joint_probabilities_for_category(
                li_filledtemplates_with_answers, 
                self.llm.pipeline.model, 
                self.llm.pipeline.tokenizer,
                batch_size=len(categorise_cats.keys()), 
                deepspeed_compat = self.deepspeed_compat,
                category_token_len=1 ) 
            li_li_probability.append(li_probability)
        
        # Convert set of perplexities into a list of list with each sublist having N perplexities for each category of answer
        li_map_probability = [ { open_response_labels[answer]:li_probability[idx] for idx, answer in enumerate(answers) } for li_probability in li_li_probability ]

        # Convert this to normalised probabilities
        li_map_probability = [  nomalized_probabilities(map_perplexities) for map_perplexities in li_map_probability ]

        return li_map_probability

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
            def is_yes_mode(li_dict_pred) -> dict[str,float]:
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
                    relative_na = d.get('NA',0.0) / sum_vals

                    li_relative_yes.append(relative_yes)
                    li_relative_no.append(relative_no)
                    li_relative_na.append(relative_na)

                avg_relative_yes = sum(li_relative_yes) / len(li_relative_yes)
                avg_relative_no = sum(li_relative_no) / len(li_relative_no)
                avg_relative_na = sum(li_relative_na) / len(li_relative_na)

                return {'Yes':avg_relative_yes, 'No':avg_relative_no, 'NA':avg_relative_na}
            
            li_pred_agg = [ avg_relative_y(li_dict_pred) for li_dict_pred in li_li_predictions ]

            li_pred_agg = [{key: round(value, 4) for key, value in nested_dict.items()} for nested_dict in li_pred_agg]

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
