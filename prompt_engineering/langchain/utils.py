from time import sleep
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain import PromptTemplate, LLMChain
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from prompt_engineering.utils_prompteng import (map_relationship_system_prompt, map_relationship_promptsmap, 
                                                map_relationship_sysprompt_categoriesanswer,
                                                map_llmname_input_format,
                                                  joint_probabilities_for_category, nomalized_probabilities,
                                                   map_category_answer, map_category_label, 
                                                   map_category_answer, map_category_label )
import random

import copy
import numpy as np
import pandas as pd
from functools import lru_cache
import peft
import warnings
import os
from contextlib import redirect_stdout
import openai

#https://old.reddit.com/r/LocalLLaMA/wiki/models#wiki_current_best_choices
HUGGINGFACE_MODELS = [ 
    'TheBloke/Wizard-Vicuna-7B-Uncensored-HF',
    'TheBloke/Wizard-Vicuna-13B-Uncensored-HF',
    'CalderaAI/30B-Lazarus',
    'TheBloke/Wizard-Vicuna-30B-Uncensored-fp16',

    'stabilityai/StableBeluga-7B',
    'NousResearch/Nous-Hermes-llama-2-7b',

    'stabilityai/StableBeluga-13B',
    'NousResearch/Nous-Hermes-Llama2-13b',
    'openaccess-ai-collective/minotaur-13b-fixed',

    'upstage/llama-30b-instruct-2048',
    'upstage/Llama-2-70b-instruct-v2',
    'stabilityai/StableBeluga2',
    ]

MAP_LOAD_IN_NBIT = {
    'TheBloke/Wizard-Vicuna-7B-Uncensored-HF':4,
    'TheBloke/Wizard-Vicuna-13B-Uncensored-HF':4,
    'CalderaAI/30B-Lazarus':4,
    'TheBloke/Wizard-Vicuna-30B-Uncensored-fp16':4,

    'stabilityai/StableBeluga-7B':4,
    'NousResearch/Nous-Hermes-llama-2-7b':4,

    'stabilityai/StableBeluga-13B':4,
    'NousResearch/Nous-Hermes-Llama2-13b':4,
    'openaccess-ai-collective/minotaur-13b-fixed':4,

    'upstage/llama-30b-instruct-2048':4,
    'upstage/Llama-2-70b-instruct-v2':4,
    'stabilityai/StableBeluga2':4,
      }

OPENAI_MODELS = ['gpt-3.5-turbo', 'gpt-4']
ALL_MODELS = HUGGINGFACE_MODELS + OPENAI_MODELS
from collections import Counter
import logging
       
def load_llm( llm_name:str, finetuned:bool=False, local_or_remote:str='local', api_key:str|None=None, device=-1, finetune_dir='./finetune/finetuned_models/', exp_name='', finetune_version=0 ):
    from  langchain.chat_models import ChatOpenAI
    from  langchain.llms import HuggingFaceHub
    from langchain import HuggingFacePipeline
    import torch

    assert local_or_remote in ['local', 'remote'], f"local_or_remote must be either 'local' or 'remote', not {local_or_remote}"
    if local_or_remote == 'remote': assert api_key is not None, f"api_key must be provided if local_or_remote is 'remote'"
    if local_or_remote == 'local': assert llm_name not in OPENAI_MODELS, f"llm_name must be a HuggingFace model if local_or_remote is 'local'" 
    assert llm_name in ALL_MODELS, f"llm_name must be a valid model name, not {llm_name}"
    # TODO: All models used done on Huggingface Hub
    # TODO: if user wants to run it faster they can download the model from Huggingface Hub and load it locally and use the predict step with different --local_or_remote set to local
    if local_or_remote == 'local':

        if llm_name in HUGGINGFACE_MODELS:
            from transformers import BitsAndBytesConfig
            
            bool_8=MAP_LOAD_IN_NBIT[llm_name] == 8
            bool_4=MAP_LOAD_IN_NBIT[llm_name] == 4
            quant_config = BitsAndBytesConfig(
                
                load_in_8bit=bool_8,
                llm_int8_has_fp16_weights=bool_8,
                
                load_in_4bit=bool_4,
                bnb_4bit_quant_type="nf4" ,
                bnb_4bit_use_double_quant=bool_4,
                bnb_4bit_compute_dtype=torch.bfloat16 if bool_4 else None
            )

            if not finetuned:
                model_id = llm_name
                llm = HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    model_kwargs={'trust_remote_code':True,
                                    'quantization_config':quant_config,
                                    'do_sample':False,
                                    'device_map':'auto'
                                    },
                    )
            else:
                # load pytorch lightning LightningModule from finetune_dir then extract the llm as lightningModule.model
                # Load the best checkpoint automatically

                from prompt_engineering.finetune import PromptEngineeringLM
                import re

                # Set your checkpoints directory
                ckpt_dir = os.path.join(finetune_dir, exp_name, f'version_{str(finetune_version)}/checkpoints')

                # Get list of all checkpoint files
                ckpt_files = os.listdir(ckpt_dir)

                # Filter out non-checkpoint files
                ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]

                # Find the checkpoint with the highest accuracy
                best_ckpt = max(ckpt_files, key=lambda f: float(re.search(r"val_spot_acc=(\d+(.\d+)?)", f).group(1)) if re.search(r"val_spot_acc=(\d+(.\d+)?)", f) else 0)

                best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)

                # Load the best model
                model = PromptEngineeringLM.load_from_checkpoint(checkpoint_path=best_ckpt_path)

                # Extract the llm
                llm = model.model



        else:
            raise NotImplementedError(f"llm_name {llm_name} is not implemented for local use")
        
    elif local_or_remote == 'remote':
        if llm_name in OPENAI_MODELS:
            
            llm = ChatOpenAI(
                client=openai.ChatCompletion,
                model_name=llm_name,
                openai_api_key=api_key )    #ignore: type        
        
        elif llm_name in HUGGINGFACE_MODELS:
            llm = HuggingFaceHub(
                    repo_id=llm_name, huggingfacehub_api_token=api_key, model_kwargs={ 'do_sample':False } ) #type: ignore
        else:
            raise NotImplementedError(f"llm_name {llm_name} is not implemented for remote use")

    else:
        llm = None

    return llm 

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
                  effect_type:str='directly',
                **kwargs ):
                
        if parse_style in ['categories_perplexity']: 
            assert local_or_remote == 'local', "Can not get model logits scores from remote models"
            assert not isinstance(llm, langchain.chat_models.ChatOpenAI), "Can not get model logits scores from ChatOpenAI"

        # Restrictions on combinations of parse style and edge value
        if edge_value in ['distribution'] and parse_style in ['rules','categories_rules']:
            assert ensemble_size>1, "Can not get a float edge value with ensemble size 1 and parse_style:{parse_style}.\
                                                         To use ensemble size 1, please use parse_style='categories_perplexity'.\
                                                         Alternatively use ensemble size > 1, "
            
        self.llm = llm
        self.llm_name = llm_name
        

        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_style = parse_style
        self.relationship = relationship
        self.edge_value   = edge_value
        self.local_or_remote = local_or_remote
      
        self.effect_type = effect_type

    # @lru_cache(maxsize=2)
    # def get_generation_params(self, prompt_style:str):
    #     generation_params = {}
    #     k = isinstance(self.llm, langchain.llms.huggingface_pipeline.HuggingFacePipeline )*'max_new_tokens' + isinstance(self.llm, langchain.chat_models.ChatOpenAI)*'max_tokens' + isinstance(self.llm, peft.peft_model.PeftModelForCausalLM )*'max_new_tokens'
    #     if prompt_style == 'yes_no':
    #         generation_params[k] = 10
    #     elif prompt_style == 'open':
    #         generation_params[k] = 10
    #     elif prompt_style == 'categorise':
    #         generation_params[k] = 10
    #     elif prompt_style == 'cot_categorise':
    #         generation_params[k] = 250

    #     if isinstance(self.llm, langchain.llms.huggingface_pipeline.HuggingFacePipeline ):
    #         generation_params['early_stopping'] = True
        
    #     return generation_params

    def predict(self, li_li_filled_template:list[list[str]], reverse_categories=False)->tuple[ list[list[str]], list[list[str]], list[list[dict[str,int|float]]] ]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        

        # Parse {'Yes':prob_yes, 'No':prob_no, 'Nan':prob_nan } from the predictions
        if self.parse_style == 'rules':
            li_li_pred = [ self.parse_outp_rules(li_filled_template) for li_filled_template in li_li_filled_template]
        elif self.parse_style == 'categories_rules':
            li_li_pred = [ self.parse_outp_categories_rules(li_filled_template) for li_filled_template in li_li_filled_template]        
        elif self.parse_style == 'categories_perplexity':
            if self.prompt_style == 'categorise':
                li_li_pred = [ self.parse_outp_categories_perplexity(li_filled_template, reverse_categories=reverse_categories) for li_filled_template in li_li_filled_template]
            elif self.prompt_style == 'cot_categorise':
                li_li_pred = [ self.parse_outp_cotcategories_perplexity(li_filled_template, reverse_categories=reverse_categories) for li_filled_template in li_li_filled_template]
        else:
            raise ValueError(f"parse_style {self.parse_style} not recognized")

        return li_li_pred

    def parse_outp_rules(self, li_filled_template:list[str]) -> list[dict[str,float]] :
        
        li_preds = [{}]*len(li_filled_template)

        # Parse yes/no from falsetrue
        for idx, prediction in enumerate(li_filled_template):
            
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
               
    def parse_outp_categories_rules(self, li_preds:list[str])-> list[dict[str, float]] :
        
        """
            # Parse desired category from prediction based on category number present in the answer or if category name is present in the answer
        """
        
        def parse_outp_from_catpred(pred:str)->dict[str,float]:
            # Parse desired category from prediction based on category number present in the answer or if category name is present in the answer

            pred_lower = pred.lower()

            outp = {'Yes':0.0,'No':0.0, 'NA':1.0 }

            # Cycle through categories and check if any is included in the answer. If so, set that category to 1.0 and NA to 0.0

            # High Precision: Check if first chracter of pred is category number
            for cat in map_category_answer.keys():
                if cat in pred_lower[:1]:
                    label = map_category_label[cat]
                    outp[label] = 1.0
                    outp['NA'] = 0.0
                    return outp
            
            # Medium Precision: Check if whole category label is in prediction
            for cat in map_category_answer.keys():
                if map_category_answer[cat] in pred_lower:
                    label = map_category_label[cat]
                    outp[label] = 1.0
                    outp['NA'] = 0.0
                    return outp
            
            # Low Precision: Less robust test if any character in pred is a category number                
            for cat in map_category_answer.keys():
                if cat in pred_lower:
                    label = map_category_label[cat]
                    outp[label] = 1.0
                    outp['NA'] = 0.0
                    return outp

            return outp
            
        li_preds_parsed:list[dict[str,float]] = [ parse_outp_from_catpred(pred) for pred in li_preds]


        return li_preds_parsed
    
    def parse_outp_cotcategories_perplexity(self, li_filledtemplate:list[str], reverse_categories:bool=False)->  list[dict[str,float]] :

        # This function outputs a distribution of probabilities over Yes,:
        #   - First select a prompt which contains the initial statement to verify and a space for potential answers
        #   - Then fill the prompt with each of the possible answers (Yes/No) - creating 2 filled prompts
        #   - Then calculate the perplexity of each of the filled prompts - we only get perplexity of the fill answered for the text, not the whole prompt
        #   - Then calculate the probability of each of the answers as the ratio of the perplexities
        #   - Finally, return the probabilities as a distribution over the Yes/No/NA categories

        # New Addition:
        #   - For each datum we repeat the experiment with the numbers for the categorical answers swapped
        #   - Then we average the two sets of probabilities for each Yes / No
        #   - This is done to account for the fact that the model may be biased towards one of the categorical numbers
   

        # Formatting prompts to adhere to format required by Base Language Model
        li_filledtemplate_fmtd = [
                map_llmname_input_format(self.llm_name,
                                        user_message = filledtemplate, 
                                        system_message = map_relationship_sysprompt_categoriesanswer[self.relationship] )
                                    for filledtemplate in li_filledtemplate ] #Added some base model formatting
        
        # For each template, create a set of templates that ask a categorical question about the LLMs answer to if Budget Item affects an indiciator 
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = list(  map_category_answer.keys() )
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ans for ans in answers ] for filledtemplate in li_filledtemplate_fmtd ]

        # For each filled template set calcualte the relative probability of each answer
        li_li_probability = []
        for li_filledtemplates_with_answers in li_li_filledtemplates_with_answers:
            li_probability = joint_probabilities_for_category(
                li_filledtemplates_with_answers, 
                self.llm.pipeline.model, 
                self.llm.pipeline.tokenizer,
                batch_size=len(map_category_answer.keys()), 
                category_token_len=1 ) 
            
            li_li_probability.append(li_probability)
        
        # Convert set of perplexities into a list of list with each sublist having N perplexities for each category of answer
        _categorise_response_labels = copy.deepcopy(map_category_label)
        if reverse_categories:
            _ = map_category_label[answers[0]]
            _categorise_response_labels[answers[0]] = _categorise_response_labels[answers[1]] 
            _categorise_response_labels[answers[1]] = _

        li_map_probability = [ { _categorise_response_labels[answer]:li_probability[idx] for idx, answer in enumerate(answers) } for li_probability in li_li_probability ]

        # Convert this to normalised probabilities
        li_map_probability = [  nomalized_probabilities(map_perplexities) for map_perplexities in li_map_probability ]

        return li_map_probability
    
    def parse_outp_categories_perplexity(self, li_filledtemplate:list[str], reverse_categories=False)->  list[dict[str,float]] :

        # Formatting prompts to adhere to format required by Base Language Model
        li_filledtemplate_fmtd = [
                map_llmname_input_format(self.llm_name,
                                        user_message = prompt, 
                                        system_message = map_relationship_system_prompt[self.relationship][self.effect_type] + ' ' + map_relationship_system_prompt[self.relationship][self.prompt_style] 
                                        )
                                    for prompt in li_filledtemplate ] #Added some base model formatting

        # For each template, create a set of 2 filled templates with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = [ str(num) for num in range(1,1+len(map_category_answer.keys()) ) ]
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ans for ans in answers ] for filledtemplate in li_filledtemplate_fmtd ]


        # For each filled template set calcualte the relative probability of each answer
        li_li_probability = []
        for li_filledtemplates_with_answers in li_li_filledtemplates_with_answers:

            li_probability = joint_probabilities_for_category(
                li_filledtemplates_with_answers, 
                self.llm.pipeline.model, 
                self.llm.pipeline.tokenizer,
                batch_size=len(map_category_answer.keys()), 
                category_token_len=1 ) 
            
            li_li_probability.append(li_probability)
        
        # Convert set of perplexities into a list of list with each sublist having a probability for each category of answer
        _categorise_response_labels = copy.deepcopy(map_category_label)
        if reverse_categories:
            _ = map_category_label[answers[0]]
            _categorise_response_labels[answers[0]] = _categorise_response_labels[answers[1]] 
            _categorise_response_labels[answers[1]] = _

        li_map_probability = [ { _categorise_response_labels[answer]:li_probability[idx] for idx, answer in enumerate(answers) } for li_probability in li_li_probability ]

        # Convert this to normalised probabilities
        li_map_probability = [  nomalized_probabilities(map_probability) for map_probability in li_map_probability ]

        return li_map_probability

    def aggregate_predictions(self, li_li_dict_preds:list[list[dict[str,int|float]]] )->  list[float | dict[str,float] ] :
        
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
            def mode(li_dict_pred) -> dict[str,float]:
                li_argmax_pred = [ max(d, key=d.get) for d in li_dict_pred ]
                most_common_pred = Counter(li_argmax_pred).most_common(1)[0][0]                
                y_val = 1.0 if most_common_pred == 'Yes' else 0.0
                n_val = 1.0 if most_common_pred == 'No' else 0.0
                na_val =1 - y_val - n_val

                d = {'Yes':y_val,'No':n_val,'NA':na_val}
                return d
           
            li_pred_agg = [ mode(li_dict_pred) for li_dict_pred in li_li_dict_preds ]
  
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
            
            li_pred_agg = [ avg_relative_y(li_dict_pred) for li_dict_pred in li_li_dict_preds ]

            li_pred_agg = [{key: round(value, 3) for key, value in nested_dict.items()} for nested_dict in li_pred_agg]

        else:
            raise NotImplementedError(f"Aggregation method {self.edge_value} not implemented")
        
        return li_pred_agg #type: ignore

def load_annotated_examples(k_shot_example_dset_name:str|None, 
                            relationship_type:str='budgetitem_to_indicator') -> list[dict[str,str]] | None:
    
    li_records: list[dict[str,str]] | None = None

    if k_shot_example_dset_name == 'spot' and relationship_type == 'budgetitem_to_indicator':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./data/spot/spot_b2i_broad_train.csv')
        
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
