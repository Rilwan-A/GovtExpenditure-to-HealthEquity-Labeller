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

from prompt_engineering import utils_prompteng
from prompt_engineering.utils_prompteng import PromptBuilder

import copy 

from collections import Counter
import ujson as json
import yaml
import numpy as np


def main(
        nn_name:str,
        prompt_style:str,
        dset_name:str,
        k_shot:int=0,
        ensemble_size:int=1,
        finetuned:bool=False,
        batch_size:int=1,

        aggregation_method='majority_vote',
        parse_output_method='rule_based',
        effect_order='directly',

        remove_public_health:bool=False,
        
        model=None,
        tokenizer=None,
        
        save_output:bool=True,
        debugging:bool=False):

        
    # Load Model and Tokenizer
    if not finetuned:
        model = transformers.AutoModelForCausalLM.from_pretrained( nn_name, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        path = './finetune/finetuned_models/' + nn_name + '/checkpoints/'
        model = model if (model is not None) else transformers.AutoModelForCausalLM.from_pretrained( path, load_in_8bit=True, device_map="auto")
        tokenizer = copy.deepcopy(tokenizer) if (tokenizer is not None) else transformers.AutoTokenizer.from_pretrained(nn_name)
        
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = 'left'
    
    # Load Dataset
    train_dset, test_dset = load_dataset(dset_name, remove_public_health=remove_public_health)
    test_dset = test_dset if not debugging else test_dset[:5]

    train_dset_records = train_dset.to_dict('records')
    test_dset_records = test_dset.to_dict('records')
    ## Convert pandas dictionary to list of dictionaries batched together
    test_dset_records = [test_dset_records[i:i+batch_size] for i in range(0, len(test_dset), batch_size)]

    # Create Prompt Builder
    prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset_records, effect_order )
    prediction_generator = PredictionGenerator(model, tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method, deepspeed_compat=False)

    # Creating Predictions for each row in the test set
    li_prompt_ensemble = []
    li_pred_ensemble = []
    li_pred_ensemble_parsed = []
    preds_agg = []

    for idx, batch in enumerate(test_dset_records):
        # Create prompts
        batch_prompt_ensembles, batch_pred_ensembles, batch_pred_ensembles_parsed, batch_pred_agg = step(batch, prompt_builder, prediction_generator)

        # Extract predictions from the generated text
        li_prompt_ensemble.append(batch_prompt_ensembles)  # type: ignore
        li_pred_ensemble.append( batch_pred_ensembles ) # type: ignore
        li_pred_ensemble_parsed.append( batch_pred_ensembles_parsed ) # type: ignore
        preds_agg.append(batch_pred_agg) # type: ignore
                  
    # Align the outputs with the original test set
    test_dset['preds_aggregate'] = sum( preds_agg, [] )
    test_dset['preds_ensemble_parsed'] = [ json.encode(pred_ensemble) for pred_ensemble in sum(li_pred_ensemble_parsed, []) ]
    test_dset['preds_ensemble'] = [ json.encode(pred_ensemble) for pred_ensemble in sum(li_pred_ensemble, []) ]
    test_dset['preds_prompts'] = [ json.encode(prompt_ensemble) for prompt_ensemble in  sum(li_prompt_ensemble, []) ]
    
    if save_output:
        # Save CSV with predictions
        parent_dir = "./prompt_engineering/output/"
        exp_name = experiment_name(nn_name, finetuned, prompt_style, k_shot, ensemble_size, 
                                   dset_name, aggregation_method, parse_output_method, effect_order,
                                   remove_public_health)
        os.makedirs(os.path.join(parent_dir,exp_name), exist_ok=True )
        test_dset.to_csv( os.path.join( parent_dir, exp_name, "predictions.csv"), index=False )
        
        experiment_config = {'nn_name':nn_name, 'finetuned':finetuned, 'prompt_style':prompt_style, 'k_shot':k_shot, 'ensemble_size':ensemble_size, 'dset_name':dset_name, 
                                'aggregation_method':aggregation_method, 'parse_output_method':parse_output_method, 'effect_order':effect_order,
                                 'remove_public_health':remove_public_health }
        # Save experiment config as a yaml file
        yaml.safe_dump(experiment_config, open( os.path.join(parent_dir,exp_name, 'exp_config.yml'), 'w') )
    
    return test_dset

class PredictionGenerator():
    def __init__(self, model, tokenizer:transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast, prompt_style:str, ensemble_size:int,
                  aggregation_method:str='majority_vote', parse_output_method:str='rule_based',
                    device=None, deepspeed_compat:bool=False ):
        
        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_output_method = parse_output_method
        self.deepspeed_compat = deepspeed_compat
        self.model = model
        
        if (device is not None) and getattr(model, "is_loaded_in_8bit", False) is False:
            self.model = self.model.to(device)

        if (device is not None):
            self.model.eval()

        self.tokenizer = tokenizer
        
        self.setup_generation_config()
        self.aggregation_method = aggregation_method

    def setup_generation_config(self):
        
        model_max_tokens = self.model.config.max_position_embeddings

        if hasattr(self.model.config, 'n_positions'):
            model_max_tokens = self.model.config.max_position_embeddings

        elif hasattr(self.model.config, 'max_position_embeddings'):
            model_max_tokens = self.model.config.max_position_embeddings

        elif hasattr(self.model.config, 'n_ctx'):
            model_max_tokens = self.model.config.max_position_embeddings

        else:
            model_max_tokens = 512
        self.model_max_tokens = model_max_tokens

        #TODO: optimize the GenerationConfigs settings
        if 'yes_no' in self.prompt_style:
            eos_token_ids =  [self.tokenizer.eos_token_id] + [ self.tokenizer(text).input_ids[-1] for text in ['.',' .']]
            suppress_tokens = [self.tokenizer(text).input_ids[-1] for text in ['\n','\n\n','Question','Answer']]
            gen_config = GenerationConfig(
                max_new_tokens = 10, min_new_tokens = 1,
                early_stopping = True, temperature=0.7,
                do_sample = False,
                eos_token_id=eos_token_ids,
                suppress_tokens=suppress_tokens,
                pad_token_id = self.tokenizer.pad_token_id
            )
                    
        elif 'open' in self.prompt_style:
            eos_token_ids =  [self.tokenizer.eos_token_id] + [ self.tokenizer(text).input_ids[-1] for text in ['.',' .']]
            suppress_tokens = [ self.tokenizer(text).input_ids[-1] for text in ['\n','\n\n','Question','Answer'] ]

            gen_config = GenerationConfig(max_new_tokens = 20,
                                           min_new_tokens = 1,
                                           early_stopping=True,
                                           temperature=0.7,
                                           eos_token_id=eos_token_ids,
                                           suppress_tokens=suppress_tokens,
                                           beam_size=2,
                                           length_penalty=-0.5,
                                           no_repeat_ngram_size=3)
        
        elif 'ama' == self.prompt_style:
            gen_config = GenerationConfig(max_new_tokens = 100, min_new_tokens = 4 , early_stopping=True, 
                                          eos_token_id=self.tokenizer.eos_token_id, temperature=0.7, 
                                          do_sample = False, pad_token_id = self.tokenizer.pad_token_id)
        else:
            gen_config = None
            logging.info(f"Prompt style {self.prompt_style} not recognized")

        self.gen_config = gen_config
        return None

    def predict(self, li_li_prompts:list[list[str]])->tuple[list[list[str]], list[list[str]]]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        # Tokenize prompts
        li_batch_encoding = [self.tokenizer(li_prompt, return_tensors='pt', padding=True, truncation_strategy='do_not_truncate') for li_prompt in li_li_prompts]
        
        # Generate predictions
        li_outputs = []
        for batch_encoding in li_batch_encoding:

            # Move to device
            batch_encoding = batch_encoding.to(self.model.device)    

            # Generate prediction
            outputs = self.model.generate(
                **batch_encoding,
                max_new_tokens = min( self.gen_config.max_new_tokens, self.model_max_tokens - batch_encoding['input_ids'].shape[1]), #type: ignore
                generation_config=self.gen_config,
                pad_token_id = self.tokenizer.pad_token_id)

            li_outputs.append(outputs)

        li_li_predictions = [ self.tokenizer.batch_decode(output, skip_special_tokens=True) for output in li_outputs]

        # Extract just the answers from the decoded texts, removing the prompts
        li_li_predictions = [ [ pred.replace(prompt,'') for pred, prompt in zip(li_prediction, li_prompt) ] for li_prediction, li_prompt in zip(li_li_predictions, li_li_prompts) ]
        
        # Parse Yes/No/Nan from the predictions
        if self.parse_output_method == 'rule_based':
            li_li_predictions_parsed = [ self.parse_yesno_with_rules(li_predictions) for li_predictions in li_li_predictions]

        elif self.parse_output_method == 'language_model_perplexity':
            li_li_predictions_parsed = [ self.parse_yesno_with_lm_perplexity(li_predictions) for li_predictions in li_li_predictions]
        
        elif self.parse_output_method == 'language_model_generation':
            li_li_predictions_parsed = [ self.parse_yesno_with_lm_generation(li_predictions) for li_predictions in li_li_predictions]
        
        else:
            raise ValueError(f"parse_output_method {self.parse_output_method} not recognized")

        return li_li_predictions, li_li_predictions_parsed

    def parse_yesno_with_rules(self, li_predictions:list[str])->list[str]:
        
        li_preds_parsed = ['NA']*len(li_predictions)

        # Parse yes/no from falsetrue
        for idx in range(len(li_predictions)):
            
            prediction = li_predictions[idx].lower()

            if 'yes' in prediction:
                prediction = 'Yes'
            
            elif 'no' in prediction:
                prediction = 'No'
            
            elif any( ( neg_phrase in prediction for neg_phrase in ['not true','false', 'is not', 'not correct', 'does not', 'can not', 'not'])):
                prediction = 'No'

            else:
                prediction = 'NA'
        
            li_preds_parsed[idx] = prediction
                   
        return li_predictions
               
    def parse_yesno_with_lm_generation(self, li_predictions:list[str])->list[str]:
        
        # Template to prompt language model to simplify the answer to a Yes/No output
        template = copy.deepcopy( utils_prompteng.li_prompts_parse_yesno_from_answer[0] )

        # Create filled versions of the template with each of the predictions
        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions]

        # Create batch encoding
        batch_encoding = self.tokenizer(li_filledtemplate, return_tensors='pt', padding=True, truncation_strategy='do_not_truncate')

        # Move to device
        batch_encoding = batch_encoding.to(self.model.device)


        # setup generation config for parsing yesno
        eos_token_ids = [self.tokenizer.eos_token_id] + [ self.tokenizer(text)['input_ids'][-1] for text in ['"Negation".', '"Affirmation".', 'Negation.', 'Affirmation.','\n' ] ]
        
        gen_config = transformers.GenerationConfig(max_new_tokens = 20, min_new_tokens = 2, early_stopping=True, 
                                            temperature=0.7, no_repeat_ngram_size=3,
                                            eos_token_id=eos_token_ids,
                                            do_sample=False,
                                            pad_token_id = self.tokenizer.pad_token_id   )
        
        # Generate prediction
        output = self.model.generate(**batch_encoding, generation_config=gen_config )

        # Decode output
        predictions_w_prompt = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # Extract just the answers from the decoded texts, removing the prompts
        li_predictions = [ pred.replace(template,'') for pred,template in zip(predictions_w_prompt, li_filledtemplate) ]

        # Parse Yes/No/Na from the prediction
        li_predictions_parsed = [ 'Yes' if 'affirm' in pred.lower() else 'No' if 'negat' in pred.lower() else 'NA' for pred in li_predictions]

        return li_predictions_parsed
    
    def parse_yesno_with_lm_perplexity(self, li_predictions:list[str])->list[str]:
        # Get average perplexity of text when sentence is labelled Negation vs when it is labelled Affirmation.
        # NOTE: the perpleixty is calculated as an average on the whole text, not just the answer. Therefore, we rely on the
        #       the fact that 'Negation". and "Affirmation". both contain the same number of tokens

        # Template to prompt language model to simplify the answer to a Yes/No output
        template = copy.deepcopy( utils_prompteng.li_prompts_parse_yesno_from_answer[0] )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions]

        # For each fill template, create 3 filled versions with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = ['Negation', 'Affirmation']
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answers ] for filledtemplate in li_filledtemplate ]
        li_filledtemplates_with_answers = sum(li_li_filledtemplates_with_answers,[])

        # Get the perplexity of each of the filled templates
        li_perplexity = utils_prompteng.perplexity(li_filledtemplates_with_answers, 
                                                   self.model, self.tokenizer, batch_size=6, deepspeed_compat = self.deepspeed_compat ) 

        # For each set of filltemplates get the index of the answer with the lowest perplexity
        li_idx = [ np.argmin(li_perplexity[idx:idx+len(answers)]) for idx in range(0,len(li_perplexity),len(answers)) ]

        li_predictions = [ 'No' if idx==0 else 'Yes' for idx in li_idx ]
        
        return li_predictions
         

    def aggregate_predictions(self, li_li_predictions:list[list[str]])->list[str]:
        "Given a list of predictions, returns a single prediction"
        
        if self.ensemble_size == 1:
            li_predictions = [ li_prediction[0] for li_prediction in li_li_predictions ]
        
        elif self.aggregation_method == 'majority_vote':
            li_predictions = [ Counter(li_prediction).most_common(1)[0][0] for li_prediction in li_li_predictions ]
        
        else:
            raise NotImplementedError(f"Aggregation method {self.aggregation_method} not implemented")
        
        return li_predictions


def load_dataset(dset_name:str, random_state_seed:int=10, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if dset_name == 'spot':
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

        # remove all rows from dset that have value 'Public Health' for budget_item
        if kwargs.get('remove_public_health', False):
            dset = dset[ dset['budget_item'] != 'Public Health' ]
        
        # To vauge a budget item and there are only 4 examples of it, we remove it
        dset = dset[ dset['budget_item'] != 'Central' ]

        # create negative examples
        dset = utils_prompteng.create_negative_examples(dset, random_state=random_state )

        # Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
        dset = dset.groupby(['budget_item','label']).filter(lambda x: len(x) > 1)
    
        # perform stratified split of a dataframe into train and test subsets
        train_dset, test_dset = train_test_split(dset, test_size=0.8, random_state=random_state, stratify=dset[['budget_item','label']])

    elif dset_name == 'england':
        raise NotImplementedError
    
    else:
        raise ValueError('Invalid dset_name: ' + dset_name)

    return train_dset, test_dset

def experiment_name(
        nn_name:str,
        finetuned:bool,
        prompt_style:str,
        k_shot:int,
        ensemble_size:int,
        dset_name:str,
        aggregation_method:str,
        parse_output_method:str,
        effect_order:str,
        remove_public_health:bool
):
    name = f"{dset_name}/{nn_name.replace('/','_')}/"
    name += "_FT" if finetuned else ""
    name += f"_PS{''.join([s[0] for s in prompt_style.split('_')])}"
    name += f"_K{k_shot}" if k_shot > 0 else ""
    name += f"_ES{ensemble_size}" if ensemble_size > 1 else ""
    name += f"_AG{ ''.join( [s[0] for s in aggregation_method.split('_')] ) }"
    name += f"_PO{ ''.join( [s[0] for s in parse_output_method.split('_')]) }"
    name += f"_EO{effect_order[0]}"
    name += "_RPH" if remove_public_health else ""
    return name

def step(batch, prompt_builder:PromptBuilder, prediction_generator:PredictionGenerator ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[str]]:

    # Create prompts
    batch_prompt_ensembles = prompt_builder(batch)
    
    # Generate predictions
    batch_pred_ensembles, batch_pred_ensembles_parsed = prediction_generator.predict(batch_prompt_ensembles)

    # Aggregate ensembles into predictions
    batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles_parsed)

    return batch_prompt_ensembles, batch_pred_ensembles, batch_pred_ensembles_parsed, batch_pred_agg

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
    parser.add_argument('--finetuned', action='store_true', default=False, help='Indicates whether a finetuned version of nn_name should be used' )
    
    parser.add_argument('--prompt_style',type=str, choices=['yes_no','open','ama','pilestackoverflow_yes_no', 'pilestackoverflow_open'], default='open', help='Style of prompt' )
    parser.add_argument('--parse_output_method',type=str, choices=['rule_based','language_model_perplexity', 'language_model_generation' ], default='language_model_perplexity', help='How to convert the output of the model to a Yes/No Output' )
    parser.add_argument('--k_shot', type=int, default=0, help='Number of examples to use for each prompt. Note this number must respect the maximum length allowed by the language model used' )
    parser.add_argument('--ensemble_size', type=int, default=1 )
    parser.add_argument('--effect_order', type=str, default='arbitrary', choices=['arbitrary','1st', '2nd'] )
    
    parser.add_argument('--aggregation_method', type=str, default='majority_vote', choices=['majority_vote'] )
    parser.add_argument('--dset_name',type=str, default='spot', choices=['spot','england'] )
    parser.add_argument('--batch_size', type=int, default=1 )

    parser.add_argument('--debugging', action='store_true', default=False, help='Indicates whether the script is being run in debugging mode')

    # Spot dataset specific arguments
    parser.add_argument('--remove_public_health', action='store_true', default=False, help='Indicates whether the public health budget item should be removed from the dataset' )

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))