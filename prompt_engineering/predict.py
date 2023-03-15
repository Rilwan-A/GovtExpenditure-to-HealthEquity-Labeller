# This experiment produces predictions for a given test set
import torch
import os, sys
sys.path.append(os.getcwd())
import math
import transformers
from transformers import GenerationConfig
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import random

import utils_prompteng
import copy 
from functools import reduce
import operator
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
        debugging:bool=False):
    
    
    # Load Model and Tokenizer
    if not finetuned:
        model = transformers.AutoModelForCausalLM.from_pretrained( nn_name, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        path = './finetune/finetuned_models/' + nn_name + '/checkpoints/'
        model = transformers.AutoModelForCausalLM.from_pretrained( path, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Load Dataset
    train_dset, test_dset = load_dataset(dset_name)

    test_dset = test_dset if not debugging else test_dset[:5]

    train_dset_records = train_dset.to_dict('records')
    test_dset_records = test_dset.to_dict('records')
    
    # Create Prompt Builder
    prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset_records, tokenizer )
    prediction_generator = PredictionGenerator(model, tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method)

    # Creating Predictions for each row in the test set
    ## Convert pandas dictionary to list of dictionaries batched together
    test_dset_records = [test_dset_records[i:i+batch_size] for i in range(0, len(test_dset), batch_size)]

    li_prompt_ensemble = []
    li_pred_ensemble = []
    preds_agg = []

    for idx, batch in enumerate(test_dset_records):
        # Create prompts
        batch_prompt_ensembles = prompt_builder(batch)
        
        # Generate predictions
        batch_pred_ensembles = prediction_generator.predict(batch_prompt_ensembles)

        # Aggregate ensembles into predictions
        batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles)

        # Extract predictions from the generated text
        li_prompt_ensemble.append(batch_prompt_ensembles)  # type: ignore
        li_pred_ensemble.append( batch_pred_ensembles ) # type: ignore
        preds_agg.append(batch_pred_agg) # type: ignore
                  
    # Align the outputs with the original test set
    test_dset['preds_aggregate'] = sum( preds_agg, [] )
    test_dset['preds_prompts'] = [ json.encode(prompt_ensemble) for prompt_ensemble in  sum(li_prompt_ensemble, []) ]
    test_dset['preds_ensemble'] = [ json.encode(pred_ensemble) for pred_ensemble in sum(li_pred_ensemble, []) ]
    
    # Save CSV with predictions
    parent_dir = "./prompt_engineering/predictions/"
    exp_name = experiment_name(nn_name, finetuned, prompt_style, k_shot, ensemble_size, dset_name)
    os.makedirs(os.path.join(parent_dir,exp_name), exist_ok=True )
    test_dset.to_csv( os.path.join( parent_dir, exp_name, "predictions.csv"), index=False )
    
    experiment_config = {'nn_name':nn_name, 'finetuned':finetuned, 'prompt_style':prompt_style, 'k_shot':k_shot, 'ensemble_size':ensemble_size, 'dset_name':dset_name}
    # Save experiment config as a yaml file
    yaml.safe_dump(experiment_config, open( os.path.join(parent_dir,exp_name, 'exp_config.yml'), 'w') )

    logging.info("Done")
    return None

def load_dataset(dset_name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if dset_name == 'spot':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./datasets/spot/spot_indicator_mapping_table.csv')
        
        # Remove all rows where 'type' is not 'Outcome'
        dset = dset[dset['type'] == 'Outcome']

        # Creating target field
        dset['label'] = 'Yes'

        # Rename columns to match the format of the other datasets
        dset = dset.rename( columns={'category': 'budget_item', 'name':'indicator' } )

        # Create negative examples
        dset = utils_prompteng.create_negative_examples(dset)

        # Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
        dset = dset.groupby(['budget_item','label']).filter(lambda x: len(x) > 1)
    
        # perform stratified split of a dataframe into train and test subsets
        train_dset, test_dset = train_test_split(dset, test_size=0.8, random_state=42, stratify=dset[['budget_item','label']])

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
        dset_name:str
):
    name = f"{dset_name}_{nn_name.replace('/','_')}"
    name += f"_ft" if finetuned else ""
    name += f"_ps{prompt_style}_k{k_shot}_es{ensemble_size}"
    return name

class PromptBuilder():
    def __init__(self, prompt_style:str, k_shot:int, ensemble_size:int, train_dset:list[dict], tokenizer:transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast ) -> None:
        self.prompt_style = prompt_style
        self.k_shot = k_shot
        self.ensemble_size = ensemble_size
        self.train_dset = train_dset
        self.tokenizer = tokenizer

    def __call__(self, batch:list[dict]) -> list[list[str]]:
        
        # Each element in batch has the same template
        # First we generate an ensemble of templates to be filled in for each element in the batch
        if self.prompt_style == 'yes_no':
            templates = self._yes_no_template()
        elif self.prompt_style == 'open':
            templates = self._open_template()
        elif self.prompt_style == 'ama':
            templates = self._ama_template()
        elif self.prompt_style == 'pilegithub_yes_no':
            templates = self._pilegithub_yes_no_template()
        elif self.prompt_style == 'pilegithub_open':
            templates = self._pilegithub_open_template()
        else:
            raise ValueError('Invalid prompt_style: ' + self.prompt_style)

        # Second given a k_shot prompt template, we then create n = ensemble_size, realisations of the template by sampling from the training set
        if self.prompt_style in ['yes_no', 'pilegithub_yes_no']:
            li_li_prompts = self.fill_template_yesno(templates, batch)
        
        elif self.prompt_style in ['open', 'pilegithub_open']:
            li_li_prompts = self.fill_template_open(templates, batch)
        
        else:
            li_li_prompts = []
    
        return li_li_prompts
    
    def _yes_no_template(self) -> list[str]:
        # We store 10 yes_no prommpts templates in order of quality of prompt
        # When producing a prompt set with ensemble_size<10 we simply choose the first n prompts templates
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        
        templates = copy.deepcopy( utils_prompteng.li_prompts_yes_no_template[:self.ensemble_size] )
        for ens_idx in range(self.ensemble_size):
            
            # part of prompt to be filled with information about target
            prompt = templates[ens_idx].format( budget_item='{target_budget_item}' ,  indicator='{target_indicator}' )
            prompt = prompt + "\nAnswer: " #{target_answer}."

            # Add k_shot context to prompt
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator=f'{{indicator_{k}}}' )
                context_k = context_k + f"\nAnswer: {{answer_{k}}}."
                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def _open_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_openend_template[:self.ensemble_size] )
        
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}' )
            prompt = prompt + "\nAnswer: "

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator=f'{{indicator_{k}}}' )
                context_k_response = f"\nAnswer: {{answer_{k}}}."
                                
                context_k = context_k + context_k_response

                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates

    def _ama_template(self) -> list[str]:
        raise NotImplementedError

    def _pilegithub_yes_no_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_yes_no_template[:self.ensemble_size] )
        for ens_idx in range(self.ensemble_size):
            
            # part of prompt to be filled with information about target
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator=f'{{target_indicator}}' )
            prompt = prompt + "\nAnswer:\n\n "

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator=f'{{indicator_{k}}}' )
                context_k = context_k + f"\nA:\n\n {{answer_{k}}}."
                prompt = context_k + "\n\n\n\n"+prompt
            
            templates[ens_idx] = prompt
        
        return templates

    def _pilegithub_open_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_openend_template[:self.ensemble_size] )
        
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator=f'{{target_indicator}}' )
            prompt = prompt + "\nAnswer:\n\n "

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator=f'{{indicator_{k}}}' )
                context_k_response = f"\nA:\n\n{{answer_{k}}}."

                context_k = context_k + context_k_response
                prompt = context_k + "\n\n\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def fill_template_yesno(self, templates:list[str], batch:list[dict]) -> list[list[str]]:
        
        li_li_prompts = []
        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            for ens_idx in range(self.ensemble_size):
                # Fill in the k_shot context with random extracts from dataset
                ## sample k items from our train set into a format dict for the template
                format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":d['label'] } for idx, d in  enumerate(random.sample(self.train_dset, self.k_shot) ) ], {} ) 
                    

                ## filling context examples in template and target info
                prompt = templates[ens_idx].format(
                    target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                    **format_dict
                )

                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts

    def fill_template_open(self, templates:list[str], batch:list[dict])->list[list[str]]:
        
        template_responses = copy.deepcopy( utils_prompteng.li_prompts_openend_template_open_response[:self.ensemble_size] )
        
        li_li_prompts = []
        for row in batch:
            
            li_prompts = []
            for ens_idx in range(self.ensemble_size):
                # Fill in the k_shot context with random extracts from dataset
                
                # Sample math.ceil(k/2) positive and math.floor(k/2) negative examples
                pos_examples_sample = random.sample( [d for d in self.train_dset if d['label']=='Yes'], math.ceil(self.k_shot/2) )
                neg_examples_sample = random.sample( [d for d in self.train_dset if d['label']=='No'], math.floor(self.k_shot/2) )
                
                # Creating the open ended answer version of the examples
                pos_examples_open_ended_answer = [ template_responses[ens_idx]['Yes'].format(budget_item=d['budget_item'], indicator=d['indicator']) for  d in pos_examples_sample ]
                neg_examples_open_ended_answer = [ template_responses[ens_idx]['No'].format(budget_item=d['budget_item'], indicator=d['indicator']) for d in neg_examples_sample ]

                # python shuffle two lists in the same order 
                li_examples = list(zip( list(pos_examples_sample) + list(neg_examples_sample), list(pos_examples_open_ended_answer) + list(neg_examples_open_ended_answer) ))
                random.shuffle(li_examples)
                examples_sample, examples_open_ended_answer = zip(*li_examples)

                # Creating the format dict for all examples
                format_dict =  reduce(operator.ior, ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} ) # type: ignore

                ## filling context examples in template
                prompt =  templates[ens_idx].format(target_budget_item= row['budget_item'], target_indicator=row['indicator'], **format_dict)

                # Fill in the target info
                prompt = prompt.format( )
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts

class PredictionGenerator():
    def __init__(self, model, tokenizer:transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast, prompt_style:str, ensemble_size:int,
                  aggregation_method:str='majority_vote', parse_output_method:str='rule_based' ):
        
        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_output_method = parse_output_method

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        if getattr(model, "is_loaded_in_8bit", False) is False:
            self.model = self.model.to(device)
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
            gen_config = GenerationConfig(
                max_new_tokens = 6,
                min_new_tokens = 3,
                early_stopping = True,
                temperature=0.7,
                do_sample = False,
                num_beams = 1,
                # top_k = 60,
                # top_p = 0.95,   
                # repetition_penalty = 1.0,
            )
        
        elif 'open' in self.prompt_style:
            try:
                gen_config = GenerationConfig.from_pretrained(self.model.name_or_path, 
                    max_new_tokens = 100, min_new_tokens = 5)
            except:
                gen_config = GenerationConfig(max_new_tokens = 100, min_new_tokens = 5)
        
        elif 'ama' == self.prompt_style:
            try:
                gen_config = GenerationConfig.from_pretrained(self.model.name_or_path, 
                    max_new_tokens = 100, min_new_tokens = 4)
            except:
                gen_config = GenerationConfig(max_new_tokens = 100, min_new_tokens = 4)
        else:
            logging.info(f"Prompt style {self.prompt_style} not recognized, using default generation config")
            try:
                gen_config = GenerationConfig.from_pretrained(self.model.name_or_path,
                    max_new_tokens = 100, min_new_tokens = 3)
            except:
                gen_config = GenerationConfig(max_new_tokens = 100, min_new_tokens = 3)

        self.gen_config = gen_config
        return None

    def predict(self, li_li_prompts:list[list[str]])->list[list[str]]:
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
                max_new_tokens = min( self.gen_config.max_new_tokens, self.model_max_tokens - batch_encoding['input_ids'].shape[1]   ), #type: ignore
                generation_config=self.gen_config,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = self.tokenizer.eos_token_id
                )

            li_outputs.append(outputs)

        li_li_predictions = [ self.tokenizer.batch_decode(output, skip_special_tokens=True) for output in li_outputs]

        # Extract just the answers from the decoded texts, removing the prompts
        li_li_predictions = [ [ pred.lstrip(prompt) for pred, prompt in zip(li_prediction, li_prompt) ] for li_prediction, li_prompt in zip(li_li_predictions, li_li_prompts) ]
        
        if self.parse_output_method == 'rule_based':
            # Parse label Yes or No from the answer
            if self.prompt_style in ['yes_no','pilegithub_yes_no']:
                li_li_predictions = [ self.parse_yesno_from_falsetrue(li_predictions) for li_predictions in li_li_predictions]

            if self.prompt_style in ['open','pilegithub_open']:
                li_li_predictions = [ self.parse_yesno_from_open(li_predictions) for li_predictions in li_li_predictions]
            
            else:
                raise ValueError(f"Prompt style {self.prompt_style} not recognized")

        elif self.parse_output_method == 'language_model':
            li_li_predictions = [ self.parse_yesno_with_lm(li_predictions) for li_predictions in li_li_predictions]
        
        else:
            raise ValueError(f"parse_output_method {self.parse_output_method} not recognized")

        return li_li_predictions

    def parse_yesno_from_falsetrue(self, li_predictions:list[str])->list[str]:
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
        
            li_predictions[idx] = prediction
                   
        return li_predictions
        
    def parse_yesno_from_open(self, li_predictions:list[str])->list[str]:

        li_predictions = self.parse_yesno_from_open(li_predictions)
                   
        return li_predictions         

    def parse_yesno_with_lm(self, li_predictions:list[str])->list[str]:
        
        # Template to prompt language model to simplify the answer to a Yes/No output
        template = copy.deepcopy( utils_prompteng.li_prompts_parse_yesno_from_answer[0] )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions]

        # For each fill template, create 3 filled versions with each of the possible answers
        answer = ['Agreement.', 'Disagreement.', 'Unclear.']
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answer ] for filledtemplate in li_filledtemplate ]
        li_filledtemplates_with_answers = sum(li_li_filledtemplates_with_answers,[])

        # Get the perplexity of each of the filled templates
        li_perplexity = utils_prompteng.perplexity(li_filledtemplates_with_answers, self.model, self.tokenizer, batch_size=1 ) 

        # For each set of filltemplates get the index of the answer with the lowest perplexity
        li_idx = [ np.argmin(li_perplexity[idx:idx+3]) for idx in range(0,len(li_perplexity),3) ]

        # Map the indexes to the answers
        li_predictions = [ filledtemplates[idx] for idx,filledtemplates in zip(li_idx, li_li_filledtemplates_with_answers) ]

        # remove the templates from the outputs
        li_predictions = [ pred.lstrip(template) for pred, template in zip(li_predictions, li_filledtemplate) ]

        # Map the answers to Yes/No/Na
        li_predictions = [ 'Yes' if 'Agreement' in pred else 'No' if 'Disagreement' in pred else 'NA' for pred in li_predictions ]
                
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


def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
    parser.add_argument('--finetuned', action='store_true', default=False, help='Indicates whether a finetuned version of nn_name should be used' )
    
    parser.add_argument('--prompt_style',type=str, choices=['yes_no','open','ama','pilegithub_yes_no', 'pilegithub_open'], help='Style of prompt' )
    parser.add_argument('--parse_output_method',type=str, choices=['rule_based','language_model'], help='How to convert the output of the model to a Yes/No Output' )
    parser.add_argument('--k_shot', type=int, default=1, help='Number of examples to use for each prompt. Note this number must respect the maximum length allowed by the language model used' )
    parser.add_argument('--ensemble_size', type=int, default=1 )

    parser.add_argument('--aggregation_method', type=str, default='majority_vote', choices=['majority_vote'] )
    parser.add_argument('--dset_name',type=str, default='spot', choices=['spot','england'] )
    parser.add_argument('--batch_size', type=int, default=1 )

    parser.add_argument('--debugging', action='store_true', default=False, help='Indicates whether the script is being run in debugging mode')


    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))