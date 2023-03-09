# This experiment produces predictions for a given test set
import os, sys
sys.path.append(os.getcwd())
import math
import transformers
from argparse import ArgumentParser
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from collections import ChainMap
import random

import utils_prompteng
import copy 

def main(
        nn_name:str,
        finetuned:bool,
        prompt_style:str,
        k_shot:int,
        ensemble_size:int,
        dset_name:str,
        batch_size:int=1):
    
    
    # Load Model and Tokenizer
    if not finetuned:
        model = transformers.AutoModel.from_pretrained( nn_name, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        path = './finetune/finetuned_models/' + nn_name + '/checkpoints/'
        model = transformers.AutoModel.from_pretrained( path, load_in_8bit=True, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Dataset
    train_dset, test_dset = load_dataset(dset_name)

    # Create Prompt Builder
    prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset, tokenizer )
    prediction_generator = PredictionGenerator(model, tokenizer, prompt_style, ensemble_size)

    # Creating Predictions for each row in the test set
    ## Convert pandas dictionary to list of dictionaries batched together
    test_dset_records = test_dset.to_dict('records')
    test_dset_records = [test_dset_records[i:i+batch_size] for i in range(0, len(test_dset), batch_size)]

    outp = []
    for batch in test_dset_records:
        # Create prompts
        li_li_prompts = prompt_builder(batch)
        
        # Generate predictions
        predictions = prediction_generator.pred(li_li_prompts)

    outp.append(predictions)  
    
    # Align the outputs with the original test set
    test_dset['predictions'] = outp

    # Save CSV with predictions
    test_dset.to_csv(f"./prompt_engineering/output/"+experiment_name(nn_name, finetuned, prompt_style, k_shot, ensemble_size, dset_name)+".csv")

    logging.info("Done")
    return None

def load_dataset(dset_name:str):
    
    if dset_name == 'spot':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./data/spot/test.csv')
        
        # Remove all rows where 'type' is not 'Outcome'
        dset = dset[dset['type'] == 'Outcome']

        # Creating target field
        dset['label'] = 'Yes'

        # Create negative examples
        dset = utils_prompteng.create_negative_examples(dset)

        # Rename columns to match the format of the other datasets
        dset.rename( {'category': 'budget_item', 'name':'indicator' } )

        # perform stratified split of a dataframe into train and test subsets
        train_dset, test_dset = train_test_split(dset, test_size=0.8, random_state=42, stratify=dset['category'])

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
    name = f"{dset_name}/{nn_name.replace('/','_')}"
    name += f"_ft" if finetuned else ""
    name += f"_ps{prompt_style}_k{k_shot}_es{ensemble_size}"
    return name

class PromptBuilder():
    def __init__(self, prompt_style:str, k_shot:int, ensemble_size:int, train_dset:list[dict], tokenizer ) -> None:
        self.prompt_style = prompt_style
        self.k_shot = k_shot
        self.ensemble_size = ensemble_size
        self.train_dset = train_dset
        self.tokenizer = tokenizer


    def __call__(self, batch:list[dict]) -> list[list[str]]:
        
        # Each element in batch has the same template

        # First we generate an ensemble of templates
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
        
        elif self.prompt_style in ['pilegithub_yes_no', 'pilegithub_open']:
            li_li_prompts = self.fill_template_open(templates, batch)
    
        return li_li_prompts
    
    
    def _yes_no_template(self) -> list[str]:
        # We store 10 yes_no prommpts templates in order of quality of prompt
        # When producing a prompt set with ensemble_size<10 we simply choose the first n prompts templates
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        
        templates = copy.deepcopy( utils_prompteng.li_prompts_yes_no_template[:self.ensemble_size] )
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator='{{target_indicator}}' )

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator='{{indicator_{k}}}' )
                context_k = context_k + f"\nAnswer: {{answer_{k}}}."
                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def _open_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_openend_template[:self.ensemble_size] )
        
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}',  indicator='{{target_indicator}}' )

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator='{{indicator_{k}}}' )
                context_k_response = f"\nAnswer: {{answer_{k}}}."
                
                # template_responses[ens_idx]['Yes'|'No'].format( budget_item=f'{{budget_item_{k}}}' ,  indicator='{{indicator_{k}}}' )
                
                context_k = context_k + context_k_response

                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates

    def _ama_template(self) -> list[str]:
        raise NotImplementedError

    def _pilegithub_yes_no_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_yes_no_template[:self.ensemble_size] )
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator='{{target_indicator}}' )

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator='{{indicator_{k}}}' )
                context_k = context_k + f"\nA:\n\n {{answer_{k}}}."
                prompt = context_k + "\n\n\n\n"+prompt
            
            templates[ens_idx] = prompt
        
        return templates

    def _pilegithub_open_template(self) -> list[str]:
        templates = copy.deepcopy( utils_prompteng.li_prompts_openend_template[:self.ensemble_size] )
        

        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator='{{target_indicator}}' )

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator='{{indicator_{k}}}' )
                context_k_response = f"\nA:\n\n{{answer_{k}}}."

                context_k = context_k + context_k_response
                prompt = context_k + "\n\n\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def fill_template_yesno(self, templates:list[str], batch:list[str])->list[list[str]]:
        
        li_li_prompts = []
        for row in batch:
            
            li_prompts = []
            for ens_idx in range(self.ensemble_size):
                # Fill in the k_shot context with random extracts from dataset
                ## sample k items from a list into a format dict                
                format_dict = ChainMap( ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":d['label'] } for idx, d in  enumerate(random.sample(self.train_dset, self.k_shot) ) ) )

                ## filling context examples in template
                prompt = templates[ens_idx].format(
                    **format_dict
                )

                # Fill in the target info
                prompt = prompt.format( target_budget_item= row['budget_item'], target_indicator=row['indicator'])
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(prompt)

    def fill_template_open(self, templates:list[str], batch:list[str])->list[list[str]]:
        
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
                format_dict =  ChainMap( ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ) )

                ## filling context examples in template
                prompt =  templates[ens_idx].format(**format_dict)

                # Fill in the target info
                prompt = prompt.format( target_budget_item= row['budget_item'], target_indicator=row['indicator'])
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(prompt)
        
        return li_li_prompts

class PredictionGenerator():
    def __init__(self, model, tokenizer, prompt_style:str, ensemble_size:int):
        
        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.model.eval()

        self.tokenizer
        
    # def predict(self, li_budget_items:list[str], li_indicators:list[str], li_labels:list[str])->list[str]:
    
    def predict(self, li_li_prompts)->list[str]:
                
        # Tokenize prompts
        li_batch_encoding = [self.tokenizer(li_prompt, return_tensors='pt', truncation=True, padding=True) for li_prompt in li_li_prompts]
        
        # Generate predictions
        li_outputs = []
        for batch_encoding in li_batch_encoding:
            
            # for ens_idx in range(self.ensemble_size):
                
            #     input_ids = batch_encoding['input_ids'][ens_idx].unsqueeze(0).to('cuda')
            #     attention_mask = batch_encoding['attention_mask'][ens_idx].unsqueeze(0).to('cuda')
                
                # Generate prediction
            outputs = self.model.generate(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                **batch_encoding,
                max_length=50,
                do_sample=True,
                top_p=0.95,
                top_k=60,
                temperature=0.7,
                num_return_sequences=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True)

            li_outputs.append(outputs)

        li_li_predictions = [ self.tokenizer.batch_decode(output, skip_special_tokens=True) for output in li_outputs]


        if self.prompt_style in ['yes_no','pilegithub_yes_no']:
            li_li_predictions = [ self.parse_yesno(li_predictions) for li_predictions in li_li_predictions]

        if self.prompt_style in ['open','pilegithub_open']:
            li_li_predictions = [ self.parse_yesno_from_falsetrue(li_predictions) for li_predictions in li_li_predictions]
        
        return li_li_predictions

    def parse_yesno_from_falsetrue(self, li_predictions:list[str])->list[str]:
        # Parse yes/no from falsetrue
        for idx in range(len(li_predictions)):
            
            if 'yes' not in li_predictions.lower() and 'no' not in li_predictions.lower():
                if any( ( neg_phrase in li_predictions[idx].lower() for neg_phrase in ['not true','false'])):
                    li_predictions[idx] = 'No'
                
                elif 'true' in li_predictions[idx].lower():
                    li_predictions[idx] = 'Yes'
                
                   
        return li_predictions
        
    def parse_yesno_from_open(self, li_predictions:list[str])->list[str]:
        
        for idx in range(len(li_predictions)):
            
            if 'yes' not in li_predictions.lower() and 'no' not in li_predictions.lower():
                
                if any( (neg_phrase in li_predictions[idx].lower() for neg_phrase in ['is not', 'does not', 'can not', 'not'] )) :
                    li_predictions[idx] = 'No'

                else:
                    li_predictions[idx] = 'Yes'
                   
        return li_predictions


            

            

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
    parser.add_argument('--finetuned', action='store_true', default=False, help='Indicates whether a finetuned version of nn_name should be used' )
    
    parser.add_argument('--prompt_style',type=str, default='restricted', choices=['yes_no','open','ama','pilegithub_yes_no', 'pilegithub_open'], 
                            help='Style of prompt' )
    parser.add_argument('--k_shot', type=int, default=1 )
    parser.add_argument('--ensemble_size', type=int, default=1 )
    
    parser.add_argument('--dset_name',type=str, default='spot', choices=['spot','england'] )

    parser.add_argument('--batch_size', type=int, default=1 )


    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)