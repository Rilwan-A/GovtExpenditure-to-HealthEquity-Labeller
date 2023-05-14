import pandas as pd
import math
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from contextlib import nullcontext
import random
import copy
from random import sample 
from functools import reduce
import operator
map_relationship_promptsmap ={}

# region budgetitem to indicator templates
li_prompts_yes_no_template = [    
    "Give me a Yes or No answer to the following question, is local government spending on \"{budget_item}\" {effect_order} related to \"{indicator}\"?",
    
    'Does local government spending on \"{budget_item}\" {effect_order} affect \"{indicator}\"?, True or False',
    
    'Is it true that \"{indicator}\" is {effect_order} related to local government spending on \"{budget_item}\"?',
    
    'Does \"{budget_item}\" {effect_order} affect \"{indicator}\"?, Yes or No',
    
    'Answer the following question with True or False: Does local government spending on \"{budget_item}\" {effect_order} affect \"{indicator}\"?'
]
li_prompts_openend_template = [    
    'Is local government spending on \"{budget_item}\" {effect_order} related to \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" {effect_order} affects \"{indicator}\"?',
    
    'Is \"{indicator}\" {effect_order} related to local government spending on \"{budget_item}\"?',
    
    'Local government spending on \"{budget_item}\" {effect_order} improves \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" {effect_order} affects \"{indicator}\"?'
]
li_prompts_openend_template_open_response =[
    {'Yes':'Local government spending on \"{budget_item}\" is {effect_order} related to \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" is not {effect_order} related to \"{indicator}\".'},

    {'Yes':'Local government spending on \"{budget_item}\" does {effect_order} affect \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {effect_order} affect \"{indicator}\".'},

    {'Yes':'\"{indicator}\" is {effect_order} related to local government spending on \"{budget_item}\".', 'No':'\"{indicator}\" is not {effect_order} related to local government spending on \"{budget_item}\".'},

    {'Yes':'Local government spending on \"{budget_item}\" does {effect_order} improve \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {effect_order} improve \"{indicator}\".'},

    {'Yes':'A local government can {effect_order} affect \"{indicator}\" by spending on \"{budget_item}\".', 'No':'A local government can not {effect_order} affect \"{indicator}\" by spending on \"{budget_item}\".'}
]
li_prompts_parse_yesno_from_answer = [
    """Select the grammatical category that best describes the statement.\n\"Categories\":\n- Negation\n- Affirmation\nStatement: {statement}\nThis statement belongs to the category """
]

budgetitem_to_indicator_prompts = {
    'li_prompts_yes_no_template':li_prompts_yes_no_template,
    'li_prompts_openend_template':li_prompts_openend_template,
    'li_prompts_openend_template_open_response':li_prompts_openend_template_open_response,
    'li_prompts_parse_yesno_from_answer':li_prompts_parse_yesno_from_answer
}
map_relationship_promptsmap['budgetitem_to_indicator'] = budgetitem_to_indicator_prompts
#endregion

# region indicator to indicator templates
li_prompts_yes_no_template_i2i = [
    "Give me a Yes or No answer to the following question about the relatedness of two socio-economic/health indicators, is the level of  \"{indicator1}\" {effect_order} related to the state of \"{indicator2}\"?",
    
    'Does local government spending on improving the level of \"{indicator1}\" {effect_order} affect the level of \"{indicator2}\" ?, True or False',
    
    'Is it true that the level of \"{indicator1}\" is {effect_order} related to the level of \"{indicator2}\"?',
    
    'Do improvements in {indicator1} {effect_order} affect \"{indicator2}\"?, Yes or No',
    
    'Answer the following question with True or False: Does local government spending aimed at affecting \"{indicator1}\" {effect_order} affect \"{indicator2}\"?'

] 
li_prompts_parse_yesno_from_answer_i2i = [
    """Select the grammatical category that best describes the statement.\n\"Categories\":\n- Negation\n- Affirmation\nStatement: {statement}\nThis statement belongs to the category """
]
li_prompts_openend_template_i2i = [
    'Is the level of \"{indicator1}\" {effect_order} related to the state of \"{indicator2}\"?',

    'Does local government spending on improving the level of \"{indicator1}\" {effect_order} affect the level of \"{indicator2}\"?',

    'Is the level of \"{indicator1}\" {effect_order} related to the level of \"{indicator2}\"?',

    'Improvements in {indicator1} {effect_order} affect \"{indicator2}\"?',

    'Does local government spending aimed at affecting \"{indicator1}\" {effect_order} affect \"{indicator2}\"?'

]
li_prompts_openend_template_open_response_i2i = [
    {'Yes':'The level of \"{indicator1}\" is {effect_order} related to the state of \"{indicator2}\".', 'No':'The level of \"{indicator1}\" is not {effect_order} related to the state of \"{indicator2}\".'},

    {'Yes':'Local government spending on improving the level of \"{indicator1}\" does {effect_order} affect the level of \"{indicator2}\".', 'No':'Local government spending on improving the level of \"{indicator1}\" does not {effect_order} affect the level of \"{indicator2}\".'},

    {'Yes':'The level of \"{indicator1}\" is {effect_order} related to the level of \"{indicator2}\".', 'No':'The level of \"{indicator1}\" is not {effect_order} related to the level of \"{indicator2}\".'},

    {'Yes':'Improvements in {indicator1} do {effect_order} affect \"{indicator2}\".', 'No':'Improvements in {indicator1} do not {effect_order} affect \"{indicator2}\".'},

    {'Yes':'Local government spending aimed at affecting \"{indicator1}\" does {effect_order} affect \"{indicator2}\".', 'No':'Local government spending aimed at affecting \"{indicator1}\" does not {effect_order} affect \"{indicator2}\".'}

]

indicator_to_indicator_prompts = {
    'li_prompts_yes_no_template_i2i':li_prompts_yes_no_template_i2i,
    'li_prompts_openend_template_i2i':li_prompts_openend_template_i2i,
    'li_prompts_openend_template_open_response_i2i':li_prompts_openend_template_open_response_i2i,
    'li_prompts_parse_yesno_from_answer_i2i':li_prompts_parse_yesno_from_answer_i2i
}
map_relationship_promptsmap['indicator_to_indicator'] = indicator_to_indicator_prompts
# endregion


def create_negative_examples(dset:pd.DataFrame, random_state=None) -> pd.DataFrame:
    """Create negative examples for the Spot Datasetby randomly selecting a budget item and indicator
    from the dataset and then swapping them
    
    dset: pd.DataFrame
        The dataset to create negative examples from
    
    Returns
    -------
    pd.DataFrame
        The dataset with negative
    
    """
    l = len(dset)
    # Each budget_item has n records
    # For each budget_item we sample min(n,l-n) false examples
    # These false examples are created by filtering on other budget_items and then sampling
    
    for budget_item in dset['budget_item'].unique():
        dset_budget_item = dset[dset['budget_item']==budget_item]
        n = len(dset_budget_item)
        
        # The edit below is specific to the Spot Dataset
        if budget_item == 'Public Health':
            # Some of the labels are sub-categories of Public Health
            # Examples from these categories must be ignored when creating negative
            cat_ignore = ['Public Health', 'Health Improvement', 'Sexual Health', 'Mental Health', 'Health Protection', 'Child Health', 'Healthcare']

            # Get all the examples that do not have budget items in the ignore list
            dset_budget_item_neg = dset[~dset['budget_item'].isin(cat_ignore)].sample(min(n,l-n), replace=True, random_state=random_state) 

        else:
            dset_budget_item_neg = dset[dset['budget_item']!=budget_item].sample(min(n,l-n), replace=False, random_state=random_state) 
        
        dset_budget_item_neg['budget_item'] = budget_item
        dset_budget_item_neg['label'] = 'No'
        
        dset = pd.concat([dset, dset_budget_item_neg], axis=0)

    return dset

def perplexity(
    data, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, max_length=None, deepspeed_compat:bool=False):

    model = model
    tokenizer = tokenizer

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        data,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(model.device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(model.device), attn_mask], dim=1
            )

        labels = encoded_batch

        # use a dummy context for now
        with torch.no_grad() if not deepspeed_compat else nullcontext():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_attention_mask_batch = attn_mask[..., 1:]

        if deepspeed_compat is False:
            shift_logits = shift_logits.contiguous()
            shift_labels = shift_labels.contiguous()
            shift_attention_mask_batch = shift_attention_mask_batch.contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return ppls

class PromptBuilder():
    def __init__(self, prompt_style:str, k_shot:int, ensemble_size:int, 
                 examples_dset:list[dict], effect_order:str, relationship:str="budgetitem_to_indicator"  ) -> None:
        
        assert effect_order in [ 'arbitrary', '1st', '2nd']
        assert relationship in ['budgetitem_to_indicator', 'indicator_to_indicator']

        self.prompt_style = prompt_style
        self.k_shot = k_shot
        self.ensemble_size = ensemble_size
        self.examples_dset = examples_dset
        self.effect_order = effect_order
        self.effect_order_str = ""
        self.relationship = relationship
        self.init_effect_order_str()
        # when arbitrary is subbed into the prompt template, it will result in a double space in the prompt. We use .replace("  ", " ") to remove this

    def init_effect_order_str(self):
        """ Initialize the string representation of effect order

            If we are modelling 1st order effects only then we will use the word "indirectly" in the prompt. This achieves higher recall with lower precision.
            If we are modelling 2nd order effects only then we will use the word "directly" in the prompt. This achieves higher precision with lower recall.

            This is ideal since 2nd order effects also estimates weights of 

        """
        if self.effect_order == 'arbitrary':
            self.effect_order_str = ''
        elif self.effect_order == '1st':
            self.effect_order_str = 'indirectly'
        elif self.effect_order == '2nd':
            self.effect_order_str = 'directly'

    def __call__(self, batch:list[dict]) -> list[list[str]]:
        
        # Each element in batch has the same template
        # First we generate an ensemble of templates to be filled in for each element in the batch
        if self.prompt_style == 'yes_no':
            templates = self._yes_no_template()
        elif self.prompt_style == 'open':
            templates = self._open_template()
        elif self.prompt_style == 'ama':
            templates = self._ama_template()
        elif self.prompt_style == 'pilestackoverflow_yes_no':
            templates = self._pilestackoverflow_yes_no_template()
        elif self.prompt_style == 'pilestackoverflow_open':
            templates = self._pilestackoverflow_open_template()
        else:
            raise ValueError('Invalid prompt_style: ' + self.prompt_style)

        # Second given a k_shot prompt template, we then create n = ensemble_size, realisations of the template by sampling from the training set
        if self.prompt_style in ['yes_no', 'pilestackoverflow_yes_no']:
            li_li_prompts = self.fill_template_yesno(templates, batch)
        elif self.prompt_style in ['open', 'pilestackoverflow_open']:
            li_li_prompts = self.fill_template_open(templates, batch)
        else:
            li_li_prompts = []
            
        return li_li_prompts
    
    def _yes_no_template(self) -> list[str]:
        # We store 10 yes_no prommpts templates in order of quality of prompt
        # When producing a prompt set with ensemble_size<10 we simply choose the first n prompts templates
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_yes_no_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):
            
            # part of prompt to be filled with information about target
            prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_order=self.effect_order_str ).replace('  ',' ') +"\nAnswer:"
            
            # Add k_shot context to prompt
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}',  indicator=f'{{indicator_{k}}}', effect_order=self.effect_order_str ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."
                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def _open_template(self) -> list[str]:
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_openend_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )
        
        for ens_idx in range(self.ensemble_size):
            prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_order=self.effect_order_str ).replace('  ',' ') + "\nAnswer:"

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}', indicator=f'{{indicator_{k}}}', effect_order=self.effect_order_str ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."
                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates

    def _pilestackoverflow_yes_no_template(self) -> list[str]:
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_yes_no_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )
        for ens_idx in range(self.ensemble_size):
            
            # part of prompt to be filled with information about target
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator=f'{{target_indicator}}', effect_order=self.effect_order_str ).replace('  ',' ')
            prompt = prompt + "\nA:\n\n"

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}',  indicator=f'{{indicator_{k}}}', effect_order=self.effect_order_str ).replace('  ',' ')
                context_k = context_k + f"\nA:\n\n{{answer_{k}}}."
                prompt = context_k + "\n\n\n\n"+prompt
            
            templates[ens_idx] = prompt
        
        return templates

    def _pilestackoverflow_open_template(self) -> list[str]:
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_openend_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )
        
        for ens_idx in range(self.ensemble_size):
            
            prompt = templates[ens_idx].format( budget_item=f'{{target_budget_item}}' ,  indicator=f'{{target_indicator}}', effect_order=self.effect_order_str ).replace('  ',' ')
            prompt = prompt + "\nA:\n\n"

            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                context_k = "Q:\n\n"+templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}' ,  indicator=f'{{indicator_{k}}}', effect_order=self.effect_order_str ).replace('  ',' ')
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
                format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":d['label'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} ) 
                    

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
        
        template_responses = copy.deepcopy( li_prompts_openend_template_open_response[:self.ensemble_size] )
        li_li_prompts = []
        for row in batch:
            
            li_prompts = []
            for ens_idx in range(self.ensemble_size):
                if self.k_shot > 0:
                    # Fill in the k_shot context with random extracts from dataset
                    
                    # Sample math.ceil(k/2) positive and math.floor(k/2) negative examples
                    pos_examples_sample = random.sample( [d for d in self.examples_dset if d['label']=='Yes'], math.ceil(self.k_shot/2) )
                    neg_examples_sample = random.sample( [d for d in self.examples_dset if d['label']=='No'], math.floor(self.k_shot/2) )
                    
                    # Creating the open ended answer version of the examples
                    pos_examples_open_ended_answer = [ template_responses[ens_idx]['Yes'].format(budget_item=d['budget_item'], indicator=d['indicator'], effect_order=self.effect_order_str) for  d in pos_examples_sample ]
                    neg_examples_open_ended_answer = [ template_responses[ens_idx]['No'].format(budget_item=d['budget_item'], indicator=d['indicator'], effect_order=self.effect_order_str) for d in neg_examples_sample ]

                    # python shuffle two lists in the same order 
                    li_examples = list(zip( list(pos_examples_sample) + list(neg_examples_sample), list(pos_examples_open_ended_answer) + list(neg_examples_open_ended_answer) ))
                    random.Random(48).shuffle(li_examples)

                    examples_sample, examples_open_ended_answer = zip(*li_examples)

                    # Creating the format dict for all examples
                    format_dict =  reduce(operator.ior, ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} ) # type: ignore
                else:
                    format_dict = {}

                ## filling context examples in template
                prompt =  templates[ens_idx].format(target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                                                    **format_dict).replace('  ',' ')
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts
