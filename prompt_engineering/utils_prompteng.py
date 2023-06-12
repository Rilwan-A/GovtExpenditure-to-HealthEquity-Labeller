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
import random
map_relationship_promptsmap ={}

# region budgetitem to indicator templates
li_prompts_yes_no_template = [    
    'Does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',

    # "Is local government spending on \"{budget_item}\" {effect_type} related to the state of \"{indicator}\"?",
    
    # 'Give me a yes or no answer to the following question, Does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',
    
    # 'Is the state of \"{indicator}\" {effect_type} related to local government spending on \"{budget_item}\"?',

    # 'Does local government spending on \"{budget_item}\" {effect_type} improve the level of \"{indicator}\"?',    
]

li_prompts_open_template = [    

    'Does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',

    # 'Is local government spending on \"{budget_item}\" {effect_type} related to the state of \"{indicator}\"?',

    # 'Does local government spending on \"{budget_item}\" {effect_type} relate to the level of \"{indicator}\"?'

    # 'Is the state of \"{indicator}\" {effect_type} related to local government spending on \"{budget_item}\"?',
    
    # 'Does local government spending on \"{budget_item}\" {effect_type} improve the level of \"{indicator}\"?'
]

li_prompts_open_template_open_response =[

    {'Yes':'Local government spending on \"{budget_item}\" does {effect_type} affect \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {effect_type} affect \"{indicator}\".'},

    # {'Yes':'Local government spending on \"{budget_item}\" is {effect_type} related to the state of \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" is not {effect_type} related to the state of \"{indicator}\".'},


    # {'Yes':'The state of \"{indicator}\" is {effect_type} related to local government spending on \"{budget_item}\".', 'No':'The state of \"{indicator}\" is not {effect_type} related to local government spending on \"{budget_item}\".'},

    # {'Yes':'Local government spending on \"{budget_item}\" does {effect_type} improve the level of \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {effect_type} improve the level of \"{indicator}\".'},
]


# region Prompts for the open prompt style methodology
# open_response_cats = { 'A':'Is Related', 'B':'Is Not Related', 'C':'Not Sure' }
# open_response_labels = { 'A':'Yes', 'B':'No', 'C':'NA'}
# open_response_cats = { 'A':'Does Affect', 'B':'Does Not Affect', 'C':'Not Sure' }

open_response_cats = { '1':'Spending On Government Budget Item Does Affect Indicator', '2':'Spending On Government Budget Item Does Not Affect Indicator' }
# open_response_cats = { '1': 'Spending on "{budget_item}" does affect "{indicator}"' , '2':'Spending on "{budget_item}" Does Not Affect "{indicator}"' }
# NOTE: using full label showed promise
open_response_labels = { '1':'Yes', '2':'No'}


# V2 encourages the response the have the category letter as a response 
li_prompts_categories_answer_v1: list[str] = [
    # "Below is a list of \"Categories\" and a \"Statement\" regarding whether local government spending on a government budget item has a causal relationship with a socio-economic/health indicator. Please select the category, that best describes the relationship between the government budget item and socio-economic/health indicator.\n\"Categories\":\n- A Relationship Exists\n- No Relationship Exists\n- Indetermined\n\"Statement\": {statement}"

    # "Select the letter that best categorizes the claim made in the statement regarding whether or not there is a causal link between local government spending on a particular budget item and a socio-economic or health indicator. The statement will be provided to you, and you must choose from the following categories: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your answer should consist of the letter corresponding to the most appropriate category.\n Answer: ",

    # "Please choose the letter that accurately classifies the assertion made in the statement regarding the potential causal relationship between local government spending on a specific budget item and a socio-economic or health indicator. The statement will be presented to you, and you must select one of the three categories provided: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your response should consist of the letter that corresponds to the most suitable classification."

    # "Please evaluate the statement provided, which discusses a potential causal link between local government spending on a specific budget item and a socio-economic or health indicator. Based on the information in the statement, classify the relationship into one of the following categories: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your response should be the letter that best represents your classification."

    # f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" is related to a "socio-economic/health indicator". Classify the statement\'s opinion into one of the following categories and respond only with the letter of the selected category: A) {open_response_cats["A"]}, B) {open_response_cats["B"]}, or C) {open_response_cats["C"]}.\nStatement: {"{statement}"}',

    # f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" affects a "socio-economic/health indicator". Classify the statement\'s opinion into one of the following categories and respond only with the letter (A, B or C) of the selected category: A) {open_response_cats["A"]}, B) {open_response_cats["B"]}, or C) {open_response_cats["C"]}.\nStatement: {"{statement}"}'
    # NOTE: All the above prompts included a NA category e.g. if the model was not sure. The issue was that the NA category always attracted too much weight during prediction so we removed it.
    # NOTE: All the above prompts included a letters for the category labels, issue with this is that when using perplexity method then the perplexity of category labels can also include probability of the model produce open answers that start with label lettter.

    f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" affects a "socio-economic/health indicator". Classify the statement\'s opinion using one of the following categories and respond only with the number (1 or 2) of the selected category: 1) {open_response_cats["1"]}, 2) {open_response_cats["2"]}.\nStatement: {"{statement}"}'
]

# V2 encourages the response the have the category name as a response
li_prompts_categories_answer_v2: list[str] = [
    # "Below is a list of \"Categories\" and a \"Statement\" regarding whether local government spending on a government budget item has a causal relationship with a socio-economic/health indicator. Please select the category, that best describes the relationship between the government budget item and socio-economic/health indicator.\n\"Categories\":\n- A Relationship Exists\n- No Relationship Exists\n- Indetermined\n\"Statement\": {statement}"

    # "Select the letter that best categorizes the claim made in the statement regarding whether or not there is a causal link between local government spending on a particular budget item and a socio-economic or health indicator. The statement will be provided to you, and you must choose from the following categories: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your answer should consist of the letter corresponding to the most appropriate category.\n Answer: ",

    # "Please choose the letter that accurately classifies the assertion made in the statement regarding the potential causal relationship between local government spending on a specific budget item and a socio-economic or health indicator. The statement will be presented to you, and you must select one of the three categories provided: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your response should consist of the letter that corresponds to the most suitable classification."

    # "Please evaluate the statement provided, which discusses a potential causal link between local government spending on a specific budget item and a socio-economic or health indicator. Based on the information in the statement, classify the relationship into one of the following categories: A) A Relationship Exists, B) No Relationship Exists, or C) Relationship Indeterminate. Your response should be the letter that best represents your classification."

    # f"Please evaluate the statement provided, which discusses a potential causal relationship between local government spending on a specific budget item and a socio-economic or health indicator. Classify the statement into one of the following categories: A) {open_response_cats['A']}, B) {open_response_cats['B']}, or C) {open_response_cats['C']}. Please provide your answer as the category that best fits your classification. \nStatement: {'{statement}'}",

<<<<<<< HEAD
    f"The statement below expresses an opinion on whether local government spending on a 'specific budget item' is related to a 'socio-economic/health indicator'. Classify the statement's opinion into one of the following categories and respond only with the selected category: A) {open_response_cats['A']}, B) {open_response_cats['B']}, or C) {open_response_cats['C']}.\nStatement: {'{statement}'}\nASSISTANT:"
=======
    # f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" is related to a "socio-economic/health indicator". Classify the statement\'s opinion into one of the following categories and respond only with the selected category: A) {open_response_cats["A"]}, B) {open_response_cats["B"]}, or C) {open_response_cats["C"]}.\nStatement: {"{statement}"}'

    # f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" affects a "socio-economic/health indicator". Classify the statement\'s opinion into one of the following categories and respond only with the selected category: A) {open_response_cats["A"]}, B) {open_response_cats["B"]}, or C) {open_response_cats["C"]}.\nStatement: {"{statement}"}'

    f'The statement below expresses an opinion on whether local government spending on a specific "government budget item" affects a "socio-economic/health indicator". Classify the statement\'s opinion using one of the following categories and respond only with the selected category: 1) {open_response_cats["1"]}, 2) {open_response_cats["2"]}.\nStatement: {"{statement}"}'
>>>>>>> origin/main
]
# endregion

# region Prompts for the category prompt style methodology
# categorise_cats = { '1':'Spending On "Government Budget Item" Does Affect "Socio-Economic/Health Indicator"' , '2':'"Spending On Government Budget Item" Does Not Affect "Socio-Economic/Health Indicator"' }
categorise_cats = { '1': 'Spending on "{budget_item}" does affect "{indicator}"' , '2':'Spending on "{budget_item}" Does Not Affect "{indicator}"' }
categorise_response_labels = { '1':'Yes', '2':'No' }

li_prompts_categorise_template: list[str] = [
    # f'Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"? Please answer the question using one of the following categories and respond only with the number (1 or 2) of the selected category: 1) {categorise_cats["1"]}, 2) {categorise_cats["2"]}.'

    f'Please answer the following question using one of the following categories and respond only with the number (1 or 2) of the selected category.\nCategories:\n1) {categorise_cats["1"]}\n2) {categorise_cats["2"]}. \nQuestion: Does local government spending on "{{budget_item}}" {{effect_type}} affect "{{indicator}}"?'
]
# endregion

# region Prompts for the cot prompt style methodology
li_prompts_cot_categorise_template: list[str] = [
    'Does local government spending on \"{budget_item}\" {effect_type} affect \"{indicator}\"?',
]



budgetitem_to_indicator_prompts = {
    'li_prompts_yes_no_template':li_prompts_yes_no_template,

    'li_prompts_open_template':li_prompts_open_template,
    'li_prompts_open_template_open_response':li_prompts_open_template_open_response,
    'li_prompts_categories_answer_v1':li_prompts_categories_answer_v1,
    'li_prompts_categories_answer_v2':li_prompts_categories_answer_v2,
    

    'li_prompts_categorise_template':li_prompts_categorise_template,
    
    'li_prompts_cot_categorise_template':li_prompts_cot_categorise_template
}
map_relationship_promptsmap['budgetitem_to_indicator'] = budgetitem_to_indicator_prompts
#endregion

# region indicator to indicator templates
li_prompts_yes_no_template_i2i = [
    "Does the level of  \"{indicator1}\" {effect_type} influence the state of \"{indicator2}\"?",
    
    'Does local government spending on improving the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\" ?, yes or no',
    
    'Is it true that the level of \"{indicator1}\" is {effect_type} related to the level of \"{indicator2}\"?',
    
    'Do improvements in {indicator1} {effect_type} affect \"{indicator2}\"?, Yes or No',
    
    'Answer the following question with yes or no: Does local government spending aimed at affecting \"{indicator1}\" {effect_type} affect \"{indicator2}\"?'

] 
li_prompts_categories_answer_i2i = [
]
li_prompts_open_template_i2i = [
    'Does the level of \"{indicator1}\" {effect_type} influence the state of \"{indicator2}\"?',

    'Does local government spending on improving the level of \"{indicator1}\" {effect_type} affect the level of \"{indicator2}\"?',

    'Is the level of \"{indicator1}\" {effect_type} related to the level of \"{indicator2}\"?',

    'Do improvements in {indicator1} {effect_type} affect \"{indicator2}\"?',

    'Does local government spending aimed at affecting \"{indicator1}\" {effect_type} affect \"{indicator2}\"?'

]
li_prompts_open_template_open_response_i2i = [
    {'Yes':'The level of \"{indicator1}\" is {effect_type} influential to the state of \"{indicator2}\".', 'No':'The level of \"{indicator1}\" is not {effect_type} influential to the state of \"{indicator2}\".'},

    {'Yes':'Local government spending on improving the level of \"{indicator1}\" does {effect_type} affect the level of \"{indicator2}\".', 'No':'Local government spending on improving the level of \"{indicator1}\" does not {effect_type} affect the level of \"{indicator2}\".'},

    {'Yes':'The level of \"{indicator1}\" is {effect_type} related to the level of \"{indicator2}\".', 'No':'The level of \"{indicator1}\" is not {effect_type} related to the level of \"{indicator2}\".'},

    {'Yes':'Improvements in {indicator1} do {effect_type} affect \"{indicator2}\".', 'No':'Improvements in {indicator1} do not {effect_type} affect \"{indicator2}\".'},

    {'Yes':'Local government spending aimed at affecting \"{indicator1}\" does {effect_type} affect \"{indicator2}\".', 'No':'Local government spending aimed at affecting \"{indicator1}\" does not {effect_type} affect \"{indicator2}\".'}

]

indicator_to_indicator_prompts = {
    'li_prompts_yes_no_template_i2i':li_prompts_yes_no_template_i2i,
    'li_prompts_open_template_i2i':li_prompts_open_template_i2i,
    'li_prompts_open_template_open_response_i2i':li_prompts_open_template_open_response_i2i,
    'li_prompts_categories_answer_i2i':li_prompts_categories_answer_i2i
}
map_relationship_promptsmap['indicator_to_indicator'] = indicator_to_indicator_prompts
# endregion

# region SystemMessages
system_prompt_b2i_arbitrary = 'You are a socio-economic researcher tasked with answering a question about whether a "government budget item" affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'
system_prompt_b2i_directly = 'You are a socio-economic researcher tasked with answering a question about whether a "government budget item" directly affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'
system_prompt_b2i_indirectly = 'You are a socio-economic researcher tasked with answering a question about whether a "government budget item" indirectly affects a "socio-economic/health indicator". In the question the government budget item and socio-economic/health indicator will be presented within quotation marks.'
system_prompt_i2i = 'You are an analyst tasked with determining if there\'s a causal relationship between a specific "socio-economic/health indicator" and another "socio-economic/health indicator". Both socio-economic/health indicators will be presented within quotation marks as "indicator1" and "indicator2". Your analysis should consider potential direct and indirect impacts, as well as confounding factors that could influence this relationship. Use your expertise to provide the correct answer to the following question. Please make sure to only evaluate for a causal relationship in the direction implied by the question.'

map_system_prompts_b2i = {
    'arbitrary':system_prompt_b2i_arbitrary,
    'directly':system_prompt_b2i_directly,
    'indirectly':system_prompt_b2i_indirectly,
    'yes_no':'Answer the following question with a yes or no.',
    'open':'Please use your expertise to answer the following question with a conclusive, one sentence answer.',
    'categorise':'',
    # 'cot':'Please use your expertise to provide a detailed four sentence answer to the following question.'
    'cot':'Using your expert knowledge, please provide a thorough, detailed and conclusive four sentence answer to the following question.'

}


map_system_prompts_i2i = {
    'indirectly':system_prompt_i2i,
    'directly':system_prompt_i2i,
    'arbitrary':system_prompt_i2i,
    'yes_no':'Please provide a Yes or No answer the following question.',
    'open':'Please use your expertise to answer the following question with a very short one sentence answer.',
}

map_relationship_system_prompt = {
    'budgetitem_to_indicator':map_system_prompts_b2i,
    'indicator_to_indicator':map_system_prompts_i2i
}

system_prompt_parse_outp_categories_rules_b2i = None
system_prompt_parse_outp_categories_rules_i2i = None

map_relationship_sysprompt_categoriesanswer = {
    'budgetitem_to_indicator':system_prompt_parse_outp_categories_rules_b2i,
    'indicator_to_indicator':system_prompt_parse_outp_categories_rules_i2i
}
# endregion

# region BaseModelFormat - The format required by the underlying language model
format_vicuna_1_1 = "USER: {system_message} {user_message}\nASSISTANT:"
format_vicuna_1_1_no_sysmessage = "USER: {user_message}\nASSISTANT:"
format_alpaca = "### Instruction:\n{system_message}\n\n### Input:\n{user_message}\n\n### Response:\n"
format_alpaca_no_sysmessage = "### Input:\n{user_message}\n\n### Response:\n"
format_mpt = "{system_message}\n\n{user_message}\n\n"
format_mpt_no_sysmessage = "{user_message}\n\n"

def map_llmname_input_format(llm_name, user_message, system_message=None):

    assert user_message is not None

    if 'vicuna' in llm_name and system_message is not None:
        template = format_vicuna_1_1
    elif 'vicuna' in llm_name and system_message is None:
        template = format_vicuna_1_1_no_sysmessage
    
    elif 'alpaca' in llm_name and system_message is not None:
        template = format_alpaca
    elif 'alpaca' in llm_name and system_message is None:
        template = format_alpaca_no_sysmessage
    
    elif 'guanaco' in llm_name and system_message is not None:
        template = format_alpaca
    elif 'guanaco' in llm_name and system_message is None:
        template = format_alpaca_no_sysmessage
    
    elif 'mpt' in llm_name and system_message is not None:
        template = format_mpt
    elif 'mpt' in llm_name and system_message is None:
        template = format_mpt_no_sysmessage
    
    else:
        raise ValueError(f'Unknown llm_name: {llm_name}')

    if system_message is not None:
        template = template.format(system_message=system_message, user_message=user_message)
    else:
        template = template.format(user_message=user_message)
    
    return template

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

def perplexity_for_category(
    data, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, max_length=None, deepspeed_compat:bool=False, category_token_len=1):

    """Calculate the perplexity of the final token for a given set of sentences"""
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

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

        # Slice to get only the last positions which is the category label:
        shift_logits = shift_logits[..., -category_token_len:, :]
        shift_labels = shift_labels[..., -category_token_len:]
        shift_attention_mask_batch = shift_attention_mask_batch[..., -category_token_len:]

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

def joint_probabilities_for_category(
    data, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, max_length=None, deepspeed_compat:bool=False, category_token_len=1):


    """For a given prompt taking the style of "Answer with the letter of the Category which best answers my question", This function returns the joint probabilities for the category tokens in each posible answer,
        NOTE: by design the category responses must all be the same length, ideally 1 token length.

        NOTE: However the function is currently written to work on sequencs longer than 1 token, but this is not recommended.
    """

    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from torch.nn.functional import log_softmax

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert category_token_len == 1, "Currently only supports category tokens of length 1"

    model = model
    tokenizer = tokenizer

    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
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

    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    joint_probs = []

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

        with torch.no_grad() if not deepspeed_compat else nullcontext():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_attention_mask_batch = attn_mask[..., 1:]

        shift_logits = shift_logits[..., -category_token_len:, :]
        shift_labels = shift_labels[..., -category_token_len:]
        shift_attention_mask_batch = shift_attention_mask_batch[..., -category_token_len:]

        if deepspeed_compat is False:
            shift_logits = shift_logits.contiguous()
            shift_labels = shift_labels.contiguous()
            shift_attention_mask_batch = shift_attention_mask_batch.contiguous()


        # Calculate probabilities from logits
        log_probs  = log_softmax(shift_logits, dim=-1)

        # Use gather to select the log probabilities for the actual tokens
        gathered_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        gathered_log_probs = gathered_log_probs * shift_attention_mask_batch

        # Sum the log probabilities for the actual tokens to get the joint log probability
        joint_log_prob_batch = gathered_log_probs.sum(dim=-1)

        joint_prob_batch = torch.exp(joint_log_prob_batch)

        joint_probs += joint_prob_batch.tolist()

    return joint_probs

def perplexity_to_normalised_probability( perplexities: dict[str,float]) -> dict[str,float]:

    """Converts a dictionary of perplexity scores to normalised probabilities"""
    # Convert perplexity to probabilities
    probs = {}
    for k,v in perplexities.items():
        probs[k] = 1/v

    # Normalise probabilities
    total = sum(probs.values())
    for k,v in probs.items():
        probs[k] = v/total

    return probs

def nomalized_probabilities( probs: dict[str,float]) -> dict[str,float]:
    
        """Normalises a dictionary of probabilities"""
        # Normalise probabilities
        total = sum(probs.values())
        for k,v in probs.items():
            probs[k] = v/total
    
        return probs

class PromptBuilder():
    def __init__(self, prompt_style:str, k_shot:int,
                 ensemble_size:int, 
                 examples_dset:list[dict]|None=None, 
                 effect_type:str="arbitrary", 
                 relationship:str="budgetitem_to_indicator",
                seed:int=10  ) -> None:
        
        assert prompt_style in ['yes_no','open','categorise','cot']
        assert effect_type in [ 'arbitrary', 'directly', 'indirectly'], "Effect order must be either arbitrary, directly or indirectly"
        assert relationship in ['budgetitem_to_indicator', 'indicator_to_indicator'], "Relationship must be either budgetitem_to_indicator or indicator_to_indicator"
        assert k_shot <= len(examples_dset) if examples_dset is not None else True, "User can not create a K-shot context with more examples than the number of examples in the dataset"

        if prompt_style == 'cot':
            assert k_shot == 0, "K-shot must be 0 for cot prompts"


        self.prompt_style = prompt_style
        self.k_shot = k_shot    # Number of examples to use as context for each prompt
        self.ensemble_size = ensemble_size # Number of different prompts to use per prediction
        self.examples_dset = examples_dset
        self.effect_type = '' if effect_type == 'arbitrary' else effect_type 
        self.relationship = relationship
        random.seed(seed)
        # when arbitrary is subbed into the prompt template, it will result in a double space in the prompt. We use .replace("  ", " ") to remove this


    def __call__(self, batch:list[dict]) -> list[list[str]]:
        """
        Given a batch of examples, this function returns a list of prompts for each example in the batch
        """

        # First we generate an ensemble of templates to be filled in for each element in the batch
        if self.prompt_style == 'yes_no':
            templates = self._yes_no_template()
        elif self.prompt_style == 'open':
            templates = self._open_template()
        elif self.prompt_style == 'categorise':
            templates = self._categorise_template()
        elif self.prompt_style == 'cot':
            templates: list[str] = self._cot_template()
        else:
            raise ValueError('Invalid prompt_style: ' + self.prompt_style)

        # Second given a k_shot prompt template, we then create n = ensemble_size, realisations of the template by sampling from the training set
        if self.prompt_style in ['yes_no']:
            li_li_prompts = self.fill_template_yesno(templates, batch)
        elif self.prompt_style in ['open']:
            li_li_prompts = self.fill_template_open(templates, batch)
        elif self.prompt_style in ['categorise']:
            li_li_prompts = self.fill_template_categorise(templates, batch)
        elif self.prompt_style in ['cot']:
            li_li_prompts = self.fill_template_cot(templates, batch)
        
        else:
            li_li_prompts = []
            
        return li_li_prompts
    
    def _yes_no_template(self) -> list[str]:
        # This creates sets of prompt templates, one prompt set for each prediction and M different prompts within each prompt set
        # When producing a prompt set of size M<N we randomly sample 
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        
        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_yes_no_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):
            
            # This handles the actual question excl the k_shot context
            if self.relationship == 'budgetitem_to_indicator':
                if self.k_shot >0:
                    prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ') +"\nAnswer: "
                else:
                    prompt = templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ')
                
            elif self.relationship == 'indicator_to_indicator':
                prompt = "Question: "+templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type ).replace('  ',' ') +"\nAnswer: "

            # Add k_shot context to prompt
            for k in reversed(range(self.k_shot)):
                if self.relationship == 'budgetitem_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}',  indicator=f'{{indicator_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."
                elif self.relationship == 'indicator_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx].format( indicator1=f'{{indicator1_{k}}}',  indicator2=f'{{indicator2_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."
                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates
    
    def _open_template(self) -> list[str]:
        # This creates sets of prompt templates, one prompt set for each prediction and M different prompts within each prompt set
        # When producing a prompt set of size M<N we randomly sample
        # For each member of the ensemble we then extend the prompt to have self.k_shots context
        # This output leaves gaps for budget_items, indicators, and responses ot be filled in 

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_open_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )
        
        for ens_idx in range(self.ensemble_size):
            if self.relationship == 'budgetitem_to_indicator':
                if self.k_shot >0:
                    prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ') + "\nAnswer: "
                else:
                    prompt = templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ')

            elif self.relationship == 'indicator_to_indicator':
                if self.k_shot >0:
                    prompt = "Question: "+templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type ).replace('  ',' ') + "\nAnswer: "
                else:
                    prompt = templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type ).replace('  ',' ')
                
            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                if self.relationship == 'budgetitem_to_indicator':
                    context_k = "Question: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}', indicator=f'{{indicator_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."
                elif self.relationship == 'indicator_to_indicator':
                    context_k = "Question: " +templates[ens_idx].format( indicator1=f'{{indicator1_{k}}}', indicator2=f'{{indicator2_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nAnswer: {{answer_{k}}}."

                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt

        return templates

    def _categorise_template(self)  -> list[str]:

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_categorise_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):

            if self.relationship == 'budgetitem_to_indicator':
                if self.k_shot >0:
                    prompt = "Question: "+templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ') + "\nAnswer: "
                else:
                    prompt = templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ')
            
            elif self.relationship == 'indicator_to_indicator':
                prompt = templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type ).replace('  ',' ')
            
            # Add k_shot context
            for k in reversed(range(self.k_shot)):
                if self.relationship == 'budgetitem_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx].format( budget_item=f'{{budget_item_{k}}}', indicator=f'{{indicator_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."
                elif self.relationship == 'indicator_to_indicator':
                    context_k = "Example Question {k}: " +templates[ens_idx].format( indicator1=f'{{indicator1_{k}}}', indicator2=f'{{indicator2_{k}}}', effect_type=self.effect_type ).replace('  ',' ') + f"\nExample Answer {k}: {{answer_{k}}}."

                prompt = context_k + "\n\n"+prompt
            
            templates[ens_idx] = prompt
        return templates
    
    def _cot_template(self)  -> list[str]:

        li_prompts = map_relationship_promptsmap[self.relationship]['li_prompts_cot_categorise_template']
        templates = copy.deepcopy( sample(li_prompts, self.ensemble_size)  )

        for ens_idx in range(self.ensemble_size):

            if self.relationship == 'budgetitem_to_indicator':
                prompt = templates[ens_idx].format( budget_item='{target_budget_item}',  indicator='{target_indicator}', effect_type=self.effect_type ).replace('  ',' ')
            
            elif self.relationship == 'indicator_to_indicator':
                prompt = templates[ens_idx].format( indicator1='{target_indicator1}',  indicator2='{target_indicator2}', effect_type=self.effect_type ).replace('  ',' ')
                        
            templates[ens_idx] = prompt
        return templates

    def fill_template_yesno(self, templates:list[str], batch:list[dict]) -> list[list[str]]:
        """Fill in the template with the target and k_shot context"""

        li_li_prompts = []

        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            for ens_idx in range(self.ensemble_size):
                # This indented code section fills in the k_shot context with random extracts from dataset

                # sample k items from our train set into a format dict for the template
                if self.k_shot == 0:
                    format_dict = {}
                elif self.relationship == 'budgetitem_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":d['label'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} ) 
                elif self.relationship == 'indicator_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}":d['label'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} )
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                        **format_dict
                    )
                elif self.relationship == 'indicator_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_indicator1= row['indicator1'], target_indicator2=row['indicator2'],
                        **format_dict
                    )
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts

    def fill_template_open(self, templates:list[str], batch:list[dict])->list[list[str]]:
        li_answer_templates = map_relationship_promptsmap[self.relationship]['li_prompts_open_template_open_response']
        template_responses = copy.deepcopy( sample(li_answer_templates, self.ensemble_size)  )

        li_li_prompts = []
        for row in batch:
            
            li_prompts = []
            for ens_idx in range(self.ensemble_size):
                
                ## filling k-shot examples in template
                if self.k_shot > 0:
                    # Fill in the k_shot context with random extracts from dataset
                    
                    # Sample math.ceil(k/2) positive and math.floor(k/2) negative examples
                    pos_examples_sample = random.sample( [d for d in self.examples_dset if d['label']=='Yes'], math.ceil(self.k_shot/2) )
                    neg_examples_sample = random.sample( [d for d in self.examples_dset if d['label']=='No'], math.floor(self.k_shot/2) )
                    
                    # Creating the open ended answer version of the examples
                    if self.relationship == 'budgetitem_to_indicator':
                        pos_examples_open_ended_answer = [ template_responses[ens_idx]['Yes'].format(budget_item=d['budget_item'], indicator=d['indicator'], effect_type=self.effect_type).replace('  ',' ') for  d in pos_examples_sample ]
                        neg_examples_open_ended_answer = [ template_responses[ens_idx]['No'].format(budget_item=d['budget_item'], indicator=d['indicator'], effect_type=self.effect_type).replace('  ',' ') for d in neg_examples_sample ]
                    elif self.relationship == 'indicator_to_indicator':
                        pos_examples_open_ended_answer = [ template_responses[ens_idx]['Yes'].format(indicator1=d['indicator1'], indicator2=d['indicator2'], effect_type=self.effect_type.replace('  ',' ')) for  d in pos_examples_sample ]
                        neg_examples_open_ended_answer = [ template_responses[ens_idx]['No'].format(indicator1=d['indicator1'], indicator2=d['indicator2'], effect_type=self.effect_type).replace('  ',' ') for d in neg_examples_sample ]

                    # python shuffle two lists in the same order 
                    li_examples = list(zip( list(pos_examples_sample) + list(neg_examples_sample), list(pos_examples_open_ended_answer) + list(neg_examples_open_ended_answer) ))
                    random.shuffle(li_examples)

                    examples_sample, examples_open_ended_answer = zip(*li_examples)

                    # Creating the format dict for all examples
                    if self.relationship == 'budgetitem_to_indicator':
                        format_dict =  reduce(operator.ior, ( { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} ) # type: ignore
                    elif self.relationship == 'indicator_to_indicator':
                        format_dict =  reduce(operator.ior, ( { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}": answer } for idx, (d, answer) in  enumerate( zip( examples_sample, examples_open_ended_answer ) ) ), {} )
                else:
                    format_dict = {}

                # filling in the target info
                if self.relationship == 'budgetitem_to_indicator':
                    prompt =  templates[ens_idx].format(target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                                                    **format_dict).replace('  ',' ')
                elif self.relationship == 'indicator_to_indicator':
                    prompt =  templates[ens_idx].format(target_indicator1= row['indicator1'], target_indicator2=row['indicator2'],
                                                    **format_dict).replace('  ',' ')
                
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts

    def fill_template_categorise(self, templates:list[str], batch:list[dict])->list[list[str]]:

        """Fill in the template with the target and k_shot context"""

        li_li_prompts = []

        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            categorise_response_labels_inverted = {v:k for k,v in categorise_response_labels.items()}
            for ens_idx in range(self.ensemble_size):
                # This indented code section fills in the k_shot context with random extracts from dataset

                if self.k_shot == 0:
                    format_dict = {}
                # sample k items from our train set into a format dict for the template
                elif self.relationship == 'budgetitem_to_indicator':
                    format_dict = reduce( operator.ior, [ { f'budget_item_{idx}':d['budget_item'], f"indicator_{idx}":d['indicator'], f"answer_{idx}":categorise_response_labels_inverted[d['label']] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} ) 
                elif self.relationship == 'indicator_to_indicator':
                    raise NotImplementedError("Categorise response not implemented for indicator_to_indicator")
                    format_dict = reduce( operator.ior, [ { f'indicator1_{idx}':d['indicator1'], f"indicator2_{idx}":d['indicator2'], f"answer_{idx}":['label'] } for idx, d in  enumerate(random.sample(self.examples_dset, self.k_shot) ) ], {} )
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                        **format_dict
                    )
                elif self.relationship == 'indicator_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_indicator1= row['indicator1'], target_indicator2=row['indicator2'],
                        **format_dict
                    )
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts

    def fill_template_cot(self, templates:list[str], batch:list[dict])->list[list[str]]:

        li_li_prompts = []

        # for each row in batch
        for row in batch:
            
            li_prompts = []
            prompt = None
            # for each member of the ensemble (note each ensemble member has a different prompt template)
            
            for ens_idx in range(self.ensemble_size):
                    
                ## filling context examples in template and target info
                if self.relationship == 'budgetitem_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_budget_item= row['budget_item'], target_indicator=row['indicator'],
                        
                    )
                elif self.relationship == 'indicator_to_indicator':
                    prompt = templates[ens_idx].format(
                        target_indicator1= row['indicator1'], target_indicator2=row['indicator2'],
                        
                    )
                li_prompts.append(prompt)

            # Add prompt to list
            li_li_prompts.append(li_prompts)
        
        return li_li_prompts
