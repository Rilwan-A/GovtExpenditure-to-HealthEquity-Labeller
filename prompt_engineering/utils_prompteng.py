import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from contextlib import nullcontext

li_prompts_yes_no_template = [    
    "Give me a Yes or No answer to the following question, is local government spending on \"{budget_item}\" {directly_or_indirectly} related to \"{indicator}\"?",
    
    'Does local government spending on \"{budget_item}\" {directly_or_indirectly} affects \"{indicator}\"?, True or False',
    
    'Is it true that \"{indicator}\" {directly_or_indirectly} related to local government spending on \"{budget_item}\"?',
    
    'Does \"{budget_item}\" {directly_or_indirectly} affects \"{indicator}\"?, Yes or No',
    
    'Answer the following question with True or False: Does local government spending on \"{budget_item}\" {directly_or_indirectly} affects \"{indicator}\"?'
]

li_prompts_openend_template = [    
    'Is local government spending on \"{budget_item}\" {directly_or_indirectly} related to \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" {directly_or_indirectly} affects \"{indicator}\"?',
    
    'Is \"{indicator}\" {directly_or_indirectly} related to local government spending on \"{budget_item}\"?',
    
    'Local government spending on \"{budget_item}\" {directly_or_indirectly} improves \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" {directly_or_indirectly} affects \"{indicator}\"?'
]

li_prompts_openend_template_open_response =[
    {'Yes':'Local government spending on \"{budget_item}\" is {directly_or_indirectly} related to \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" is not {directly_or_indirectly} related to \"{indicator}\".'},

    {'Yes':'Local government spending on \"{budget_item}\" does {directly_or_indirectly} affect \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {directly_or_indirectly} affect \"{indicator}\".'},

    {'Yes':'\"{indicator}\" is {directly_or_indirectly} related to local government spending on \"{budget_item}\".', 'No':'\"{indicator}\" is not {directly_or_indirectly} related to local government spending on \"{budget_item}\".'},

    {'Yes':'Local government spending on \"{budget_item}\" does {directly_or_indirectly} improve \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not {directly_or_indirectly} improve \"{indicator}\".'},

    {'Yes':'A local government can {directly_or_indirectly} affect \"{indicator}\" by spending on \"{budget_item}\".', 'No':'A local government can not {directly_or_indirectly} affect \"{indicator}\" by spending on \"{budget_item}\".'}
]

li_prompts_parse_yesno_from_answer = [
    """Select the grammatical category that best describes the statement.\n\"Categories\":\n- Negation\n- Affirmation\nStatement: {statement}\nThis statement belongs to the category"""
]

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
