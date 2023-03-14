import pandas as pd

li_prompts_yes_no_template = [    
    "Give me a Yes or No answer to the following question, is local government spending on \"\"{budget_item}\"\" related to \"{indicator}\"?",
    
    'Does local government spending on \"{budget_item}\" affect \"{indicator}\"?, True or False',
    
    'Is it true that \"{indicator}\" related to local government spending on \"{budget_item}\"?',
    
    'Does \"{budget_item}\" affect \"{indicator}\"?, Yes or No',
    
    'Answer the following question with True or False: Does local government spending on \"{budget_item}\" affect \"{indicator}\"?',
    
]

li_prompts_openend_template = [    
    'Is local government spending on \"{budget_item}\" related to \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" affect \"{indicator}\"?',
    
    'Is \"{indicator}\" related to local government spending on \"{budget_item}\"?',
    
    'local goernment spending on \"{budget_item}\" improves \"{indicator}\"?',
    
    'Does local government spending on \"{budget_item}\" affect \"{indicator}\"?',
    
]

li_prompts_openend_template_open_response =[
    {'Yes':'Local government spending on \"{budget_item}\" is related to \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" is not related to \"{indicator}\".'},

    {'Yes':'Local government spending on \"{budget_item}\" does affect \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not affect \"{indicator}\".'},

    {'Yes':'\"{indicator}\" is related to local government spending on {budget_itme}.', 'No':'\"{indicator}\" is not related to local government spending on {budget_itme}.'},

    {'Yes':'Local government spending on \"{budget_item}\" does improve \"{indicator}\".', 'No':'Local government spending on \"{budget_item}\" does not improve \"{indicator}\".'},

    {'Yes':'A local government can effect \"{indicator}\" by spending on \"{budget_item}\".', 'No':'A local government can not effect \"{indicator}\" by spending on \"{budget_item}\".'},
]

li_prompts_parse_yesno_from_answer = [
    "Select the category that best describes the statement. \n\" Categories \":\n- Disagree\n- Agree\nStatement : {statement}"
]

def create_negative_examples(dset:pd.DataFrame) -> pd.DataFrame:
    # Create negative examples by randomly selecting a budget item and indicator
    # from the dataset and then swapping them
    # 
    # dset: pd.DataFrame
    #     The dataset to create negative examples from
    # 
    # Returns
    # -------
    # pd.DataFrame
    #     The dataset with negative

    l = len(dset)
    # Each budget_item has n records
    # For each budget_item we sample min(n,l-n) false examples
    # These false examples are created by filtering on other budget_items and then sampling

    for budget_item in dset['budget_item'].unique():
        dset_budget_item = dset[dset['budget_item']==budget_item]
        n = len(dset_budget_item)
        dset_budget_item_neg = dset[dset['budget_item']!=budget_item].sample(min(n,l-n), replace=False)
        
        dset_budget_item_neg['budget_item'] = budget_item
        dset_budget_item_neg['label'] = 'No'
        
        dset = pd.concat([dset, dset_budget_item_neg], axis=0)

    return dset
