# Collect the information for the language model to use
import os
from argparse import ArgumentParser
from prompt_engineering import predict 

# Define Model Kwargs For Models to Test
kwargs_7bn = {
    'exp_group':'ppi_b2i_7bn',
    'exp_name':'sbeluga7b',

    'finetune_version':4,
    'finetuned':True,

    'llm_name':'stabilityai/StableBeluga-7B',
    'local_or_remote':'local',

    "unbias_categorisations": False,
}

kwargs_13bn = {
    'exp_group':'ppi_b2i_13bn',
    'exp_name':'sbeluga13b',

    'finetune_version':0,
    'finetuned':True,

    'llm_name':'stabilityai/StableBeluga-13B',
    'local_or_remote':'local',

    "unbias_categorisations": False,

}

kwargs_30bn = {
    'exp_group':'ppi_b2i_30bn',
    'exp_name':'upllama30b',

    'finetuned':False,
    'finetune_version':None,

    'llm_name':'upstage/llama-30b-instruct-2048',
    'local_or_remote':'local',

    "unbias_categorisations": False,
}

exps_kwargs = [kwargs_7bn, kwargs_13bn, kwargs_30bn]

# Import variable kwargs. e.g. whether to use CPUQ and Verbalize
parser = ArgumentParser()
parser.add_argument('--exp_idx', type=int, choices=[0,1,2], default=0)
parser.add_argument('--cpuq_verbalize', type=str, choices=['cpuq','verbalize'] )
parser.add_argument('--debugging', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--finetune_dir', type=str)
parse_kwargs = parser.parse_args()

model_kwargs = exps_kwargs[parse_kwargs.exp_idx]
if parse_kwargs.cpuq_verbalize == 'cpuq':
    prompt_style = 'categorise'
    parse_style = 'categories_perplexity'
    model_kwargs['exp_group'] = model_kwargs['exp_group'] + '_cpuq'

elif parse_kwargs.cpuq_verbalize == 'verbalize':
    prompt_style = 'yes_no'
    parse_style = 'rules'
    model_kwargs['exp_group'] = model_kwargs['exp_group'] + '_verbalize'



# Call the predict.py script with the correct arts
general_exp_kwargs  = {

    'predict_b2i': True,
    'predict_i2i': False,

    'prompt_style': prompt_style,
    'parse_style': parse_style,
    'effect_type': 'arbitrary',
    
    'edge_value': 'binary_weight',

    'input_file':os.path.join('./data','ppi','b2i_networks','b2i_candidates.csv'),

    "batch_size":parse_kwargs.batch_size,
    "ensemble_size": 1,
    "k_shot_b2i": 0,
    "k_shot_example_dset_name_b2i": None,
    "k_shot_example_dset_name_i2i": None,
    "k_shot_i2i": 0,

    "save_output":True,
    "debugging":parse_kwargs.debugging,

    
    "finetune_dir":parse_kwargs.finetune_dir
   
}

predict.main(
    **model_kwargs,
    **general_exp_kwargs
)