# In this script we add edge weight estimations to the i2i predictions made by the llm models.
# We use 3 different methodologies: multionomial_mean, entropy over binomial edge existance (rescaled to 0-1), verbalization of the binomial edge existance (rescaled to 0-1)

import pandas as pd
import os
import json
import yaml
from argparse import ArgumentParser
from scipy.stats import entropy

from prompt_engineering.my_logger import setup_logging_add_i2i_edge_weights

from prompt_engineering.langchain.utils import PredictionGenerator, load_llm
from prompt_engineering.langchain.utils_prompteng import PromptBuilder

exp_dirs = [
    os.path.join('prompt_engineering','output','spot','exp_i2i_7bn_distr', 'exp_sbeluga7b_non_uc'),
    os.path.join('prompt_engineering','output','spot','exp_i2i_13_distr', 'exp_sbeluga13_non_uc'),
    os.path.join('prompt_engineering','output','spot','exp_i2i_30b_distr','exp_upllama30b_non_uc')
]


def main( experiment_dir, debugging=False, batch_size=1, finetune_dir='', scale_max=5 ):
    
    # Set up logging
    logger = setup_logging_add_i2i_edge_weights(debugging)


    df_preds = pd.read_csv(os.path.join( experiment_dir, 'predictions_i2i.csv'))
    config = yaml.safe_load( open( os.path.join( experiment_dir, 'config.yaml'), "r" ) )
    config['finetune_dir'] = finetune_dir

    # Log the config file used
    logger.info("Arguments:")
    for key, value in config.items():
        logger.info(f'\t{key}: {value}')


    indicator1 = df_preds['indicator1'].tolist()
    indicator2 = df_preds['indicator2'].tolist()
    preds = df_preds['pred_aggregated'].apply(json.loads).tolist()

    li_records = [ {'indicator1':ind1, 'indicator2':ind2, 'pred_aggregated':pred} for ind1, ind2, pred in zip(indicator1, indicator2, preds) ]

    # get index of positions where the binomial prediction for 'Yes' is higher than 0.5
    idx_existing_edges = [i for i, x in enumerate(preds) if x['Yes'] >= 0.5]

    # For each indicator pair we produce an ensemble of predictions
    # Retrieve a list of the prediction ensembles list[ list[ dict[str,float] ] ]
    # Retreive the mean of the sets of prediction ensembles list[ dict['mean',float] ]
    multinomial_means, multinomial_distributions = multinomial_edge_weights(li_records, config, scale_max, idx_existing_edges, logger)
    
    # We note that verbalization is samem as argmax multinomial prediction
    verbalization_preds = verbalize_edge_weights(multinomial_distributions)

    # We use the entropy over the binomial distribution as a measure of relationship strength 
    entropy_preds = entropy_edge_weights( li_records, idx_existing_edges)

    # Saving these results to file
    path_mn = os.path.join(experiment_dir, 'i2i_mn_weights.csv')
    path_vb = os.path.join(experiment_dir, 'i2i_vb_weights.csv')
    path_et = os.path.join(experiment_dir, 'i2i_et_weights.csv')

    df_mn = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':multinomial_means})
    df_vb = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':verbalization_preds})
    df_et = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':entropy_preds})

    df_mn.to_csv(path_mn, index=False)
    df_vb.to_csv(path_vb, index=False)
    df_et.to_csv(path_et, index=False)



def multinomial_edge_weights(li_records, config, scale_max, idx_existing_edges=None, logger=None):

    if idx_existing_edges is not None:
        original_len = len(li_records)
        li_records_ = [li_records[i] for i in idx_existing_edges]

    # Load the LLM used to make the predictions
    llm_name = config['llm_name']
    model = load_llm(llm_name, 
                                                        finetuned=config['finetuned'],
                                                        local_or_remote='local',
                                                        finetune_dir=config['finetune_dir'],
                                                        exp_name = config['exp_name'],
                                                        finetune_version = config['finetune_version'],

                                                          )

    # Get the prompt template for querying the strength of the relationship between the two indicators
    relationship_type = 'arbitrary'
    prompt_builder = PromptBuilder(
        model,
        llm_name,
        prompt_style='relationship_strength',
        k_shot = 0,
        ensemble_size = 1,
        effect_type = 'arbitrary',
        relationship_type = 'indicator_to_indicator'
    )

    prediction_generator = PredictionGenerator(
        llm_name,
        prompt_style = 'categories_scale',
        ensemble_size=1,
        edge_value = 'scale_10',
        parse_style = 'categories_perplexity',
        relationship = 'indicator_to_indicator',
        local_or_remote = 'local')
        
    li_pred_agg = []
    li_pred_ensembles = []

    # Running Prompt Generations and Predictions
    li_li_record = [ li_record[i:i+batch_size] for i in range(0, len(li_record), batch_size) ]

    for idx, batch in enumerate(li_li_record):
        if logger is not None:
            current_time = time.time()
            if current_time - last_time > 30:
                logger.info(f'Processing batch {idx} out of {len(li_li_record)}')
                last_time = current_time

        #  Create prompts
        batch_li_li_statement, batch_li_li_discourse = prompt_builder(batch, scale_max=scale_max)

        # Generate predictions
        batch_pred_ensembles = prediction_generator.predict(batch_li_li_statement)

        # Aggregate ensembles into predictions - calculate the mean of the multinomial distribution
        batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles, scale_max=scale_max)

        li_pred_ensembles.extend(batch_pred_ensembles) # type: ignore
        li_pred_agg.extend(batch_pred_agg) # type: ignore
    
    
    if idx_existing_edges is not None:
        # Creating an output that factors in elements which output was not predicted for
        _iter = iter(li_pred_agg)
        li_pred_agg_ = [ next(_iter) if i in idx_existing_edges else None for i in range(original_len) ]

        # Creating an output that factors in elements which output was not predicted for
        _iter = iter(li_pred_ensembles)
        li_pred_ensembles_ = [ next(_iter) if i in idx_existing_edges else None for i in range(original_len) ]

    return li_pred_agg, li_pred_ensembles

def verbalize_edge_weights(li_multinomial_distr_set:list[dict[str,float]] ):
    # Verbalize the multinomial distributions
    verbalization_preds = [None for _ in range((li_multinomial_distr_set))]
    
    for mds in li_multinomial_distr_set:

        if mds is None:
            continue
        # Assuming ensemble size 1, retreive first prediction
        # TODO: implement a better mechanism for handling ensemboles of predictions
        mds_ = mds[0] # dict[str,float]

        # Get the key with the highest value
        key_max = max(mds_, key=mds_.get)

        verbalized_score = key_max

        verbalization_preds.append(verbalized_score)

    return verbalization_preds

def entropy_edge_weights(li_records, idx_existing_edges=None ):
    # Calculate an edge weight based on the entropy over a bernoulli estimation of edge existence
    
    li_entorpy_preds = [None for _ in range(len(li_records))]

    for idx, record in enumerate(li_records):

        if idx not in idx_existing_edges:
            continue
        
        yes_pred = record['pred_aggregated']['Yes']
        no_pred = record['pred_aggregated']['No']

        # normalize the predictions j.i.c
        yes_pred = yes_pred / (yes_pred + no_pred)
        no_pred = no_pred / (yes_pred + no_pred)

        entropy_val = entropy([yes_pred, no_pred], base=2)

        # Transform so more uncertainty is weaker weight

        entropy_val = 1 - ( entropy_val / math.log(2) )

        li_entorpy_preds[idx] = entropy_val
    
    return li_entorpy_preds


def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--exp_idx', type=int, choices=[0,1,2], )
    
    parser.add_argument('--batch_size', type=int, default=1 )

    parser.add_argument('--line_range', type=str, default=None, help='The range of lines to load from the input file' )

    parser.add_argument('--debugging', action='store_true', default=False, help='Indicates whether to run in debugging mode' )

    parser.add_argument('--finetune_dir', type=str, default='/mnt/Data1/akann1warw1ck/AlanTuring/prompt_engineering/finetune/ckpt', help='Directory where finetuned model is stored' )
    
    parser.add_argument('--scale_max', type=int, default=5, help='The maximum value of the scale used to generate the prompt for multinomial edge prediction method')


    args = parser.parse_known_args()[0]

    args.exp_dir = exp_dirs[args.exp_idx]

    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args)) 