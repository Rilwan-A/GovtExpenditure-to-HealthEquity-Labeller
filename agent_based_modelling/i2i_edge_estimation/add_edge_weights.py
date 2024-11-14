# In this script we add edge weight estimations to the i2i predictions made by the llm models.
# We use 3 different methodologies: multionomial_mean, entropy over binomial edge existance (rescaled to 0-1), verbalization of the binomial edge existance (rescaled to 0-1)import pandas as pd
import os
import yaml
import time
import pandas as pd

from argparse import ArgumentParser
from scipy.stats import entropy
import gc
from prompt_engineering.my_logger import setup_logging_add_i2i_edge_weights
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from prompt_engineering.utils import PredictionGenerator, load_llm
from prompt_engineering.utils_prompteng import PromptBuilder
from torch.cuda import empty_cache
from torch import no_grad
# exp_dirs = [
#     os.path.join('prompt_engineering','output','spot','exp_i2i_7bn_distr', 'exp_sbeluga7b_non_uc'),
#     os.path.join('prompt_engineering','output','spot','exp_i2i_13b_distr', 'exp_sbeluga13b_non_uc'),
#     os.path.join('prompt_engineering','output','spot','exp_i2i_30b_distr','exp_upllama30b_non_uc'),

    # # TODO: update the following paths when model to be used is decided
    # os.path.join('prompt_engineering','output','spot','exp_i2i_30bn_distr', '30bn_HMRC'),

# ]


def main( experiment_dir, debugging=False, batch_size=1, finetune_dir='', scale_max=5, gpu_batch_size=1 ):
    if gpu_batch_size > batch_size:
        batch_size = gpu_batch_size

    # Set up logging
    logger = setup_logging_add_i2i_edge_weights(debugging)


    df_preds = pd.read_csv(os.path.join( experiment_dir, 'predictions_i2i.csv'))
    config = yaml.safe_load( open( os.path.join( experiment_dir, 'config.yaml'), "r" ) )
    config['finetune_dir'] = finetune_dir

    # Log the config file used
    logger.info("Arguments:")
    for key, value in config.items():
        logger.info(f'\t{key}: {value}')

    
    if debugging:
        # sample 10 items
        df_preds = df_preds.sample( max(20, gpu_batch_size)  )

    indicator1 = df_preds['indicator1'].tolist()
    indicator2 = df_preds['indicator2'].tolist()
    preds = df_preds['pred_aggregated'].apply(eval).tolist()

    li_records = [ {'indicator1':ind1, 'indicator2':ind2, 'pred_aggregated':pred} for ind1, ind2, pred in zip(indicator1, indicator2, preds) ]      

    # get index of positions where the binomial prediction for 'Yes' is higher than 0.5
    if debugging:
        idx_existing_edges = list( range( 0, len(df_preds) ) )
        pass
        # llm_name = 'mlabonne/dummy-llama-2'
    else:
        idx_existing_edges = [i for i, x in enumerate(preds) if x['Yes'] >= 0.5]
    
    original_len = len(li_records)
    li_records_filtered = [li_records[i] for i in idx_existing_edges]

    # Load the LLM used to make the predictions
    llm_name = config['llm_name']
    model, tokenizer = load_llm(llm_name, 
                        finetuned=config['finetuned'],
                        local_or_remote='local',
                        finetune_dir=config['finetune_dir'],
                        exp_name = config['exp_name'],
                        finetune_version = config.get('finetune_version',None) )

    # verbalization on a scale 0 to scale_max
    li_pred_agg_verbalized, li_pred_ensembles_verbalized = verbalize_edge_weights(model, llm_name, tokenizer, li_records, config, scale_max, logger, batch_size, gpu_batch_size=gpu_batch_size)

    # For each indicator pair we produce an ensemble of predictions
    # Retrieve a list of the prediction ensembles list[ list[ dict[str,float] ] ]
    # Retreive the mean of the sets of prediction ensembles list[ dict['mean',float] ]
    multinomial_means, multinomial_distributions = multinomial_edge_weights(model, llm_name, li_records_filtered, config, scale_max, logger, batch_size)

    # We use the entropy over the binomial distribution as a measure of relationship strength 
    entropy_preds = entropy_edge_weights( li_records_filtered )

    # Adjusting for the fact that we only made predictions for a subset of the indicator pairs
    if idx_existing_edges is not None:
        
        _iter = iter(multinomial_means)
        multinomial_means = [ next(_iter) if i in idx_existing_edges else None for i in range(original_len) ]

        _iter = iter(multinomial_distributions)
        multinomial_distributions = [ next(_iter) if i in idx_existing_edges else None for i in range(original_len) ]

        _iter = iter(entropy_preds)
        entropy_preds = [ next(_iter) if i in idx_existing_edges else None for i in range(original_len) ]
        
    # Saving these results to file
    path_mn = os.path.join(experiment_dir, 'i2i_mn_weights.csv')
    path_vb = os.path.join(experiment_dir, 'i2i_vb_weights.csv')
    path_et = os.path.join(experiment_dir, 'i2i_et_weights.csv')

    df_mn = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':multinomial_means, 'distribution':multinomial_distributions})
    df_vb = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':li_pred_agg_verbalized, 'preds':li_pred_ensembles_verbalized })
    df_et = pd.DataFrame({'indicator1':indicator1, 'indicator2':indicator2, 'weight':entropy_preds})

    logger.info(f'Saving multinomial edge weights to {path_mn}')
    logger.info(f'Saving verbalization edge weights to {path_vb}')
    logger.info(f'Saving entropy edge weights to {path_et}')
    df_mn.to_csv(path_mn, index=False)
    df_vb.to_csv(path_vb, index=False)
    df_et.to_csv(path_et, index=False)

def multinomial_edge_weights(model, llm_name, li_records, config, scale_max, logger=None, batch_size=2):

    # Get the prompt template for querying the strength of the relationship between the two indicators
    prompt_builder = PromptBuilder(
        model,
        llm_name,
        prompt_style='categories_scale',
        k_shot = 0,
        ensemble_size = 1,
        effect_type = 'arbitrary',
        relationship = 'indicator_to_indicator'
    )

    prediction_generator = PredictionGenerator(
        model,
        llm_name,
        prompt_style = 'categories_scale',
        ensemble_size=1,
        edge_value = 'multinomial_distribution',
        parse_style = 'categories_perplexity',
        relationship = 'indicator_to_indicator',
        local_or_remote = 'local')
        
    li_pred_agg = []
    li_pred_ensembles = []

    # Running Prompt Generations and Predictions
    li_li_record = [ li_records[i:i+batch_size] for i in range(0, len(li_records), batch_size) ]

    last_time = time.time()
    logger.info('Generating Multinomial Edges Predictions')
    for idx, batch in enumerate(li_li_record):
        if logger is not None:
            
            if time.time() - last_time  > 60:
                logger.info(f'\tProcessing batch {idx} out of {len(li_li_record)}')
                last_time = time.time()

        with no_grad():
            #  Create prompts
            batch_li_li_statement, batch_li_li_discourse = prompt_builder(batch, scale_max=scale_max)

            # Generate predictions
            batch_pred_ensembles = prediction_generator.predict(batch_li_li_statement, scale_max=scale_max)

            # Aggregate ensembles into predictions - calculate the mean of the multinomial distribution
            batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles, scale_max=scale_max)

        li_pred_ensembles.extend(batch_pred_ensembles ) # type: ignore
        li_pred_agg.extend(batch_pred_agg ) # type: ignore
    
    # Free up memory and GPU memory
    empty_cache()
    gc.collect()
    logger.info('Finished Generating Multinomial Edges Predictions')

    return li_pred_agg, li_pred_ensembles

def verbalize_edge_weights(model, llm_name, tokenizer, li_records, config, scale_max, logger=None, batch_size=2, gpu_batch_size=None):


    # Get the prompt template for querying the strength of the relationship between the two indicators
    if isinstance(model, HuggingFacePipeline):
        tokenizer = tokenizer
        model = model.pipeline.model
        
    else:
        raise ValueError(f'Unexpected model type: {type(model)}')
        
    prompt_builder = PromptBuilder(
        # model,
        model,
        llm_name,
        tokenizer = tokenizer,
        prompt_style='verbalize_scale',
        k_shot = 0,
        ensemble_size = 1,
        effect_type = 'arbitrary',
        relationship = 'indicator_to_indicator'
    )

    prediction_generator = PredictionGenerator(
        model,
        llm_name,
        prompt_style = 'verbalize_scale',
        ensemble_size=1,
        edge_value = 'scale',
        parse_style = 'rules',
        relationship = 'indicator_to_indicator',
        local_or_remote = 'local')
        
    li_pred_agg = []
    li_pred_ensembles = []

    # Running Prompt Generations and Predictions
    li_li_record = [ li_records[i:i+batch_size] for i in range(0, len(li_records), batch_size) ]

    logger.info('Generating Verbalization Edges')
    last_time = time.time()
    for idx, batch in enumerate(li_li_record):
        if logger is not None:
            
            if time.time() - last_time  > 60:
                logger.info(f'\tProcessing batch {idx} out of {len(li_li_record)}')
                last_time = time.time()

        #  Create prompts
        batch_li_li_statement, batch_li_li_discourse = prompt_builder(batch, scale_max=scale_max, gpu_batch_size=gpu_batch_size)

        # Generate predictions
        batch_pred_ensembles = prediction_generator.predict(batch_li_li_statement, scale_max=scale_max)

        # Aggregate ensembles into predictions - calculate the mean of the multinomial distribution
        batch_pred_agg = prediction_generator.aggregate_predictions(batch_pred_ensembles, scale_max=scale_max)

        li_pred_ensembles.extend(batch_pred_ensembles ) # type: ignore
        li_pred_agg.extend(batch_pred_agg ) # type: ignore
    
    # Free up memory and GPU memory
    del model
    del prediction_generator
    del prompt_builder
    empty_cache()

    return li_pred_agg, li_pred_ensembles

def entropy_edge_weights(li_records):
    # Calculate an edge weight based on the entropy over a bernoulli estimation of edge existence
    
    li_entorpy_preds = [None for _ in range(len(li_records))]

    
    for idx, record in enumerate(li_records):

        # if idx not in idx_existing_edges:
        #     continue
        yes_pred = record['pred_aggregated']['Yes']
        no_pred = record['pred_aggregated']['No']

        # normalize the predictions j.i.c
        yes_pred = yes_pred / (yes_pred + no_pred)
        no_pred = no_pred / (yes_pred + no_pred)

        entropy_val = entropy([yes_pred, no_pred], base=2)

        # Transform so more uncertainty is weaker weight

        entropy_val = 1 - entropy_val

        li_entorpy_preds[idx]= {'mean':entropy_val }
    
    return li_entorpy_preds

def parse_args():
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    # parser.add_argument('--exp_idx', type=int, choices=[0,1,2] )
    parser.add_argument('--experiment_dir', type=str, default=os.path.join('prompt_engineering','output','spot','exp_i2i_30b_distr','exp_upllama30b_non_uc') )
    parser.add_argument('--batch_size', type=int, default=1 )
    parser.add_argument('--gpu_batch_size', type=int, default=1 )
    parser.add_argument('--debugging', action='store_true', default=False, help='Indicates whether to run in debugging mode' )
    parser.add_argument('--finetune_dir', type=str, default='', help='Directory where finetuned model is stored' )
    parser.add_argument('--scale_max', type=int, default=5, help='The maximum value of the scale used to generate the prompt for multinomial edge prediction method')


    args = parser.parse_known_args()[0]

    # args.experiment_dir = exp_dirs[args.exp_idx]
    # del args.exp_idx
    return args

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args)) 