"""

"""

import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse 
from agent_based_modelling import ppi
from prompt_engineering.utils import ALL_MODELS
import yaml
warnings.simplefilter("ignore")
import time
import glob
from collections import defaultdict

from agent_based_modelling.i2i_edge_estimation.create_i2i_candidates import concatenate_name_age

from prompt_engineering.my_logger import setup_logging_calibration
logging = None
from builtins import FileNotFoundError

#TODO: ensure the b2i method is handled e.g. figure out what edits to do in order to get the b2i matrix

def main(start_year, end_year, parallel_processes,
            b2i_method, i2i_method, model_size,
            thresholds:list[float], low_precision_counts, increment,
            debugging, time_experiments=False,
            exp_samples=1,
            verbose=False ):

    global logging
    logging =  setup_logging_calibration(debugging=debugging)
    
    # Log parameters for this experiment
    logging.info("Parameters for this experiment:")
    logging.info(f"\tstart_year: {start_year}")
    logging.info(f"\tend_year: {end_year}")
    logging.info(f"\tparallel_processes: {parallel_processes}")
    logging.info(f"\tthresholds: {thresholds}")
    logging.info(f"\tlow_precision_counts: {low_precision_counts}")
    logging.info(f"\tincrement: {increment}")
    logging.info(f"\tb2i_method: {b2i_method}")
    logging.info(f"\ti2i_method: {i2i_method}")
    logging.info(f"\tmodel_size: {model_size}")
    logging.info(f"\tverbose: {verbose}")
    logging.info(f"\ttime_experiments: {time_experiments}")

    # Get calibration kwargs
    calibration_kwargs = get_calibration_kwargs(
                                                b2i_method,
                                                i2i_method,model_size,
                                                start_year, end_year, 
                                                )

    exp_dir = os.path.join('.','agent_based_modelling','output', 'ppi_calibration' )

    # Create seperate experiment directory for each threshold
    for threshold in thresholds:
        
        logging.info(f'Calibrating with threshold {threshold}')
        
        # TODO: if time-experiments is set collect information on #1) experiment time to complete 2) number of steps takn
        li_exp_output:list[dict] = []
        
        for run in range(exp_samples):
            logging.info(f'Calibration run {run+1} of {exp_samples}')

            dict_output = calibrate( low_precision_counts=low_precision_counts,
                    threshold=threshold,
                    parallel_processes=parallel_processes,
                    verbose=verbose,
                    time_experiments=time_experiments,
                  **calibration_kwargs )

            li_exp_output.append(dict_output)
            logging.info(f'Calibration run {run+1} of {exp_samples} complete')
        
        logging.info(f'Calibration with threshold {threshold} complete')

        # Create experiment number which accounts for any missing numbers in the sequence and selects the lowest possible number available
        existing_exp_numbers = [int(exp_number) for exp_number in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, exp_number))]
        existing_exp_numbers.sort()
        exp_number = next( (num for num in range(existing_exp_numbers[-1]+1) if num not in existing_exp_numbers ), existing_exp_numbers[-1]+1)
        

        os.makedirs( os.path.join(exp_dir, exp_number), exist_ok=True)
        
        # Saving parameters for each sample run
        for idx, dict_output in enumerate(li_exp_output):
            df_parameters = pd.DataFrame(dict_output['parameters'])
            df_parameters.to_csv(os.path.join(exp_dir, exp_number, f'calibration_parameters_{ str(idx).zfill(3) }.csv'), index=False)
        
        # Saving statistics on run times
        if time_experiments:
            time_elapsed:list[int] = [dict_output['time_elapsed'] for dict_output in li_exp_output]
            iterations:list[int] = [dict_output['iterations'] for dict_output in li_exp_output]
            
            df_time = pd.DataFrame({'time_elapsed':time_elapsed, 'iterations':iterations, 'exp_number':list(range(exp_samples))})
            df_time.to_csv(os.path.join(exp_dir, exp_number, f'calibration_time.csv'), index=False)

        # Save hyperparameters as yaml
        experiment_hyperparams = {
            'start_year':start_year,
            'end_year':end_year,
            'parallel_processes':parallel_processes,
            'threshold':thresholds,
            'low_precision_counts':low_precision_counts,
            'increment':increment,
            'b2i_method':b2i_method,
            'i2i_method':i2i_method,
            'model_size':model_size,
            'exp_number':exp_number,
            'time_experiments':time_experiments
        }

        with open(os.path.join(exp_dir, exp_number, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(experiment_hyperparams, f)

    return True

def get_calibration_kwargs(
                        b2i_method,
                        i2i_method, model_size,
                        start_year=None,
                           end_year=None,
                            ):

    # TODO: need to recreate pipeline_indicators_normalized file (rl,R,successRates) are based on test period including 2019
    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
    colYears = [col for col in df_indic.columns if str(col).isnumeric()]

    # TODO: need to create a pipeline_expenditure_finegrained which focuses on the fine grained budget items
    df_exp = pd.read_csv('./agent_based_modelling/data/pipeline_expenditure_finegrained.csv')
    expCols = [col for col in df_exp.columns if str(col).isnumeric()]

    if start_year == None:
        start_year = colYears[0]
    
    if end_year == None:
        end_year = colYears[-1]

    num_years = len(colYears)
    T = len(expCols) #Timesteps

    indic_count = len(df_indic) # number of indicators
    indic_final = []
    series = df_indic[colYears].values
    indic_start = series[:,0]
    
    # TODO: Figure out why they do a linear prediction for the final value
    x = np.array([float(year-start_year+1) for year in colYears]).reshape((-1, 1))
    for serie in series:
        y = serie
        
        # If no difference the indicator would not move
        if serie[0] == serie[-1]:
            model = LinearRegression().fit(x, y)
            indic_final.append(model.predict([[ float(end_year-start_year+1) ]])[0])
        else:
            indic_final.append(serie[-1])

    indic_final = np.array(indic_final)
    success_rates = df_indic.successRates.values # success rates


    R = np.ones(indic_count) # instrumental indicators
    qm = df_indic.qm.values # quality of monitoring
    rl = df_indic.rl.values # quality of the rule of law
    # indis_index = dict([(code, i) for i, code in enumerate(df_indic.seriesCode)]) # used to build the network matrix

    Bs = df_exp[expCols].values # disbursement schedule (assumes that the expenditure programmes are properly sorted)

    # TODO: need to create a pipeline_relation_table which focuses on relationships between finegrained budget items and indicators
    # TODO: this also needs to be dependent on the b2i method being used
    

    b2i_network = get_b2i_network( b2i_method, model_size )

    # Load in the i2i relation table
    i2i_network = get_i2i_network( i2i_method=i2i_method, model_size=model_size, indic_count=indic_count )

    return {
        'indic_start': indic_start,
        'indic_final': indic_final,
        'success_rates': success_rates,
        'R': R,
        'qm': qm,
        'rl': rl,
        'Bs': Bs,
        'B_dict': b2i_network,
        'T': T,
        # 'indic_count': indic_count,
        # 'indis_index': indis_index,
        'i2i_network':i2i_network
    }

def get_b2i_network(b2i_method,  model_size) -> dict[int, list[int]]:
    # Create a dictionary which aligns indicators with budget items
    # Both indicators and budget items are referred to by their index in data_expenditure_trend_finegrained and data_expenditure_raw respectively
    # B = { 'idic_idx0':[bi_idx0, bi_idx1, ... ] }

    if b2i_method == 'ea':
        # Load in the b2i relation table
        df_rela = pd.read_csv(os.path.join('data','ppi','pipeline_relation_table_finegrained.csv'))

        B_dict = {} # PPI needs the relational table in the form of a Python dictionary
        for index, row in df_rela.iterrows():
            B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]

    elif b2i_method in ['verbalize', 'CPUQ_binomial']:
        
        # Load the varbalize
        if b2i_method == 'verbalize':
            dir_ = os.path.join('prompt_engineering', 'output', 'spot', f'ppi_b2i_{model_size}_verbalize')
        elif b2i_method == 'CPUQ_binomial':
            dir_ = os.path.join('prompt_engineering', 'output', 'spot', f'ppi_b2i_{model_size}_cpuq')

        _ =  glob.glob(os.path.join(dir_, '**' ,'**predictions_b2i.csv')) # (budget_item,indicator, related, pred_aggregated, prompts, predictions, discourse)
        if len(_) == 0:
            raise FileNotFoundError(f'No b2i predictions found at {dir_}')
        b2i_preds = pd.read_csv(_[0])
        
        # load info on the index ordering of indicators

        # convert to a dictionary where the key is the indicator_name and the value is the indicator's index
        indicator_ref = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_normalized_finegrained.csv'), usecols=['indicator_name'])
        dict_indic_idx = {v: k for k, v in indicator_ref['indicator_name'].to_dict().items()}

        # contains info on the index ordering of budget items 
        budget_item_ref = pd.read_csv(os.path.join('data','ppi','data_expenditure_trend_finegrained.csv'), usecols=['seriesName'])
        dict_bi_idx = {v: k for k, v in budget_item_ref['seriesName'].to_dict().items()}

        B_dict = defaultdict(list) # PPI needs the relational table in the form of a Python dictionary
        for _, row in b2i_preds.iterrows():
            
            bi_idx = dict_bi_idx[ row.budget_item ] # get the index of the indicator
            indic_idx = dict_indic_idx[ row.indicator]
            # value in dict @idx 249 'Proportion of children aged 2-2Ã\x83Â\x82Ã\x82Â½yrs receiving ASQ-3 as part of the Healthy Child Programme or integrated review'
            # value in dict_indic_idx
            B_dict[indic_idx].append(bi_idx)

    return B_dict

def get_i2i_network(i2i_method, indic_count, model_size=None):
    # Creates an array representing indicator to indicator relationships

    assert i2i_method in ['ccdr','CPUQ_multinomial','verbalize','entropy','zero'], f'Spillover Creator Model name {model_name} not recognised'
    
    if i2i_method == 'zero':
        i2i_network = np.zeros((indic_count, indic_count))

    elif i2i_method == 'ccdr':
        _path = os.path.join('data','ppi','i2i_networks','ccdr.csv')
        if os.path.exists(_path):
            df_net = pd.read_csv(_path)
        else:
            raise FileNotFoundError(f'i2i network not created - no file at {_path}')

        i2i_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        for index, row in df_net.iterrows():
            i = int(row.From)
            j = int(row.To)
            w = row.Weight
            i2i_network[i,j] = w

    elif i2i_method in [ 'CPUQ_multinomial', 'verbalize', 'entropy']:

        # Reading in the i2i network
        _1 = {
            '7bn':os.path.join('exp_i2i_7bn_distr','exp_sebeluga7b_non_uc'),
            '13bn':os.path.join('exp_i2i_13_distr','exp_sebeluga13b_non_uc'),
            '30bn':os.path.join('exp_i2i_30b_distr','exp_upllama30b_non_uc'),
        }

        _2 = {'CPUQ':'mn', 'verbalize':'verb', 'entropy':'ent'}
        

        _path = os.path.join('prompt_engineering','output', 'spot', _1[model_size], 
            f'i2i_{_2[i2i_method]}_weights.csv')
        
        if os.path.exists(_path):
            df_net = pd.read_csv(_path) # columns = ['indicator1', 'indicator2', 'weight']
            # filtering out rows where no weight predict or weight was None
            df_net = df_net[ ~df_net.weight.isnull() ]
            df_net.weight =  df_net.weight.apply(lambda x: eval(x))

        else:
            raise FileNotFoundError(f'i2i Network not created - no file at {_path}')
        
        # Establishing the indices for the indicators
        # The order can be found from the pipeline_indicators_normalized_2013_2016.csv file

        # convert to a dictionary where the key is the indicator_name and the value is the indicator's index
        indicator_ref = pd.read_csv(os.path.join('data','ppi','pipeline_indicators_normalized_finegrained.csv'), usecols=['indicator_name'])
        dict_indic_idx = {v: k for k, v in indicator_ref['indicator_name'].to_dict().items()}

        i2i_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        
        for index, row in df_net.iterrows():
            
            
            weight = row.get('scale_mean', row.get('mean', 0.0))
            
            if weight == 0.0:
                continue

            i = dict_indic_idx[row.indicator1] 
            j = dict_indic_idx[row.indicator2] 
            
            i2i_network[i,j] = weight
            
    return i2i_network

def calibrate(indic_start, indic_final, success_rates, R, qm, rl, Bs, B_dict, T, i2i_network,
              parallel_processes=6, threshold=0.8,
              low_precision_counts=75,
              increment=100,
              verbose=True, 
              time_experiments=False):

    dict_output = ppi.calibrate(indic_start, indic_final, success_rates, so_network=i2i_network, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
                T=T, threshold=threshold, parallel_processes=parallel_processes, verbose=True,
                low_precision_counts=low_precision_counts,
                increment=100)

    return dict_output


# Create an argparse function to parse args
def get_args():
    parser = argparse.ArgumentParser(description='Run the PPI model')
    parser.add_argument('--start_year', type=int, default=2013, help='Start year')
    parser.add_argument('--end_year', type=int, default=2018, help='End year')
    parser.add_argument('--parallel_processes', type=int, default=40, help='Number of parallel processes')
    parser.add_argument('--thresholds', type=float, nargs='+', default=0.8, help='Threshold for the calibration')
    parser.add_argument('--low_precision_counts', type=int, default=75, help='Number of low-quality iterations to accelerate the calibration')
    parser.add_argument('--increment', type=int, default=100, help='Number of iterations between each calibration check')

    parser.add_argument('--b2i_method', type=str, default='ccdr', choices=[ 'ea', 'verbalize', 'CPUQ_binomial' ], help='Name of the spillover predictor model')
    parser.add_argument('--i2i_method', type=str, default='ccdr', choices=['ccdr', 'CPUQ_multinomial', 'zero', 'entropy'], help='Name of the indicator to indicator edge predictor method')
    parser.add_argument('--model_size', type=str, default='7bn', choices=['7bn','13bn','30bn' ], help='Name of the indicator to indicator edge predictor method')
    
    parser.add_argument('--verbose', action='store_true', default=False, help='Print progress to console')
    parser.add_argument('--time_experiments', action='store_true', default=False, help='Record Calibration Time for Experiments')
    parser.add_argument('--exp_samples',type=int, default=1, help='Number of samples to take for time experiments')
    parser.add_argument('--debugging', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # pass args in as kwargs
    main(**vars(args))
    