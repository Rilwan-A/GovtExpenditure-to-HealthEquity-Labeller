"""

"""

import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse 
import ppi
from prompt_engineering.utils import ALL_MODELS
import yaml
warnings.simplefilter("ignore")
import time

from i2i_edge_estimation.create_i2i_candidates import concatenate_name_age
from prompt_engineering.my_logger import setup_logging_calibration
logging = None

#TODO: ensure the b2i method is handled e.g. figure out what edits to do in order to get the b2i matrix

def main(start_year, end_year, parallel_processes, experiment_mode,
            b2i_method, i2i_method, i2i_model_size,
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
    logging.info(f"\ti2i_model_size: {i2i_model_size}")
    logging.info(f"\texperiment_mode: {experiment_mode}")   
    logging.info(f"\tverbose: {verbose}")
    logging.info(f"\ttime_experiments: {time_experiments}")

    # Get calibration kwargs
    calibration_kwargs = get_calibration_kwargs(
                                                b2i_method,
                                                i2i_method,i2i_model_size,
                                                start_year, end_year, 
                                                experiment_mode )

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
            'i2i_model_size':i2i_model_size,
            'experiment_mode':experiment_mode,
            'exp_number':exp_number,
            'time_experiments':time_experiments
        }

        with open(os.path.join(exp_dir, exp_number, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(experiment_hyperparams, f)

    return True

def get_calibration_kwargs(
                        b2i_method,
                        i2i_method, i2i_model_size,
                        start_year=None,
                           end_year=None,
                            ):

    # TODO: need to recreate pipeline_indicators_normalized file (rl,R,successRates) are based on test period including 2019
    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_2013_2016.csv', encoding='unicode_escape') 
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
    df_rela = pd.read_csv(os.path.join('agent_based_modelling','data','pipeline_relation_table.csv'))

    B_dict = {} # PPI needs the relational table in the form of a Python dictionary
    for index, row in df_rela.iterrows():
        B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]

    i2i_network = get_i2i_network( i2i_method=i2i_method, i2i_model_size=i2i_model_size, indic_count=indic_count )

    return {
        'indic_start': indic_start,
        'indic_final': indic_final,
        'success_rates': success_rates,
        'R': R,
        'qm': qm,
        'rl': rl,
        'Bs': Bs,
        'B_dict': B_dict,
        'T': T,
        # 'indic_count': indic_count,
        # 'indis_index': indis_index,
        'i2i_network':i2i_network
    }

def get_i2i_network(i2i_method, indic_count, i2i_model_size=None):

    assert i2i_method in ['bdag','CPUQ','verbalize','entropy','zero'], f'Spillover Creator Model name {model_name} not recognised'
    
    if i2i_method == 'zero':
        i2i_network = np.zeros((indic_count, indic_count))

    elif i2i_method == 'bdag':
        _path = os.path.join('data','ppi','i2i_networks','bdag.csv')
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

    elif i2i_method in [ 'CPUQ', 'verbalize', 'entropy']:

        # Reading in the i2i network
        _1 = {
            '7bn':os.path.join('exp_i2i_7bn_distr','exp_sebeluga7b_non_uc'),
            '13bn':os.path.join('exp_i2i_13_distr','exp_sebeluga13b_non_uc'),
            '30bn':os.path.join('exp_i2i_30b_distr','exp_upllama30b_non_uc'),
        }

        _2 = {'CPUQ':'mn', 'verbalize':'verb', 'entropy':'ent'}
        

        _path = os.path.join('prompt_engineering','output', 'spot', _1[i2i_model_size], 
            f'i2i_{_2[i2i_method]}_weights.csv')
        
        if os.path.exists(_path):
            df_net = pd.read_csv(_path) # columns = ['indicator1', 'indicator2', 'weight']
        else:
            raise FileNotFoundError(f'i2i Network not created - no file at {_path}')
        
        # Establishing the indices for the indicators and budget items
        # The order can be found from the pipeline_indicators_normalized_2013_2016.csv file
        try:
            df_index_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_2013_2016.csv', usecols=['idx','indicator_name_ftmd'] )
            # catch exception of cols in use_cols not present
        except ValueError:
            # run a python file
            logging.info("Pipeline indicators file `/pipeline_indicators_normalized_2013_2016` does not have columns 'idx' and 'indicator_name_fmtd'")
            logging.info("Running create_indicators_index.py to add the columns to the file")
            import subprocess
            subprocess.run(['python3', './agent_based_modelling/i2i_edge_estimation/create_indicators_index.py'])
            df_index_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_2013_2016.csv', usecols=['idx','indicator_name_ftmd'] )

        # dict connecting the index to the indicator name (formatted)
        dict_indicators_idx = df_index_indic.set_index('indicator_name_fmtd')['idx'].to_dict()

        i2i_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        for index, row in df_net.iterrows():
            
            # Checking if edge exists
            edge_exists = row.weight is not 'None'
            if edge_exists is False:
                continue
            
            i = dict_indicators_idx[row.indicator1] 
            j = dict_indicators_idx[row.indicator2] 
            
            weight_dict = eval(row.weight)

            if 'mean_scaled' in weight_dict.keys():
                weight = weight_dict['mean_scaled']
            elif 'mean' in weight_dict.keys():
                weight = weight_dict['mean']
            else:
                raise ValueError('No weight found in weight_dict')
            
            i2i_network[i,j] = weight
            
    return i2i_network

def calibrate(indic_start, indic_final, success_rates, R, qm, rl, Bs, B_dict, T, i2i_network,
              parallel_processes=6, threshold=0.8,
              low_precision_counts=75,
              increment=100,
              verbose=True, 
              time_experiments=False):

    # if parallel_processes == None:
    #     parallel_processes =  6 # number of cores to use
    # if threshold == None:
    #     threshold =  0.8 # the quality of the calibration (I choose a medium quality for illustration purposes)
    # if low_precision_counts == None:
    #     low_precision_counts =  75  # number of low-quality iterations to accelerate the calibration


    
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

    parser.add_argument('--b2i_method', type=str, default='bdag', choices=['naive','verbalize','zero'], help='Name of the spillover predictor model')
    parser.add_argument('--i2i_method', type=str, default='bdag', choices=['bdag','verbalize','CPUQ', 'zero', 'entropy'], help='Name of the indicator to indicator edge predictor method')
    parser.add_argument('--i2i_model_size', type=str, default='7bn', choices=['7bn','13bn','30bn' ], help='Name of the indicator to indicator edge predictor method')
    
    parser.add_argument('--experiment_mode', type=str, choices=[ 'calibrate' ])
    parser.add_argument('--verbose', action='store_true', default=False, help='Print progress to console')
    parser.add_argument('--time_experiments', action='store_true', default=False, help='Record Calibration Time for Experiments')
    parser.add_argument('--exp_samples',type=int, default=1, help='Number of samples to take for time experiments')
    parser.add_argument('--debugging', action='store_true',, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # pass args in as kwargs
    main(**vars(args))
    