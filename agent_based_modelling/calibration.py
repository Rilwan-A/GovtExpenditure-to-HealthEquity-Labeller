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

def main(start_year, end_year, parallel_processes, 
            spillover_predictor_model_name, thresholds:list[float], low_precision_counts,
            experiment_mode, verbose=False ):

    calibration_kwargs = get_calibration_kwargs(start_year, end_year, spillover_predictor_model_name,
                                                 experiment_mode)


    if experiment_mode == 'calibrate':
        assert len(thresholds) == 1, 'Only one threshold can be used for experiment_mode == calibration'
        threshold = thresholds[0]
        
        df_parameters = calibrate( low_precision_counts=low_precision_counts,
                    threshold=threshold,
                    parallel_processes=parallel_processes,
                    verbose=verbose,
                  **calibration_kwargs )
        save_fn = f'calibration_parameters.csv'      
        save_dir = os.path.join('.','agent_based_modelling','outputs', experiment_mode, spillover_predictor_model_name.replace('/','_') )
        
        # get exp_number as a 3 digit string
        exp_number = len(os.listdir(save_dir))
        exp_number = str(exp_number).zfill(3)
        os.makedirs( os.path.join(save_dir, exp_number), exist_ok=True)
        
        df_parameters.to_csv(os.path.join(save_dir, exp_number, save_fn), index=False)

        # save hyperparams as yaml
        hyperparams = {
            'start_year':start_year,
            'end_year':end_year,
            'parallel_processes':parallel_processes,
            'threshold':threshold,
            'low_precision_counts':low_precision_counts,
            'spillover_predictor_model_name':spillover_predictor_model_name,
        }
        with open(os.path.join(save_dir, exp_number, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(hyperparams, f)
        
        # save calibration_kwargs as yaml
        import pickle
        with open(os.path.join(save_dir, exp_number, 'calibration_kwargs.pkl'), 'wb') as f:
            pickle.dump(calibration_kwargs, f)

    elif experiment_mode == 'time_to_calibrate':
        dict_thresh_convtime = calibration_time_given_fixed_accuracy(start_year, end_year, parallel_processes, thresholds, low_precision_counts, **calibration_kwargs)
        df_convergence_times = pd.DataFrame(columns=['threshold', 'run', 'convergence_time'],
                                            data=[(threshold, run, convergence_time) for threshold, convergence_times in dict_thresh_convtime.items() for run, convergence_time in enumerate(convergence_times)]
                                            )
        # Define save filename and directory as per 'calibrate' mode
        save_fn = 'convergence_times.csv'
        save_dir = os.path.join('.', 'agent_based_modelling', 'outputs', experiment_mode)
        exp_number = len(os.listdir(save_dir))
        exp_number = str(exp_number).zfill(3)
        os.makedirs(os.path.join(save_dir, exp_number), exist_ok=True)
        
        # Save the DataFrame to CSV
        # Save the convergence times dictionary of lists
        df_convergence_times.to_csv(os.path.join(save_dir, exp_number, save_fn), index=False)
        
        # Save hyperparameters as yaml
        hyperparams = {
            'start_year': start_year,
            'end_year': end_year,
            'parallel_processes': parallel_processes,
            'thresholds': thresholds, # Note the change here to 'thresholds'
            'low_precision_counts': low_precision_counts,
            'spillover_predictor_model_name': spillover_predictor_model_name,
        }
        with open(os.path.join(save_dir, exp_number, 'hyperparams.yaml'), 'w') as f:
            yaml.dump(hyperparams, f)
            
    else:
        raise ValueError('Experiment mode not recognised')

    return True

def get_calibration_kwargs(start_year=None,
                           end_year=None,
                            spillover_predictor_model_name='bdag' ):

    df_indic = pd.read_csv('./agent_based_modelling/data/pipeline_indicators_normalized.csv', encoding='unicode_escape') 
    colYears = [col for col in df_indic.columns if str(col).isnumeric()]

    df_exp = pd.read_csv('./agent_based_modelling/data/pipeline_expenditure.csv')
    expCols = [col for col in df_exp.columns if str(col).isnumeric()]

    if start_year == None:
        start_year = colYears[0]
    
    if end_year == None:
        end_year = colYears[-1]

    num_years = len(colYears)
    T = len(expCols)

    N = len(df_indic) # number of indicators
    indic_final = []
    series = df_indic[colYears].values
    indic_start = series[:,0]
    
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


    R = np.ones(N) # instrumental indicators
    qm = df_indic.qm.values # quality of monitoring
    rl = df_indic.rl.values # quality of the rule of law
    # indis_index = dict([(code, i) for i, code in enumerate(df_indic.seriesCode)]) # used to build the network matrix

    Bs = df_exp[expCols].values # disbursement schedule (assumes that the expenditure programmes are properly sorted)

    df_rela = pd.read_csv(os.path.join('agent_based_modelling','data','pipeline_relation_table.csv'))

    B_dict = {} # PPI needs the relational table in the form of a Python dictionary
    for index, row in df_rela.iterrows():
        B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]

    so_network = get_spillover_network( model_name=spillover_predictor_model_name , indic_count=N )

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
        # 'N': N,
        # 'indis_index': indis_index,
        'so_network':so_network
    }

def get_spillover_network(model_name, indic_count):

    assert model_name in ALL_MODELS+['bdag'], f'Spillover Creator Model name {model_name} not recognised'

    if model_name == 'bdag':
        _path = os.path.join('data','spillover_networks','bdag.csv')
        if os.path.exists(_path):
            df_net = pd.read_csv(_path)
        else:
            raise FileNotFoundError(f'Spillover network not created - no file at {_path}')

        so_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        for index, row in df_net.iterrows():
            i = int(row.From)
            j = int(row.To)
            w = row.Weight
            so_network[i,j] = w

    elif model_name in ALL_MODELS:
        _path = os.path.join('data','spillover_networks',f'{model_name}.csv')
        if os.path.exists(_path):
            df_net = pd.read_csv(_path)
        else:
            raise FileNotFoundError(f'Spillover Network not created - no file at {_path}')
        
        so_network = np.zeros((indic_count, indic_count)) # adjacency matrix
        for index, row in df_net.iterrows():
            i = int(row.indicator1)
            j = int(row.indicator2)
            
            prob_yes = eval(row.pred_aggregated).get('Yes')
            prob_yes = prob_yes if prob_yes > 0.5 else 0
            w = prob_yes

            so_network[i,j] = w
    
    return so_network

def calibrate(indic_start, indic_final, success_rates, R, qm, rl, Bs, B_dict, T, so_network,
              parallel_processes=None, threshold=None, low_precision_counts=None, verbose=True):

    if parallel_processes == None:
        parallel_processes =  40 # number of cores to use
    if threshold == None:
        threshold =  0.8 # the quality of the calibration (I choose a medium quality for illustration purposes)
    if low_precision_counts == None:
        low_precision_counts =  75  # number of low-quality iterations to accelerate the calibration

    parameters = ppi.calibrate(indic_start, indic_final, success_rates, so_network=so_network, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
                T=T, threshold=threshold, parallel_processes=parallel_processes, verbose=True,
                low_precision_counts=low_precision_counts)

    dff = pd.DataFrame(parameters[1::,:], columns=parameters[0])

    return dff

def calibration_time_given_fixed_accuracy( parallel_processes, thresholds, low_precision_counts, **calibration_kwargs) -> dict(float, list(float)):
    """
    Measure the convergence time for a set of ppi.calibration calls at different thresholds.
    
    Args:
    ... (same as main function)
    
    Returns:
    pd.DataFrame: A DataFrame containing the threshold levels and corresponding convergence times.
    """
    sample_count = 100

    # Create a list to store results (thresholds and their corresponding convergence time)
    from collections import defaultdict

    dict_thresh_convtime = defaultdict(list)
    
    # Iterate over different thresholds
    for threshold in thresholds:
        # Repeat the calibration process a number of times to get an convergence time statistics
        for _ in range(sample_count):
            # Start the timer
            start_time = time.time()
            
            _ = ppi.calibrate(
                parallel_processes=parallel_processes,
                threshold=threshold,
                low_precision_counts=low_precision_counts,
                **calibration_kwargs,
                verbose = False
                )
            
            # End the timer
            end_time = time.time()
            
            # Calculate the elapsed time (convergence time) for this threshold
            convergence_time = end_time - start_time
            
            # Append the threshold and corresponding convergence time to the results list
            dict_thresh_convtime[threshold].append((threshold, convergence_time))
            
    
    return dict_thresh_convtime

# Create an argparse function to parse args
def get_args():
    parser = argparse.ArgumentParser(description='Run the PPI model')
    parser.add_argument('--start_year', type=int, default=2013, help='Start year')
    parser.add_argument('--end_year', type=int, default=2018, help='End year')
    parser.add_argument('--parallel_processes', type=int, default=40, help='Number of parallel processes')
    parser.add_argument('--thresholds', type=float, nargs='+', default=0.8, help='Threshold for the calibration')
    parser.add_argument('--spillover_predictor_model_name', type=str, default='bdag', choices=['bdag']+ALL_MODELS , help='Name of the spillover predictor model')
    parser.add_argument('--low_precision_counts', type=int, default=75, help='Number of low-quality iterations to accelerate the calibration')
    parser.add_argument('--experiment_mode', type=str, choices=[ 'calibrate', 'calibration_time_given_fixed_accuracy', 'accuracy_given_fixed_time' ])
    parser.add_argument('--verbose', type=bool, default=False, help='Print progress to console')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    # pass args in as kwargs
    main(**vars(args))
    