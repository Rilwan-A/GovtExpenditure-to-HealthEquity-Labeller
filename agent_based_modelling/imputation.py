"""
    In this script we are focused on imputing the level of the indicators over the next n-1 timesteps after learning (calibrating) a model on the first m timesteps.
    We use the indicator level at step m+n to define the development goal. We input resource allocation and the true rule of law over between step m and step m+n-1
    We then can impute the indicator values between step m and step m+n-1
    We can then compare the imputed indicator values to the true indicator values between step m and step m+n-1,
    We can then evaluate how well our model has learnt the dynamics between budget items, indicators and agents.
    

    The key steps would be:
    0) Load in the parameters saved from the previous run
    1) Initialize the model with the current indicator levels.
    2) Specify the resource allocation (P) for each indicator that you expect in the next n time steps. 
    3) Specify the projected rule of law (fR) for the next n time steps.
    4) Run the model for n time steps, using the specified P and fR as inputs. 
    5) The model will then forecast the indicator levels (I) for the next n time steps, based on:

"""
import argparse
import pandas as pd
import os
import numpy as np
import yaml
from ppi import run_ppi
import pickle
from prompt_engineering.utils import ALL_MODELS

def main( impute_periods:int, spillover_predictor_model_name, exp_num:int=0):

    # Load parameters from trained ppi model
    # Load calibration_kwargs e.g. the params for the PPI model
    model_params = load_model_kwargs( spillover_predictor_model_name, exp_num)

    current_I, fBs, frl, fG = load_currI_fBs_frl_fG( spillover_predictor_model_name , impute_periods=impute_periods )

    imputed_indicators = impute_indicators( impute_periods,
                                                current_I, fBs, frl, fG,
                                                model_params = model_params )
    
    true_indicators, years, indicator_names = load_true_indicators( spillover_predictor_model_name, impute_periods=impute_periods )

    #  Save the imputed and true indicators to file
    outp = {
        'imputed_indicators': imputed_indicators,
        'true_indicators': true_indicators,
        'years': years,
        'indicator_names': indicator_names,

    }

    save_dir = os.path.join('.','agent_based_modelling','outputs', 'calibrate', spillover_predictor_model_name.replace('/','_') )
    exp_num = str(exp_num).zfill(3)
    fn = f'imputed_indicators_{exp_num}.pkl'
    with open(os.path.join(save_dir, fn), 'wb') as f:
        pickle.dump(outp, f)


    # Save the imputed and true indicators to file

def load_model_kwargs( spillover_predictor_model_name, exp_num:int|None=None ) -> dict:
    
    save_fn = f'calibration_kwargs.yaml'        
    save_dir = os.path.join('.','agent_based_modelling','outputs', 'calibrate', spillover_predictor_model_name.replace('/','_') )

    # Convert
    import pickle
    with open(os.path.join(save_dir, str(exp_num).zfill(3), save_fn), 'rb') as f:
        model_params = pickle.load(f)

    return model_params

def load_currI_fBs_frl_fG(spillover_predictor_model_name, impute_periods):
    """
    Load the current indicator levels, forecasted resource allocation, and forecasted rule of law for the next n time steps.
    """

    save_dir = os.path.join('.','agent_based_modelling','outputs', 'calibrate', spillover_predictor_model_name.replace('/','_') )

    hyper_params = yaml.safe_load( open( os.path.join(save_dir, 'hyperparams.yaml'), 'r' ) )

    calibration_start_year = hyper_params['start_year']
    calibration_end_year = hyper_params['end_year']

    # Load the data
    df_indic = pd.read_csv('./agent_based_modelling/data/pipeline_indicators_normalized.csv', encoding='unicode_escape') 
    df_exp = pd.read_csv('./agent_based_modelling/data/pipeline_expenditure.csv')
    colYears = [col for col in df_indic.columns if str(col).isnumeric()]

    # Checking that the forecast period is within the bounds of the training data
    final_period = impute_periods + 1
    assert final_period <= len(colYears) - colYears.index(calibration_end_year) - 1, \
        f'Forecast period is too long. Max forecast period is {len(colYears) - colYears.index(calibration_end_year) - 1}'

    # current_I
    current_I = df_indic[colYears.index(calibration_end_year) ].iloc[-1].values

    # fBs - control budget allocation
    fBs = df_exp[colYears[colYears.index(calibration_end_year) + 1:colYears.index(calibration_end_year) + 1 + final_period]].iloc[-1].values #(assumes that the expenditure programmes are properly sorted)
        
    # fR
    frl = df_indic.rl.values # quality of the rule of law

    # fG
    fG = df_indic[colYears.index(calibration_end_year+final_period) ].iloc[-1].values

    return current_I, fBs, frl, fG
    
def impute_indicators(impute_periods, current_I, fBs, frl, fG , model_params ):
    """
    Forecast the indicator levels for the next n time steps using the PPI model.
    
    Parameters:
    - filepath: The path to the saved parameter file
    - current_I: The current indicator levels
    - P: Resource allocation for each indicator for the next n time steps
    - fR: Projected rule of law for the next n time steps
    
    Returns:
    - Forecasted indicator levels for the next n time steps
    """
    forecast_periods = impute_periods + 1
    assert len(fBs) == forecast_periods, f'P must be a list of length {forecast_periods}'
    assert len(frl) == forecast_periods, f'fR must be a list of length {forecast_periods}'
    
    # Extract the controlled parameters
    I0 = current_I # Initial indicator levels
    T = forecast_periods # Forecast period
    G = fG # Target indicator level
    rl = np.array([frl]*T) # Assuming constant Projected rule of law over imputation period
    Bs = fBs # Forecasted Budget Allocation

    # Extract the necessary parameters
    alphas = model_params['alphas']
    alphas_prime = model_params['alphas_prime']
    betas = model_params['betas']
    so_network = model_params['so_network']
    Imax = model_params['Imax']
    Imin = model_params['Imin']
    R = model_params['R']
    qm = model_params['qm']
    B_dict = model_params['B_dict']
    bs = model_params['bs'] 
           
    
    # Step 2 and 4 are combined: Run the model for frl steps, 
    # using the specified P and fR as inputs
    tsI, _, _, _, _, _ = run_ppi(I0, alphas, alphas_prime, betas, so_network, 
                                 R, bs, qm, rl, Imax, Imin, Bs, B_dict, G, T)
    

    forecasted_I = tsI
    
    return forecasted_I

def load_true_indicators(spillover_predictor_model_name, impute_periods):
    """
    Load the true indicator levels for the next forecast_periods time steps.
    """
    forecast_periods = impute_periods + 1

    save_dir = os.path.join('.','agent_based_modelling','outputs', 'calibrate', spillover_predictor_model_name.replace('/','_') )

    hyper_params = yaml.safe_load( open( os.path.join(save_dir, 'hyperparams.yaml'), 'r' ) )

    # calibration_start_year = hyper_params['start_year']
    calibration_end_year = hyper_params['end_year']

    # Load the data
    df_indic = pd.read_csv('./agent_based_modelling/data/pipeline_indicators_normalized.csv', encoding='unicode_escape') 
    colYears = [col for col in df_indic.columns if str(col).isnumeric()]

    # Checking that the forecast period is within the bounds of the training data
    assert forecast_periods <= len(colYears) - colYears.index(calibration_end_year) - 1, \
        f'Forecast period is too long. Max forecast period is {len(colYears) - colYears.index(calibration_end_year) - 1}'
    
    # True indicator levels
    true_indicators = df_indic[colYears.index(calibration_end_year) + 1:colYears.index(calibration_end_year) + 1 + forecast_periods].values

    years = np.arange(calibration_end_year, calibration_end_year + forecast_periods+1, 1.0)

    indicator_names = df_indic.seriesName.values

    return true_indicators, years, indicator_names

def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Impute missing values in a time series.')
    parser.add_argument('--spillover_predictor_model_name', type=str, default='bdag', choices=['bdag']+ALL_MODELS , help='Name of the spillover predictor model')
    parser.add_argument('--impute_periods', type=int, default=1, help='Number of periods to impute. This assumes you have the final indicator level after the periods to be imputed')
    parser.add_argument('--exp_num', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()   
    
    main(**vars(args))
    