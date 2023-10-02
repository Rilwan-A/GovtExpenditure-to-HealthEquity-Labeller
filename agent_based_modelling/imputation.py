import argparse
import pandas as pd
import os
import numpy as np
import yaml
from agent_based_modelling.ppi import run_ppi, run_ppi_parallel
import glob
import pickle
from prompt_engineering.utils import ALL_MODELS

from agent_based_modelling.calibration import get_b2i_network, get_i2i_network


def main( impute_start_year:int=2018, impute_years:int=1, exp_num:int=0):

    # Load parameters from trained ppi model
    # Load calibration_kwargs e.g. the params for the PPI model
    model_params = load_model_kwargs( exp_num)
    model_hparams = load_model_hparams( exp_num )

    current_I, fBs, frl, fG, time_refinement_factor = load_currI_fBs_frl_fG( impute_start_year=impute_start_year, impute_years=impute_years )

    b2i_network = get_b2i_network( model_hparams['b2i_method'], model_hparams['model_size'] )
    i2i_network = get_i2i_network( model_hparams['i2i_method'], current_I.shape[0], model_hparams['model_size'] )

    imputed_indicators = impute_indicators( impute_years, time_refinement_factor,
                                                current_I, fBs, frl, fG, i2i_network, b2i_network,
                                                model_params = model_params )
    
    indicator_values, indicator_names = load_true_indicators( impute_years=impute_years, impute_start_year=impute_start_year )

    #  Save the imputed and true indicators to file
    outp = {
        'imputed_indicators': imputed_indicators,
        'target_indicators': indicator_values,
        'impute_start_year': impute_start_year,
        'impute_years': impute_years,
        'indicator_names': indicator_names,
    }

    save_dir = os.path.join('.','agent_based_modelling','outputs', 'imputations' )
    
    
    fn = f'exp_{str(exp_num).zfill(2)}.pkl'
    with open(os.path.join(save_dir, fn), 'wb') as f:
        pickle.dump(outp, f)

def load_model_kwargs( exp_num:int|None=None ) -> pd.DataFrame:
        
    f_pattern = os.path.join('.','agent_based_modelling','outputs', 'calibrated_parameters', f'exp_{str(exp_num).zfill(2)}','params_v**.csv') 

    # get list of versions of parameters associated with the experiment number
    param_files = glob.glob( f_pattern )

    # get the latest version - this should the file with the highest goodness of fit
    fp = sorted(param_files)[-1]

    df_parameters = pd.read_csv(fp)

    return df_parameters

def load_model_hparams( exp_num:int|None=None ) -> pd.DataFrame:
        
    f_pattern = os.path.join('.','agent_based_modelling','outputs', 'calibrated_parameters', f'exp_{str(exp_num).zfill(2)}','hyperparams.yaml') 

    # get list of versions of parameters associated with the experiment number
    param_files = glob.glob( f_pattern )

    # get the latest version - this should the file with the highest goodness of fit
    fp = sorted(param_files)[-1]

    df_parameters = yaml.safe_load(fp)

    return df_parameters

def load_currI_fBs_frl_fG(impute_start_year=2018, impute_years=1, exp_num=0):
    """
    Load the current indicator levels, forecasted resource allocation, and forecasted rule of law for the next n time steps.
    """

    # exp_dir = os.path.join('.','agent_based_modelling','output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}' )
    # # get list of versions of parameters associated with the experiment number
    # # get the latest version - this should the file with the highest goodness of fit
    # f_pattern_params = os.path.join(exp_dir, 'params_v**.csv')
    # param_files = glob.glob( f_pattern_params )
    # fp = sorted(param_files)[-1]
    # calibration_params = pd.read_csv(fp)

    calibration_hparams = os.path.join( '.', 'agent_based_modelling', 'output', 'calibrated_parameters', f'exp_{str(exp_num).zfill(3)}', 'hyperparams.yaml')
    calibration_hparams = yaml.safe_load( open( calibration_hparams, 'r' ) )

    calibration_start_year = calibration_hparams['calibration_start_year']
    calibration_end_year = calibration_hparams['calibration_end_year']
    
    # The start and final year are used as inputs to the PPI model
    impute_final_year = impute_start_year + impute_years -1

    # Load the data
    df_indic = pd.read_csv('./agent_based_modelling/data/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
    df_exp = pd.read_csv('./agent_based_modelling/data/pipeline_expenditure_finegrained.csv')
    
    years = [col for col in df_indic.columns if str(col).isnumeric() ] #if col>=calibration_start_year and col<=impute_final_year ]
    years_int = [int(col) for col in years]
    tft = time_refinement_factor = df_exp['time_refinement_factor'].values[0]

    # Checking that the forecast period is within the bounds of the training data
    
    assert ( str(int(impute_start_year)-1) in years) and (str(impute_final_year) in years), \
        f'Impute period is not within the bounds of the available data'

    # current_I
    # we take imputation levels from the end of previous year
    current_I = df_indic[ str(impute_start_year-1) ].values

    # fBs - control budget allocation
    impute_start_year_idx = ( years_int.index(impute_start_year)-years_int.index(calibration_start_year) )
    impute_final_year_idx = ( years_int.index(impute_final_year)-years_int.index(calibration_start_year) )
    # impute_periods = impute_years*time_refinement_factor
    # fBs = df_exp[years[years.index(impute_start_year):years.index(impute_final_year) ]].iloc[-1].values #(assumes that the expenditure programmes are properly sorted)
    fBs_cols = [ str(idx) for idx in  range(impute_start_year_idx*tft, (impute_final_year_idx+1)*tft) ]
    fBs = df_exp[ fBs_cols ].values
    # fR
    frl = df_indic.rl.values # quality of the rule of law

    # fG
    fG = df_indic[str(impute_final_year)].values

    return current_I, fBs, frl, fG, time_refinement_factor
    
def impute_indicators(impute_years, time_refinement_factor, current_I, fBs, frl, fG, 
    i2i_network, b2i_network, model_params, parallel_processes=None ):
    """
    Forecast the indicator levels for the next n time steps using the PPI model.
    
    Parameters:
    - filepath: The path to the saved parameter file
    - current_I: The current indicator levels
    - P: Resource allocation for each indicator for the next n time steps
    - fBs: Forecasted Budget Allocation for the next n time steps
    - fR: Projected rule of law for the next n time steps
    
    Returns:
    - Forecasted indicator levels for the next n time steps
    """
    impute_periods = impute_years*time_refinement_factor

    assert fBs.shape[1] == impute_periods, f'fBs must be an array of shape (budget_item_count, {impute_periods} )'
    assert len(frl) == len(current_I) , f'fR must have an element for each indicator. fR has {len(frl)} elements, while current_I has {len(current_I)} elements'
    
    # Extract the controlled parameters
    I0 = current_I # Initial indicator levels
    T = impute_periods # Forecast period
    G = fG # Target indicator level
    rl = frl 
    Bs = fBs # Forecasted Budget Allocation

    so_network = i2i_network
    B_dict = b2i_network

    # Extract the necessary parameters
    alphas = model_params['alpha'].values
    alphas_prime = model_params['alpha_prime'].values
    betas = model_params['beta'].values
    
    Imax = model_params.get('Imax', None)
    Imin = model_params.get('Imin', None)

    df_indic = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 

    R = df_indic.R.values
    qm = df_indic.qm.values
    bs = df_indic.bs.values if 'bs' in df_indic.columns else None
    
    # Step 2 and 4 are combined: Run the model for frl steps, 
    # using the specified P and fR as inputs
    if parallel_processes is None or parallel_processes == 1:
        indic_impute, _, _, _, _, _ = run_ppi(I0=I0, alphas=alphas, alphas_prime=alphas_prime,
                                 betas=betas, so_network=so_network, R=R, 
                                 bs=bs, qm=qm, rl=rl,
                                 Imax=Imax, Imin=Imin, 
                                 Bs=Bs, B_dict=B_dict, G=G,
                                 T=impute_periods)
    else:
        indic_impute, _, _, _, _, _ = run_ppi_parallel(I0=I0, alphas=alphas, alphas_prime=alphas_prime,
                                 betas=betas, so_network=so_network, R=R, 
                                 bs=bs, qm=qm, rl=rl,
                                 Imax=Imax, Imin=Imin, 
                                 Bs=Bs, B_dict=B_dict, G=G,
                                 T=impute_periods,
                                 parallel_processes=parallel_processes)

    # Undoing the refinement factor
    indic_impute = indic_impute[ : ,time_refinement_factor::time_refinement_factor]
    
    return indic_impute

def load_true_indicators( impute_start_year, impute_years):
    """
    Load the true indicator levels for the next forecast_periods time steps.
    """
    # save_dir = os.path.join('.','agent_based_modelling','outputs', 'calibrate', spillover_predictor_model_name.replace('/','_') )

    df_indic = pd.read_csv('./agent_based_modelling/data/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8') 
    indicator_names = df_indic.indicator_name.values
    indicator_values = df_indic[ [str(year) for year in range(impute_start_year, impute_start_year+impute_years) ] ].values
    # hyper_params = yaml.safe_load( open( os.path.join(save_dir, 'hyperparams.yaml'), 'r' ) )
    
    return indicator_values, indicator_names

def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Impute missing values in a time series.')
    
    parser.add_argument('--impute_years', type=int, default=1, help='Number of years to impute. This assumes you have the final indicator level after the periods to be imputed')
    parser.add_argument('--exp_num', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()   
    
    main(**vars(args))
    