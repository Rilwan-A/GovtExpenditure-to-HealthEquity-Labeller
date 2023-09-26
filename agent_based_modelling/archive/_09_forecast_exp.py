# -*- coding: utf-8 -*-
"""
Expenditure Trend Forecasting for England Deprivation Data

This script loads a dataset containing expenditure trends for different regions
in England, identifies trends in the data where the expenditure decreased at any
point in time, and then forecasts future expenditure using Gaussian Process
Regressors.

Dependencies:
    - pandas
    - numpy
    - os
    - warnings
    - scipy.signal
    - statsmodels.tsa.tsatools
    - sklearn.gaussian_process
    - sklearn.gaussian_process.kernels

Functions:
    - No functions are defined; the script is intended to run as a standalone analysis.

Detailed Steps:
    1. **Environment Setup**
        - Required libraries are imported.
        - The current working directory is set to a specific parent directory.

    2. **Data Loading**
        - A CSV file containing expenditure trends for regions in England is loaded into a pandas DataFrame.
        - Only numeric columns (assumed to represent years) are retained for analysis.

    3. **Trend Analysis and Forecasting**
        - Iterate through each row (assumed to represent a different region or entity).
        - Identify if the expenditure decreases at any year for that entity.
        - For entities with decreasing expenditure, a Gaussian Process Regressor is trained
          using the years before the decrease as data.
        - The trained Gaussian Process Regressor is then used to forecast the expenditure
          for the subsequent years.
        - Ensure that the forecasted values are not less than the observed values to maintain
          an increasing or stable trend.

    4. **Saving the Forecasted Data**
        - The forecasted expenditure trends are saved to a new CSV file.

Notes:
    - The 'home' directory needs to be set properly for this script to run successfully.
    - The CSV file paths are constructed relative to the 'home' directory.
    - The script assumes that the columns of the CSV file that are purely numeric represent different years.
    - The script prints the index of the current row being processed every 100 rows.

@author: giselara
Created on Sat Jun 17 19:00:13 2023
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings
from scipy.signal import detrend
from statsmodels.tsa.tsatools import detrend as dtd
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")


home =  os.getcwd()[:-16]
os.chdir(home)


df_exp = pd.read_csv(home + 'data/england/deprivation/clean_data/data_expenditure_trend.csv', encoding='utf-8')


# define the set of columns to adjust
colYears = [col for col in df_exp.columns if col.isnumeric()]
years = np.array([int(col) for col in df_exp.columns if str(col).isnumeric()])
years_indices = df_exp.columns.isin(colYears)


new_rows = []
for index, row in df_exp.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]

    if ((row[colYears] - row[colYears].shift(1) < 0).any() == True):
        decreased = np.where( (row[colYears] - row[colYears].shift(1) < 0))[0][0]
    #    yrs_decreased = colYears[decreased]
     #   print(yrs_decreased)
        yrs_forecast = list(range(decreased,7))
        yrs_increase = list(range(0,decreased))
        new_row = row.values.copy()
    
            
        vals = row[colYears].values.copy()
    
        x = years[yrs_increase] #[observations]
        y = vals[yrs_increase] # vals[observations]
        X = x.reshape(-1, 1)
    
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)
    
        x_pred = years.reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[yrs_forecast] = y_pred[yrs_forecast]
        new_row[years_indices] = vals
        
        # these rows were added so our prediction are always equal or above observed values
        old_vals = row[colYears].values.copy()
        new_vals = vals
        for i in range(7):
            if new_vals[i] < old_vals[i]:
                new_vals[i] = old_vals[i]
        
        
        new_row[years_indices] = new_vals
        new_rows.append(new_row)
        
        
        if index%100==0:
            print(index)



dff = pd.DataFrame(new_rows, columns=df_exp.columns)
dff.to_csv(home+'data/england/deprivation/clean_data/pipeline_expenditure_forecasted.csv', index=False)
