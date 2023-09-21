import pandas as pd
import numpy as np
import os

home =  os.getcwd()[:-16]
os.chdir(home)



df = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_imputed_raw.csv', encoding='unicode_escape')
colYears = [col for col in df.columns if col.isnumeric()]
years_indices = df.columns.isin(colYears)

new_rows = []
for index, row in df.iterrows():
    vals = row[colYears].values.astype(float)
    lb = min([row.worstBound, row.worstTheoretical, vals.min()])
    ub = max([row.bestBound, row.bestTheoretical, vals.max()])
    
    nvals = (vals - lb) / (ub - lb)
    
    if row.invert==1:
        nvals = 1 - nvals

    new_row = row.values.copy()
    new_row[years_indices] = nvals
    new_rows.append(new_row)


# load governance scalers for UK / GBR
dfcc = pd.read_csv(home+"/data/england/deprivation/clean_data/controlofcorruption_GBR.csv")[colYears].iloc[0].mean()
dfrl = pd.read_csv(home+"/data/england/deprivation/clean_data/ruleoflaw_GBR.csv")[colYears].iloc[0].mean()


dff = pd.DataFrame(new_rows, columns=df.columns)
dff['qm'] = dfcc
dff['rl'] = dfrl
dff['R'] = 1.0



dfx = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_expenditure.csv')
success_dict = {}
for category in dfx.category.unique():
    M = df[(df.category1==category) | (df.category2==category) | (df.category3==category)][colYears].values
    success_dict[category] = np.sum(M[:,1::] > M[:,0:-1]) / (M.shape[0]*(M.shape[1]-1))
success_dict[np.nan] = np.nan

dff['successRates'] = [ np.nanmean([success_dict[row.category1], success_dict[row.category2], success_dict[row.category3]]) for index, row in df.iterrows() ]

dff.to_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_normalized.csv', index=False)



















































