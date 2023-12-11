import pandas as pd
import numpy as np
import os
from agent_based_modelling.b2i_edge_estimation.create_b2i_candidate_edges import concatenate_name_age


df_indic_imputed = pd.read_csv('./data/ppi/pipeline_indicators_imputed_raw_2013_2019.csv', encoding='utf-8')
colYears = [col for col in df_indic_imputed.columns if col.isnumeric()]
trainingYears = [str(year) for year in range(2013, 2018)]  # Only include years 2013 to 2017
years_indices = df_indic_imputed.columns.isin(colYears)

new_rows = []
for index, row in df_indic_imputed.iterrows():
    vals = row[colYears].values.astype(float)
    training_vals = row[trainingYears].values.astype(float)  # Only values for 2013-2017
    lb = min([row.worstBound, row.worstTheoretical, training_vals.min()])
    ub = max([row.bestBound, row.bestTheoretical, training_vals.max()])
    
    nvals = (vals - lb) / (ub - lb)
    
    if row.invert==1:
        nvals = 1 - nvals

    new_row = row.values.copy()
    new_row[years_indices] = nvals
    new_rows.append(new_row)


# load governance scalers for UK / GBR
dfcc = pd.read_csv("./data/ppi/controlofcorruption_GBR.csv")[colYears].iloc[0].mean()
dfrl = pd.read_csv("./data/ppi/ruleoflaw_GBR.csv")[colYears].iloc[0].mean()


dff = pd.DataFrame(new_rows, columns=df_indic_imputed.columns)
dff['qm'] = dfcc
dff['rl'] = dfrl
dff['R'] = 1.0


# For each budget item type this calculates the sucess rate as for each broad budget item, how often the indicators improved from one year to the next
# Note since we are operating with finegrained budget items, we use the broad budget items success rate as a proxy for the finegrained budget items success rate

dfx = pd.read_csv('./data/ppi/pipeline_expenditure_finegrained.csv')
success_dict = {}
for category in dfx.category.unique():
    # calculate success rate only over training years of 2013-2017
    M = df_indic_imputed[(df_indic_imputed.category1==category) | (df_indic_imputed.category2==category) | (df_indic_imputed.category3==category)][trainingYears].values
    success_dict[category] = np.sum(M[:,1::] > M[:,0:-1]) / (M.shape[0]*(M.shape[1]-1))
success_dict[np.nan] = np.nan

# Then for each indicator we take the average of the success rates of the broad budget items it is associated with
dff['successRates'] = [ np.nanmean([success_dict[row.category1], success_dict[row.category2], success_dict[row.category3]]) for index, row in df_indic_imputed.iterrows() ]

dff['indicator_name'] = dff[['seriesName', 'Age', 'group']].apply(lambda x: concatenate_name_age(x[0], x[1], x[2]), axis=1)

dff.to_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', index=False)



















































