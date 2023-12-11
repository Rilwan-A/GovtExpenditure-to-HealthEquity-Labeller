'''
World Development Indicators downloaded on 2023 May 09
to reshape and normalise
'''

import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-16]
os.chdir(home)


dfcc = pd.read_excel(home+"/data/england/deprivation/clean_data/wgidataset.xlsx", sheet_name='ControlofCorruption', skiprows=14)
dfrl = pd.read_excel(home+"/data/england/deprivation/clean_data/wgidataset.xlsx", sheet_name='RuleofLaw', skiprows=14)


colYears = [str(c) for c in range(1996, 2001, 2)]+[str(c) for c in range(2002, 2022)]


relevant_columns = [c for c in dfcc.columns if 'Estimate' in c or 'Code' in c]
dfcc = pd.DataFrame(dfcc[relevant_columns].values, columns=['countryCode']+colYears)
dfrl = pd.DataFrame(dfrl[relevant_columns].values, columns=['countryCode']+colYears)


# A loop loads the data frame, normalises the data and writes it on a csv file
for df in [dfcc, dfrl]:
    if(df.iloc[0,7] == dfcc.iloc[0,7]):
        identifier = 'controlofcorruption'
    else:
        identifier = 'ruleoflaw'
        
    df = df.loc[df.countryCode.isin(['GBR']),]
    min_val = min([np.nanmin(df[colYears].values), -2.5])
    max_val = max([np.nanmax(df[colYears].values), 2.5])

    new_rows = []
    for index, row in df.iterrows():
        vals = row[colYears].values.copy()
        valsn = (vals - min_val)/(max_val - min_val)
        new_row = [row.countryCode] + valsn.tolist()
        new_rows.append(new_row)
    
    dff = pd.DataFrame(new_rows, columns=df.columns)
    dff.to_csv(home+"/data/england/deprivation/clean_data/"+identifier+"_GBR.csv", index=False)
