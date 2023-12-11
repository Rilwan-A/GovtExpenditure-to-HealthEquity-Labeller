import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

warnings.simplefilter("ignore")

home =  os.getcwd()[:-16]




os.chdir(home+'/code/deprivation/')
import policy_priority_inference as ppi





df_indis = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_normalized.csv', encoding='utf-8')
colYears = [col for col in df_indis.columns if str(col).isnumeric()]
df_exp = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_expenditure.csv')
expCols = [col for col in df_exp.columns if str(col).isnumeric()]



num_years = len(colYears)
T = len(expCols)


N = len(df_indis) # number of indicators
IF = []
series = df_indis[colYears].values
I0 = series[:,0]
for serie in series:
    x = np.array([float(year) for year in colYears]).reshape((-1, 1))
    y = serie
    model = LinearRegression().fit(x, y)
    if serie[0] == serie[-1]:
        IF.append(model.predict([[2019]])[0])
    else:
        IF.append(serie[-1])
IF = np.array(IF)
success_rates = df_indis.successRates.values # success rates
R = np.ones(N) # instrumental indicators
qm = df_indis.qm.values # quality of monitoring
rl = df_indis.rl.values # quality of the rule of law
indis_index = dict([(code, i) for i, code in enumerate(df_indis.seriesCode)]) # used to build the network matrix


df_net = pd.read_csv(home+'data/england/deprivation/clean_data/network.csv')

A = np.zeros((N, N)) # adjacency matrix
for index, row in df_net.iterrows():
    i = int(row.From)
    j = int(row.To)
    w = row.Weight
    A[i,j] = w

Bs = df_exp[expCols].values # disbursement schedule (assumes that the expenditure programmes are properly sorted)

df_rela = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_relation_table.csv')

B_dict = {} # PPI needs the relational table in the form of a Python dictionary
for index, row in df_rela.iterrows():
    B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]


parallel_processes = 40 # number of cores to use
threshold = 0.8 # the quality of the calibration (I choose a medium quality for illustration purposes)
low_precision_counts = 75 # number of low-quality iterations to accelerate the calibration

parameters = ppi.calibrate(I0, IF, success_rates, A=A, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
              T=T, threshold=threshold, parallel_processes=parallel_processes, verbose=True,
             low_precision_counts=low_precision_counts)



dff = pd.DataFrame(parameters[1::,:], columns=parameters[0])
dff.to_csv(home+'data/england/deprivation/clean_data/parameters.csv', index=False)
