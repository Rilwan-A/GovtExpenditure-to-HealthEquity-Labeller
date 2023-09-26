"""
Policy Priority Inference (PPI) Analysis for Deprivation Data in England

This script runs a Monte Carlo simulation using the PPI algorithm to infer the
priorities of expenditure programmes in relation to a set of deprivation indicators
in England. It imports various datasets, processes them, and performs simulations
to understand how different expenditure allocations affect the indicators over time.
Results of the simulation, including changes in indicator values and expenditure
programme effects, are saved as CSV files and visualized through a series of plots.

Input Data:
- 'pipeline_indicators_normalized.csv': Normalized values of deprivation indicators.
- 'pipeline_expenditure.csv': Historical expenditure data for various programmes.
- 'network.csv': The network structure between different indicators.
- 'pipeline_relation_table.csv': The relations between indicators and expenditure programmes.
- 'parameters.csv': Parameters for the PPI model.

Data Processing:
- Imports and processes various CSV files to prepare inputs for the PPI model.
- Constructs various matrices and vectors used in the PPI algorithm, including initial
  values of indicators, success rates, instrumental indicators, quality of monitoring,
  and rule of law parameters.
- Reads and processes a network adjacency matrix representing the relationships between indicators.
- Reads and processes a relational table representing the relationships between indicators and expenditure programmes.

Simulation Approach:
- Performs Monte Carlo simulations to estimate the potential effects of different expenditure
  allocations on various indicators.
- Simulates how different expenditure programmes might improve or degrade specific indicators
  over time under different scenarios.
- Allows for flexible scenario analysis via changes in the parameters of the PPI algorithm.
- Includes a counterfactual analysis that assumes a linear growth in the expenditure.

Output Generation:
- Saves the simulated time series of indicators as CSV files.
- Creates several plots that visualize the effects of different expenditure programmes on indicators
  over time, as well as comparisons between baseline and counterfactual scenarios.
- Saves plots as PNG files.

Note:
- Ensure that the disbursement schedule is consistent with T (number of time periods for simulation),
  otherwise PPI will simulate the T of the calibration.
- The script sets a seed for the random number generator for reproducibility.

"""

import pandas as pd
import numpy as np
from math import isnan
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os, warnings

home =  os.getcwd()[:-16]
os.chdir('c:/')
import policy_priority_inference as ppi



df_indis = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_normalized.csv', encoding='utf-8')
colYears = [col for col in df_indis.columns if str(col).isnumeric()]
df_exp = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_expenditure.csv')
expCols = [col for col in df_exp.columns if str(col).isnumeric()]



num_years = len(colYears)
T = len(expCols)


N = len(df_indis) # number of indicators
IF = []
praed = []
series = df_indis[colYears].values
I0 = series[:,0]
for serie in series:
    x = np.array([float(year) for year in colYears]).reshape((-1, 1))
    y = serie
    model = LinearRegression().fit(x, y)
    praed.append(model.predict)
    if serie[0] >= serie[-1]:
        if model.predict([[2019]])[0] >= serie[0]:
            IF.append(model.predict([[2019]])[0])
        else:
            IF.append(serie[0])
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


df_rela = pd.read_csv(home+'data/england/deprivation/clean_data/pipeline_relation_table.csv')

B_dict = {} # PPI needs the relational table in the form of a Python dictionary
for index, row in df_rela.iterrows():
    B_dict[int(row.indicator_index)] = [int(programme) for programme in row.values[1::][row.values[1::].astype(str)!='nan']]


# DISBURSEMENT SCHEDULE (make sure that the disbursement schedule is consistent with T, otherwise PPI will simulate the T of the calibration)
T = 49
Bs_retrospective = df_exp.values[:,1::] # disbursement schedule (assumes that the expenditure programmes are properly sorted)
# Create a new disbursement schedule assuming that expenditure will be the same as the last period of the sample
#Bs = df_exp[expCols].values # disbursement schedule (assumes that the expenditure programmes are properly sorted)
Bs = np.tile(Bs_retrospective[:,-1], (T,1)).T
#Bs = np.array([Bs])

# PARAMETERS for specific decile
df_params = pd.read_csv(f"{home}/data/england/deprivation/clean_data/parameters.csv")
alphas = df_params.alpha.values
alphas_prime = df_params.alpha_prime.values
betas = df_params.beta.values


np.random.seed(123)
goals = np.random.rand(N)*(IF - I0) + I0
sample_size = 100 # number of Monte Carlo simulations

outputs = []
for sample in range(sample_size):
    output = ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
                Imax=IF , Imin=I0, Bs=Bs, B_dict=B_dict, T=T, G=goals)
    outputs.append(output)


    # separate the outputs into lists of time series
tsI, tsC, tsF, tsP, tsS, tsG = zip(*outputs)

    # compute the average time series of the indicators
tsI_hat = np.mean(tsI, axis=0)

    # make a new DataFrame with the indicators' information
new_rows = []
for i, serie in enumerate(tsI_hat):
    new_row = [df_indis.iloc[i].seriesCode, df_indis.iloc[i].category1, df_indis.iloc[i]['Indicator ID'], df_indis.iloc[i].group] + serie.tolist()
    new_rows.append(new_row)

df_output = pd.DataFrame(new_rows, columns=['seriesCode','category','Indicator ID','group']+list(range(T)))
colour_dict = {"Central": "#A21942",	"Child Health": "#FD9D24",	"Cultural": "#FF3A21", \
        "Drugs and Alcohol": "#E5243B",	"Education": "#DDA63A",	"Env & Reg": "#4C9F38",
        "Health Improvement": "#C5192D",	"Health Protection": "#26BDE2",	"Healthcare": "#FCC30B",
        "Highways": "#FD6925",	"Housing": "#BF8B2E",	"Mental Health": "#3F7E44",	"Planning": "#0A97D9",
        "Public Health": "#56C02B",	"Sexual Health": "#00689D",	"Social Care - Adults": "#19486A",
        "Social Care - Child": "#19486A",	"Tobacco Control": "#E5243B"}
df_output['colour'] = df_output.category.map(colour_dict)
df_output['goal'] = goals    
df_output.to_csv(f"{home}/data/england/deprivation/clean_data/df_output.csv")
    
# GRAPHICS
plt.figure(figsize=(8, 5))
for index, row in df_output.iterrows():
    plt.plot(row[range(T)], color=row.colour, linewidth=3)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlim(0,T)
plt.xlabel('time')
plt.ylabel('indicator level')
plt.tight_layout()
plt.savefig(f"{home}/data/england/deprivation/clean_data/fig1.png")


plt.figure(figsize=(8, 5))
for index, row in df_output.iterrows():
    plt.plot(row[range(T)]-row[0], color=row.colour, linewidth=3)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlim(0,T)
plt.xlabel('time')
plt.ylabel('change with respect to initial condition')
plt.tight_layout()
plt.savefig(f"{home}/data/england/deprivation/clean_data/fig2.png")

df_output['seriesName'] = df_output['Indicator ID'].map(label) + df_output.group.map(batch_dict).astype(str)
#df_output['sortable'] = df_output.groupby('seriesCode')[expCols].agg('mean') #df_output[3:50]].apply(lambda x: np.mean(x), axis=1)

#df_outputnp.mean(df_output[[expCols]])

for i in df_output.category.unique():
    df_output_sorted = df_output.loc[df_output.category == i,].reset_index(drop = True) #sort_values(by = df_output.[expCols].agg('mean')).reset_index(drop = False)
    plt.figure(figsize=(14, 5))
    for index, row in df_output_sorted.iterrows():
        plt.bar(index, row[T-1], color=row.colour, linewidth=3)
        plt.plot([index, index], [row[T-1], row.goal], color=row.colour, linewidth=1)
        plt.plot(index, row.goal, '.', mec='w', mfc=row.colour, markersize=15)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    Num = len(df_output_sorted.seriesCode) # number of indicators
    plt.xlim(-1, Num)
    plt.xticks(range(Num))
    plt.gca().set_xticklabels(df_output_sorted.seriesName, rotation=90)
    plt.xlabel('indicator')
    plt.ylabel('level')
    plt.tight_layout()
    plt.savefig(f"{home}/data/england/deprivation/clean_data/f3_{i}.png")

# define linear growth coefficients
linear_growth = np.tile(np.linspace(0, 2, T), (Bs.shape[0],1))
Bs3 = Bs*(1+linear_growth)

outputs = []
for sample in range(sample_size):
    output = ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl,
                Imax=IF , Imin=I0, Bs=Bs3, B_dict=B_dict, T=T, G=goals)
    outputs.append(output)

# separate the outputs into lists of time series
tsI, tsC, tsF, tsP, tsS, tsG = zip(*outputs)

# copmute the average time series of the indicators
tsI_hat = np.mean(tsI, axis=0)

# make a new dataframe with the indicators' information
new_rows = []
for i, serie in enumerate(tsI_hat):
    new_row = [df_indis.iloc[i].seriesCode, df_indis.iloc[i].category1] + serie.tolist()
    new_rows.append(new_row)

df_output2 = pd.DataFrame(new_rows, columns=['seriesCode','Indicator ID' ,'category']+list(range(T)))
color_dict = {"Central": "#A21942",	"Child Health": "#FD9D24",	"Cultural": "#FF3A21",	"Drugs and Alcohol": "#E5243B",	"Education": "#DDA63A",	"Env & Reg": "#4C9F38",	"Health Improvement": "#C5192D",	"Health Protection": "#26BDE2",	"Healthcare": "#FCC30B",	"Highways": "#FD6925",	"Housing": "#BF8B2E",	"Mental Health": "#3F7E44",	"Planning": "#0A97D9",	"Public Health": "#56C02B",	"Sexual Health": "#00689D",	"Social Care - Adults": "#19486A",	"Social Care - Child": "#19486A",	"Tobacco Control": "#E5243B"}
df_output2['color'] = df_output2.category.map(color_dict)

df_output2['goal'] = goals
df_output2.to_csv(f"{home}/data/england/deprivation/clean_data/output2.csv")

plt.figure(figsize=(6, 6))
for index, row in df_output.iterrows():
    if row.goal > row[T-1]:  # consider only those indicators that would not reach their goals
        plt.plot((row.goal-row[T-1]), (df_output2.iloc[index].goal-df_output2.iloc[index][T-1]),
                 '.', mec='w', mfc=row.colour, markersize=20, label="{}".format(df_output2.category))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel('baseline development gap')
plt.ylabel('counterfactual development gap')
plt.tight_layout()
plt.savefig(f"{home}/data/england/deprivation/clean_data/fig4.png")
