"""
    Analysis of England Deprivation Indicators Network

    This script reads a dataset containing deprivation indicators for regions in England,
    computes the difference between consecutive years of each indicator, and then estimates
    a Directed Acyclic Graph (DAG) representing potential causal relationships between the
    indicators. This is done by employing Bayesian network learning algorithms from the 'sparsebn'
    package in R. The script proceeds to apply certain criteria to refine the estimated network
    by removing negative links between indicators within the same category and correcting extreme weights.

    Dependencies:
        - pandas
        - numpy
        - os
        - warnings
        - rpy2
        - sparsebn R package
        - sparsebnUtils R package (optional installation)
        - ccdrAlgorithm R package

    Functions:
        - No functions are defined; the script is intended to run as a standalone analysis.

    Detailed Steps:
        1. **Environment Setup**
            - Required libraries are imported.
            - The R to numpy bridge is activated with `numpy2ri.activate()`.
            - The current working directory is set to a specific parent directory.

        2. **Data Loading**
            - A CSV file containing deprivation indicators for regions in England is loaded into a pandas DataFrame.
            - Only numeric columns (assumed to represent years) are retained for analysis.

        3. **Computing Differences Between Consecutive Years**
            - The differences between consecutive years of each indicator are computed.

        4. **Estimating DAG using R and the 'sparsebn' Package**
            - The differences are converted to an R matrix object.
            - Various functions from the 'sparsebn' and related packages are used to estimate a DAG
            and the optimal regularization parameter for this DAG.

        5. **Refining the Estimated Network**
            - Negative links between indicators within the same category are identified and removed.
            - The weights (coefficients) of the links are corrected by removing potential false positives
            with extreme values (beyond the 5th and 95th percentiles).

        6. **Saving the Resulting Network**
            - The refined network is saved to a new CSV file, with columns indicating the 'From' node,
            the 'To' node, and the 'Weight' of the edge.

    Notes:
        - The 'home' directory needs to be set properly for this script to run successfully.
        - The CSV file paths are constructed relative to the 'home' directory.

"""
import os, warnings
# os.environ['R'] = '/home/akann1w0w1ck/miniconda3/envs/r_env/lib/R/bin'
# os.environ['R_HOME'] = '/home/akann1w0w1ck/miniconda3/envs/r_env/lib/R'
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

warnings.simplefilter("ignore")
sparsebn = importr('sparsebn')


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




# ## Installing R packages (optional)
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ('sparsebnUtils')
from rpy2.robjects.vectors import StrVector
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))


df = pd.read_csv('./data/ppi/pipeline_indicators_normalized.csv', encoding='unicode_escape')
colYears = [col for col in df.columns if col.isnumeric()]

S = (df[colYears].values[:,1::] - df[colYears].values[:,0:-1]).T

nr, nc = S.shape
Sr = rpy2.robjects.r.matrix(S, nrow=nr, ncol=nc)
rpy2.robjects.r.assign("S", Sr)

rpy2.robjects.r('''
    library(sparsebnUtils)
    library(sparsebn)
    library(ccdrAlgorithm)
    data <- sparsebnData(S, type = "continuous")
    dags.estimate <- sparsebn::estimate.dag(data)
    dags.param <- estimate.parameters(dags.estimate, data=data)
    selected.lambda <- select.parameter(dags.estimate, data=data)
    dags.final.net <- dags.estimate[[selected.lambda]]
    dags.final.param <- dags.param[[selected.lambda]]
    adjMatrix <- as(dags.final.param$coefs, "matrix")
    ''')
    
A = rpy2.robjects.globalenv['adjMatrix']


## this eliminates negative links between indicators within the same category
negative_coords = np.where(A<0)
cats1 = dict(zip(df.index, df.category1))
cats2 = dict(zip(df.index, df.category2))
cats3 = dict(zip(df.index, df.category3))
for i in range(len(negative_coords[0])):
    x = negative_coords[0][i]
    y = negative_coords[1][i]
    if cats1[x]==cats1[y] or cats1[x]==cats2[y] or cats1[x]==cats3[y] or \
        cats2[x]==cats1[y] or cats2[x]==cats2[y] or cats2[x]==cats3[y] or \
        cats3[x]==cats1[y] or cats3[x]==cats2[y] or cats3[x]==cats3[y]:
        A[x,y] = 0


## correct the weights by removing potential false positives with extreme values
p5 = np.percentile(A[A!=0].flatten(), 5)
p95 = np.percentile(A[A!=0].flatten(), 95)
AC = A.copy()
AC[AC>p95] = 0
AC[AC<p5] = 0


    

edges = []
for i, rowi in df.iterrows():
    for j, rowj in df.iterrows():
        if A[i,j] != 0:
            edges.append((i, j, AC[i,j]))

dff = pd.DataFrame(edges, columns=['From', 'To', 'Weight'])   
dff.to_csv('./data/ppi/ccdr.csv', index=False)
