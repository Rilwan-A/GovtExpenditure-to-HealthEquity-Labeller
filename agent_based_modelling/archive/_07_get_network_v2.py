import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings

# from causallearn.preprocessing import preprocessing
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam

warnings.simplefilter("ignore")

# Load data
df = pd.read_csv('./data/ppi/pipeline_indicators_normalized_finegrained.csv', encoding='utf-8')
trainingYears = [col for col in df.columns if col.isnumeric() and int(col) < 2018]
X = (df[trainingYears].values[:, 1::] - df[trainingYears].values[:, 0:-1]).T

# Preprocess the data
# X, _ = preprocessing(S)

# We learn the DAG structure using 3 different methodologies:
    # 1) Constraint-based causal discovery method: PC algorithm
    # 2) Score-based causal discovery method: GES algorithm with BIC Score
    # 3) Causal discovery methods based on constrained functional causal models: 


# 1)Apply the PC algorithm to estimate a DAG
# causal_graph_pc = pc(X)
# adjacency_matrix_pc = causal_graph_pc.G.graph #where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i –> j; cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicate i — j; cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

# 2) Apply the GES algorithm to estimate a DAG
record_ges = ges(X[:, 0:300:15])
adjacency_matrix_ges = record_ges['G'].graph # Record[‘G’].graph[j,i]=1 and Record[‘G’].graph[i,j]=-1 indicate i –> j; Record[‘G’].graph[i,j] = Record[‘G’].graph[j,i] = -1 indicates i — j.

# 3) Apply the DirectLiNGAM algorithm and RCD
 # We assume no negative relationship for indicators that share the same borad budget item
# prior_knowledge = -np.ones((X.shape[1], X.shape[1]))
# cats1 = dict(zip(df.index, df.category1))
# cats2 = dict(zip(df.index, df.category2))
# cats3 = dict(zip(df.index, df.category3))
# for x in range( X.shape[1] ):
#     for y in range( X.shape[1] ):
#         if cats1[x] == cats1[y] or cats1[x] == cats2[y] or cats1[x] == cats3[y] or \
#             cats2[x] == cats1[y] or cats2[x] == cats2[y] or cats2[x] == cats3[y] or \
#             cats3[x] == cats1[y] or cats3[x] == cats2[y] or cats3[x] == cats3[y]:
#             prior_knowledge[x, y] = 0

# model_dl = lingam.DirectLiNGAM( prior_knowledge=prior_knowledge )
# model_dl.fit(X[:, 0:50:9])
# adjacency_matrix_lingam = model_dl.adjacency_matrix_


model_rcd = lingam.RCD()
model_rcd.fit(X[:, :])
adjacency_matrix_rcd = model_rcd.adjacency_matrix_

# Reformatting the adjacency matrices to standard formatting
def reformat(adjacency_matrix):

    # Formatting the adjacency matrix
    n, m = adjacency_matrix.shape
    for i in range(n):
        for j in range(m):
            if adjacency_matrix[j, i] == 1 and adjacency_matrix[i, j] == -1:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 0
            elif adjacency_matrix[j, i] == -1 and adjacency_matrix[i, j] == 1:
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 1
            elif adjacency_matrix[j, i] == 1 and adjacency_matrix[i, j] == 1:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
            elif adjacency_matrix[j, i] == -1 and adjacency_matrix[i, j] == -1:
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 0    
    return adjacency_matrix

# adjacency_matrix_pc = reformat(adjacency_matrix_pc)
# adjacency_matrix_ges = reformat(adjacency_matrix_ges)

# This eliminates negative links between indicators within the same category
def get_network(A, df):
    negative_coords = np.where(A < 0)
    cats1 = dict(zip(df.index, df.category1))
    cats2 = dict(zip(df.index, df.category2))
    cats3 = dict(zip(df.index, df.category3))

    for i in range(len(negative_coords[0])):
        x = negative_coords[0][i]
        y = negative_coords[1][i]
        if cats1[x] == cats1[y] or cats1[x] == cats2[y] or cats1[x] == cats3[y] or \
            cats2[x] == cats1[y] or cats2[x] == cats2[y] or cats2[x] == cats3[y] or \
            cats3[x] == cats1[y] or cats3[x] == cats2[y] or cats3[x] == cats3[y]:
            A[x, y] = 0

    # Correct the weights by removing potential false positives with extreme values
    p5 = np.percentile(A[A != 0].flatten(), 5)
    p95 = np.percentile(A[A != 0].flatten(), 95)
    AC = A.copy()
    AC[AC > p95] = 0
    AC[AC < p5] = 0

    edges = []
    for i, rowi in df.iterrows():
        for j, rowj in df.iterrows():
            if A[i, j] != 0:
                edges.append((i, j, AC[i, j]))
    
    dff = pd.DataFrame(edges, columns=['From', 'To', 'Weight'])
    return dff


df_ges = get_network(adjacency_matrix_ges, df)
df_ges.to_csv('./data/ppi/i2i_networks/ges.csv', index=False)

df_lingam = get_network(adjacency_matrix_lingam, df)
df_lingam.to_csv('./data/ppi/i2i_networks/lingam.csv', index=False)

df_rcd = get_network(adjacency_matrix_rcd, df)
df_rcd.to_csv('./data/ppi/i2i_networks/rcd.csv', index=False)
