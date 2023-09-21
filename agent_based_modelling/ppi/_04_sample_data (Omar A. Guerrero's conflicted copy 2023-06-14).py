import pandas as pd
import numpy as np
import os

home =  os.getcwd()[:-16]
os.chdir(home)


# We limit the sample of indicators to those with 5 or more years of data from 2013,
# as expenditure data starts on 2013
df = pd.read_csv(home+'data/england/deprivation/clean_data/data_indicators_raw.csv', encoding='unicode_escape')
df = df[[c for c in df.columns if c != 'polarity']]
df = df.drop_duplicates()

# clean the name of columns
df.rename(columns=dict([(col, col[0:4]) for col in df.columns if col.isnumeric()]), inplace=True)

# discard 2012 and earlier data
df.drop([col for col in df.columns if col.isnumeric() and int(col) < 2013], inplace=True, axis=1)
df.drop([col for col in df.columns if col.isnumeric() and int(col) > 2019], inplace=True, axis=1)
colYears = [col for col in df.columns if col.isnumeric()]

# select indicators with 5 or more observation
df = df[(~df[colYears].isnull()).sum(axis=1) >= 5]

# select national or deciles
new_rows = []
for seriesCode, group in df.groupby('seriesCode'):
    for index, row in group.iterrows():
        if row.group not in ['Least deprived decile', 'Most deprived decile', 'national']:
            continue
        elif row.group == 'national': 
            if 'Least deprived decile' not in group.group.values or 'Most deprived decile' not in group.group.values:
                new_rows.append(row.values)
        elif 'Least deprived decile' in group.group.values and 'Most deprived decile' in group.group.values:
            new_rows.append(row.values)
    
df = pd.DataFrame(new_rows, columns=df.columns)


# add categories of expenditures (or public health profiles)
#categories = pd.read_csv(home+'data/england/deprivation/clean_data/data_relation_table.csv', encoding = 'unicode_escape')
#df = df.merge(categories, on = 'seriesCode', validate = 'many_to_many', how = 'left')

# reject rows with null initial and filnal values
# df = df.loc[df.I0.notnull(),]
# df = df.loc[df.IF.notnull(),]

# write file
df.to_csv(home+'data/england/deprivation/clean_data/pipeline_indicators_sample_raw.csv', index=False)



 



# 2) subset expenditutre to categories of expenditure for which we have indicators
dfx = pd.read_csv(home+'data/england/deprivation/clean_data/data_expenditure_trend.csv', encoding='unicode_escape')

dfx.rename(columns=dict([(col, col[0:4]) for col in dfx.columns if col.isnumeric()]), inplace=True)
dfx.drop([col for col in dfx.columns if col.isnumeric() and int(col)<2013], inplace=True, axis=1)
dfx = dfx[(dfx[colYears[0:-3]]<0).sum(axis=1) == 0]
dfx['mean'] = dfx[colYears[0:-3]].mean(axis=1)

T = 49
t = int(T/len(colYears))

new_rows = []
for index, row in dfx.iterrows():
    new_row = [row['seriesCode'], row['Area Code'], row.category]
    for year in colYears:
        new_row += [row['mean']/t for i in range(t)]
    new_rows.append(new_row)
    
dfxf = pd.DataFrame(new_rows, columns=['seriesCode', 'Area Code', 'category']+[str(i) for i in range(T)])
categories = set(df[['category1', 'category2', 'category3']].values.flatten()[df[['category1', 'category2', 'category3']].values.flatten().astype(str)!='nan'])
dfxf = dfxf[dfxf.category.isin(categories)]
dfxf = dfxf.groupby('category').sum()
dfxf.reset_index(inplace=True)
dfxf.loc[:, 'program'] = range(len(dfxf))
dfxf.drop(['seriesCode', 'Area Code'], axis='columns', inplace=True)
dfxf.to_csv(home+'data/england/deprivation/clean_data/pipeline_expenditure.csv', index=False)



dict_rela = {}
for index, row in df.iterrows():
    dict_rela[row.seriesCode] = dfxf[dfxf.category.isin(row[['category1', 'category2', 'category3']])].program.unique().tolist()

ncols = max([len(c) for c in dict_rela.values()])

new_rows = []
for key, value in dict_rela.items():
    new_row = [key] + value + [np.nan for i in range(ncols-len(value))]
    new_rows.append(new_row)

dfr = pd.DataFrame(new_rows, columns=['seriesCode']+[str(i) for i in range(ncols)])
dfr.to_csv(home + 'data/england/deprivation/clean_data/pipeline_relation_table.csv', index=False)


