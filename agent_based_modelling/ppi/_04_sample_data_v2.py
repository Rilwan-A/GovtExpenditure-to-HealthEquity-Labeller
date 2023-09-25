import pandas as pd
import numpy as np
import os


# We limit the sample of indicators to those with 5 or more years of data from 2013,
# as expenditure data starts on 2013
df = pd.read_csv('data/ppi/data_indicators_raw.csv', encoding='unicode_escape')
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

# select indicators with variance higher than 0.0001
# threshold 0.0001 discards no indicators but 5 rows
# threshold 0.001 discards 4 indicators (258,90316,91269,93353)
# 91269 is on "Drugs and Alcohol"
# took the lower thresholds to keep the most indicators
df = df.loc[ (np.var(df[colYears], axis = 'columns') >= 0.0001) ]

# select national or deciles
new_rows = []
for seriesCode, group in df.groupby('Indicator ID'):
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





# 2) subset finegrained budget items by choosing those which share a broad budget item with an indicator 
dfx = pd.read_csv('data/ppi/data_expenditure_trend_finegrained_2013_2019.csv', encoding='unicode_escape')

dfx.rename(columns=dict([(col, col[0:4]) for col in dfx.columns if col.isnumeric()]), inplace=True)
dfx.drop([col for col in dfx.columns if col.isnumeric() and int(col)<2013], inplace=True, axis=1)
dfx = dfx[(dfx[colYears[0:-3]]<0).sum(axis=1) == 0]

T = 49
t = int(T/len(colYears))

new_rows = []
for index, row in dfx.iterrows():
    new_row = [row['seriesCode'], row['seriesName'], row.category]
    prev_value = row['2013']  # Initialize with 2013 for backward interpolation
    for year in colYears:
        if year == '2013':
            # Calculate the slope between 2013 and 2014
            slope = row['2014'] - row['2013']
            # Estimate the value in 2012 using linear extrapolation
            estimated_2012_value = row['2013'] - slope
            # Interpolate values from estimated 2012 value to 2013 value
            interpolated_values = np.linspace(estimated_2012_value, row['2013'], t+1)[1:]
        else:
            # For other years, interpolate from previous year's end value to the current year value
            interpolated_values = np.linspace(prev_value, row[year], t+1)[1:]
        new_row.extend(interpolated_values)
        prev_value = row[year]  # Update the previous year's value
    new_rows.append(new_row)

dfxf = pd.DataFrame(new_rows, columns=['seriesCode', 'seriesName', 'category']+[str(i) for i in range(T)])
dfxf.drop(['seriesCode'], axis='columns', inplace=True)
categories = set(df[['category1', 'category2', 'category3']].values.flatten()[df[['category1', 'category2', 'category3']].values.flatten().astype(str)!='nan'])
dfxf = dfxf[dfxf.category.isin(categories)]
dfxf.reset_index(inplace=True)
dfxf.loc[:, 'program'] = range(len(dfxf))
dfxf.to_csv('data/ppi/pipeline_expenditure_finegrained.csv', index=False)


# write file
#df.replace(to_replace='weight_management', value=np.nan, inplace=True)               
filter_categories = df.category1.isin(dfxf.category.unique()).astype(int) + df.category2.isin(dfxf.category.unique()).astype(int) + df.category3.isin(dfxf.category.unique()).astype(int)
df = df[filter_categories>0]

df.reset_index(inplace=True, drop=True)
df.to_csv('data/ppi/pipeline_indicators_sample_raw.csv', index=False)

dict_rela = {}
for index, row in df.iterrows():
    dict_rela[index] = dfxf[dfxf.category.isin(row[['category1', 'category2', 'category3']])].program.unique().tolist()

ncols = max([len(c) for c in dict_rela.values()])

new_rows = []
for key, value in dict_rela.items():
    new_row = [key] + value + [np.nan for i in range(ncols-len(value))]
    new_rows.append(new_row)

dfr = pd.DataFrame(new_rows, columns=['indicator_index']+[str(i) for i in range(ncols)])
dfr.to_csv('data/ppi/pipeline_relation_table_finegrained.csv', index=False)