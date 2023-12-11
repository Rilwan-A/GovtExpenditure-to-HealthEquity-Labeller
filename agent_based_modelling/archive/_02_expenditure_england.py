# -*- coding: utf-8 -*-
"""
Modified 4th March 2023
# This script aims to :
    1) retreieves metadata and data on expenditure from fingertips,
    2) checks the budget categories for which at least an indicator was retrieved
    3) format wide timeseries and creates columns needed for ppi
"""
# Install fingertips with pip install fingertips_py

# 0) import libraries, define paths
import pandas as pd
import numpy as np
import fingertips_py as ftp
import os

home =  os.getcwd()[:-16]
os.chdir(home)

# read indicators data
data_indi = pd.read_csv(home + 'data/england/deprivation/clean_data/data_indicators_raw.csv')
# spot profiles
metadata = ftp.get_metadata_for_profile_as_dataframe(155)
metadata = metadata.loc[metadata.Indicator.str.contains("Spend"),]
metadata = metadata[['Indicator ID', 'Indicator','Year type','Unit','Value type','Frequency']]
metadata['profile_id'] = 155
# please note that the following are constant throughtout the data: Unit = £, Value type=Count, Year type = financial, Polarity = BOB

# 2) based on metadata from 1) we pull each expenditure together with some metadata, 
# appends all indicators and retrieve best and worst bounds
allexpenditure = []
for index, row in metadata.iterrows(): 
    i = row['Indicator ID'] 
    j = row['profile_id'] 
    df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
        area_type_id=102, parent_area_type_id=15, profile_id=j,
        include_sortable_time_periods=1, is_test=False)
    if df.empty == True:
        df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
            area_type_id=202, parent_area_type_id=15, profile_id=j,
            include_sortable_time_periods=1, is_test=False)

   # creates a unique ID based on indicator 
    df['seriesCode'] = df['Indicator ID']
    df['seriesName'] = df['Indicator Name']
    
    # duplication is not an issue at geography-level spend data
    assert df.duplicated(subset =  ['seriesCode','Time period Sortable', 'Area Code']).any() == False

    df = df[['seriesCode','seriesName','Value','Time period Sortable','Indicator ID','Area Code']]

    dfeng = df.groupby(by = ['Indicator ID','Time period Sortable'], 
                       observed=True, dropna=True, as_index = False)['Value'].sum()
    dfeng['Area Code'] = "E92000001"  # England
    dfeng['seriesCode'] = dfeng['Indicator ID'].astype(str) + "_" + dfeng['Area Code'].astype(str)
    dfeng['seriesName'] = df['seriesName'][0]

    allexpenditure.append(dfeng)
    allexpenditure.append(df)

data_exp = pd.concat(allexpenditure)

# We add SPOT categories which match to expenditure/budget
categories = pd.read_csv('/Users/oguerrero/Library/CloudStorage/Dropbox/Projects/ppi4health/data/england/deprivation/clean_data/spot_indicator_mapping_table.csv', encoding = 'utf-8')
categories = categories.loc[categories.type == "Spend",['category','name']].rename(columns = {'name':'seriesName'}).drop_duplicates()
data_exp = data_exp.merge(categories, on = 'seriesName', validate = 'many_to_many', how = 'outer')

# 2) We drop one expenditure row for which two categories are reported (Central and Education) to avoid duplication
data_exp = data_exp.loc[~((data_exp.seriesCode==2131) & (data_exp.category == "Central")),]

# pivot wide on years, we fill NAs to avoid errors in 3) assembly disbursment matrix
data_exp = pd.pivot(data_exp, index = ['seriesCode','seriesName', 'Indicator ID', \
            'Area Code', 'category','colour'], columns = 'Time period Sortable', \
                        values = 'Value').reset_index().fillna(-1)


data_exp.to_csv(home + 'data/england/deprivation/clean_data/data_expenditure_raw.csv', index=False)



