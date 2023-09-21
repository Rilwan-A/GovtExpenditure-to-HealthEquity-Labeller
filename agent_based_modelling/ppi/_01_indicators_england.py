# -*- coding: utf-8 -*-
"""
Modified 18th April 2023
# This script aims to :
    1) retrieve metadata for profile 155 SPOT indicators,
    2) retrieve data from fingertips by indicator in SPOT profile
    3) deduplicate indicators for deprivation deciles and national estimates
    4) format wide on timeseries and
    5) creates 'invert', 'count_valid', among other columns needed for ppi 
"""
# Install fingertips with pip install fingertips_py

# 0) import libraries, define paths
import pandas as pd
import numpy as np
import os
import fingertips_py as ftp
from fingertips_py import get_data_for_indicator_at_all_available_geographies

home =  os.getcwd()[:-16]
os.chdir(home)

# load categories of outcomes and expenditures (spot)
categories = pd.read_csv(home + 'data/england/deprivation/clean_data/data_relation_table.csv', encoding = 'unicode_escape', low_memory=False).rename(columns = {'id':'seriesCode'})

# 0) I define a dictionary for 5 indicators for which national trend is separate from IMD.
national_indicators_dic = {
    2501:92785,
    2514:92795,    
    2557:90886,
    2559:90885
    }

national_category_dic = {
    92757:"Child Health", 
    92785:"Healthcare", 
    92795:"Healthcare", 
    90886:"Social Care - Child",    
    90885:"Social Care - Child",
    91126:"Central",
    91382:"Drugs and Alcohol",
    91414:"Drugs and Alcohol",
    92488:"Public Health",
    1207:"Tobacco Control"
    }

profile_category_dic = {
    86:"Health Protection",
    87:"Health Protection",
    92:"Health Protection",
    125:"Health Protection",
    100:"Health Protection",
    101:"Public Health",
    55:"Drugs and Alcohol",
    58:"Education", 
    65: "Healthcare" , 
    76:"Health Improvement",
    130:"Health Protection",    
    135:"Health Protection",    
    139:"Health Protection",    
    141:"Health Protection",    
    143:"Health Protection",    
    146:"Health Protection",    
    45:"Sexual Health",
    55:"Drugs and Alcohol",
    79:"Drugs and Alcohol",
    98:"Mental Health",
    102:"Mental Health",
    105:"Mental Health",
    106:"Mental Health",
    133:"Mental Health",
    91:"Mental Health",
    84:"Mental Health",
    41:"Mental Health",
    40:"Mental Health",
    37:"Mental Health",
    36:"Mental Health",
    99:"Public Health",
    95:"Public Health",
    32:"Public Health",
    30:"Health Protection",
    18:"Tobacco Control"
    }
# #37
# # 1) import metadata for spot profiles AND add 5 national level indicators
metadata = pd.DataFrame([], columns=['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity'])
for i in [18,30,32,36,40,41,45,55,58,65,76,79,84,86,87,91,92,95,99,98,100,101,102,105,106,125,130,133,135,139,141,143,146,155]: 
    metad = ftp.get_metadata_for_profile_as_dataframe(i)
    metad = metad.loc[~metad.Indicator.str.contains("Spend"),]
    metad = metad[metadata.columns[0:-1]]
    metad['profile_id'] = i
    metadata = pd.concat([metadata, metad])

add_national = ftp.get_metadata_for_indicator_as_dataframe(
        list(national_indicators_dic.values()), 
        is_test=False)[['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity']]
add_national['profile_id'] = 155
metadata = pd.concat([metadata, add_national])


# 2) based on metadata from 1) we pull each indicator, 
# append all indicators and standardises metadata.

alldatayears = []
for x in [18,30,32,36,40,41,45,55,58,65,76,79,84,86,87,91,92,95,99,98,100,101,102,105,106,125,130,133,135,139,141,143,146,155]: 
#for x in [135,139,141,143,146,155]: 
    metad = ftp.get_metadata_for_profile_as_dataframe(x)
    metad = metad.loc[~metad.Indicator.str.contains("Spend"),]
    metad = metad[['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity']]
    metad['profile_id'] = x

    if(x == 155): 
        add_national = ftp.get_metadata_for_indicator_as_dataframe(
            list(national_indicators_dic.values()), 
            is_test=False)[['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity']]
        add_national['profile_id'] = 155
        metad = metad.append(add_national)

    for index, row in metad.iterrows(): 
        i = row['Indicator ID']    
        j = row['profile_id'] 
        byindicator = []
        
        # 1st retrieve data for all area types available in England 
        
        df=ftp.get_all_data_for_indicators(i,area_type_id="All")
        
        # Filter for most aggregate sex group, or the most specific one 
        if (len(df['Sex'].unique()) > 1):
            if(df.Sex.str.contains('Persons').any()):
                df = df.loc[(df['Sex'] == "Persons"),]
            else:
                df = df.loc[(df['Sex'] == "Female"),]
    
        # subset to categories of interest and national level
        dfeng = df.copy()
        dfeng = dfeng.loc[dfeng['Category Type'].isna() & dfeng['Category'].isna(),] #& dfeng['Parent Code'].notnull(),]
        dfeng = dfeng.loc[(dfeng['Area Code'] == "E92000001"),]
        dfeng['group'] = dfeng['Category'].fillna('national')
        dfeng['subgroup'] = dfeng['Category Type'].fillna('national')    
    
        # append national level
        byindicator.append(dfeng)
    
        # subset all data to select deprivations groups
        df = df.loc[~(df['Category Type'].isna()),]
        df = df.loc[df['Parent Code'].isna()]
    
        # correct exception for indicator 91195 which is duplicated
        if i == 91195:
            df = df.loc[~df.Value.isna(),]
    
        # append deprivation data and concatenate all data
        byindicator.append(df)
        df = pd.concat(byindicator)
    
        # we add useful metadata columns
        df['profile_id'] = j 
        # df['unit'] = row['Unit'] 
        df['value_type'] = row['Value type']
        df['frequency'] = row['Frequency']
        df['year_type'] = row['Year type']
        df['polarity'] = row['Polarity']
        df['seriesCode'] = row['Indicator ID']  
    
        # creates a unique ID based on indicator and area code
        df['seriesName'] = df['Indicator Name']
        df['group'] = df['Category'].fillna('national')
        df['subgroup'] = df['Category Type'].fillna('national')    
        df = df.drop_duplicates()
        assert df.duplicated(subset =  ['seriesCode','Area Code','Age','group','subgroup', \
                        'Time period Sortable','Time period range']).any() == False
    
        df = df[['seriesCode','seriesName','Age','Sex','group','subgroup', \
                 'Value','value_type','polarity', 'Count', 'Denominator' , \
                'frequency','year_type','Time period Sortable','Time period range', \
                'Indicator ID','Area Code','profile_id']]
    
        alldatayears.append(df)

# get the indicator/years present on the API to exclude from comparators downloaded
nat_series_year = pd.concat(alldatayears)
nat_series_year = nat_series_year.loc[(nat_series_year.group=='national'),['seriesCode','Time period Sortable']].drop_duplicates()
nat_series_year['TimeperiodSortable'] = nat_series_year['Time period Sortable'] / 10000

## adding national data not available on the API but downloaded from webpage which compares to England
comparators = pd.read_csv(home + '/data/england/deprivation/clean_data/comparators/appended.csv', low_memory=False).drop_duplicates()
comparators['seriesCode'] = comparators['indicator_id']
comparators['Indicator ID'] = comparators['indicator_id']
# exclude those indicator/years in API
comparators = comparators.merge(nat_series_year, how='left', validate='many_to_one', on = ['seriesCode', 'TimeperiodSortable'], indicator = True)
comparators = comparators.loc[~(comparators._merge == "both"),]

comparators = comparators.merge(metadata, how='left', validate='many_to_many', on = 'Indicator ID')
comparators['Area Code'] = 'E92000001'
comparators['Age'] = 'All ages'
# finetuning indicators defined for specific populations
comparators.loc[comparators.seriesName.str.contains('Under 75'),'Age'] = "<75 yrs"
comparators.loc[comparators.seriesName.str.contains('Under 18'),'Age'] = "<18 yrs"
comparators.loc[comparators.seriesName.str.contains('Children in care'),'Age'] = "<18 yrs"
comparators.loc[comparators.seriesName.str.contains('Under 16'),'Age'] = "<16 yrs"
comparators.loc[(comparators.seriesCode.isin([90739,90740])),'Age'] = "15-44 yrs"
comparators.loc[(comparators.seriesCode.isin([22401,41401])),'Age'] = "65+ yrs"
comparators.loc[(comparators.seriesCode.isin([92447])),'Age'] = "18+ yrs"
comparators.loc[(comparators.seriesCode.isin([20602])),'Age'] = "10-11 yrs"
comparators.loc[(comparators.seriesCode.isin([10401])),'Age'] = "10-17 yrs"
comparators.loc[(comparators.seriesCode.isin([90811])),'Age'] = "12-17 yrs"
comparators.loc[(comparators.seriesCode.isin([90741,90777])),'Age'] = "15-24 yrs"
comparators.loc[(comparators.seriesCode.isin([90790])),'Age'] = "15-59 yrs"
comparators.loc[(comparators.seriesCode.isin([10601])),'Age'] = "18-64 yrs"
comparators.loc[(comparators.seriesCode.isin([10602])),'Age'] = "18-69 yrs"
comparators.loc[(comparators.seriesCode.isin([22002])),'Age'] = "25-64 yrs"
comparators.loc[(comparators.seriesCode.isin([20601])),'Age'] = "4-5 yrs"
comparators.loc[(comparators.seriesCode.isin([91099,91100,91101])),'Age'] = "40-74 yrs"
comparators.loc[(comparators.seriesCode.isin([90246,90631,90632])),'Age'] = "5 yrs"

comparators.loc[(comparators.seriesCode.isin([90283,92445])),'Age'] = "18-64 yrs"
comparators.loc[(comparators.seriesCode.isin([92313])),'Age'] = "16-64 yrs"
# drop 2 duplicated observations
#comparators = comparators.loc[~comparators.seriesCode.isin([90731,91456]),]
comparators['Sex'] = 'Persons'
comparators.loc[(comparators.seriesCode.isin([90739,90740,90731,91456,20201])),'Sex'] = "Female"
comparators['Time period Sortable'] = comparators['TimeperiodSortable'] * 10000
comparators['Time period range'] = '1y'
comparators['unit'] = comparators['Unit']
comparators['value_type'] = comparators['Value type']
comparators['frequency'] = comparators['Frequency']
comparators['year_type'] = comparators['Year type']
comparators['polarity'] = comparators['Polarity']
comparators['group'] = 'national'
comparators['subgroup'] = 'national'
comparators['profile_id'] = np.nan
comparators['Count'] = np.nan
comparators['Denominator'] = np.nan
comparators['is_comparator'] = 1
comparators = comparators[['seriesCode','seriesName','Age','Sex','group','subgroup', \
            'Value', 'unit','value_type','polarity', 'Count', 'Denominator' , \
           'frequency','year_type','Time period Sortable','Time period range', \
           'Indicator ID','Area Code','profile_id','flag_financial','is_comparator']]


# Appending all data together
alldatayears.append(comparators)
alldatayears = pd.concat(alldatayears)

# eliminate conflicting copies by unifying counts and denominators
profiles = alldatayears[['seriesCode','profile_id']].drop_duplicates()
profiles['category'] = profiles['profile_id'].map(profile_category_dic)
profiles = profiles.loc[~(profiles.seriesCode.isin([212,219,273,1730,20602,91872, 93014,92443,93088,93015, 93553]) & profiles.category.isin(['Public Health','Health Improvement','Health Protection','Healthcare'])),]
profiles = profiles.loc[~profiles.category.isna(),]
profiles = profiles[['seriesCode','category']].drop_duplicates()
profiles['n'] = profiles.groupby(['seriesCode']).cumcount().add(1)
profiles['n'] = 'category' + profiles['n'].astype(str)
profiles = profiles.pivot(index='seriesCode', values = 'category', columns  = 'n').reset_index()
categories = categories.append(profiles)
categories = categories.drop_duplicates(subset = 'seriesCode', keep = 'first')
categories['Indicator ID']=categories.seriesCode
categories = categories.drop('seriesCode', axis = 'columns').reset_index()
alldatayears = alldatayears.drop('profile_id', axis = 'columns')

# eliminate duplicates for polarity
polarity = alldatayears[['seriesCode','polarity']].drop_duplicates()
polarity['n'] = polarity.groupby(['seriesCode']).cumcount().add(1) 
polarity = polarity.pivot(index='seriesCode', values = 'polarity', columns  = 'n').reset_index().rename(columns = {1:'polarity1',2:'polarity2'})
alldatayears = alldatayears.drop('polarity', axis = 'columns')

alldatayears['Count'] = alldatayears.groupby(['seriesCode'])['Count'].transform('max')
alldatayears['Denominator'] = alldatayears.groupby(['seriesCode'])['Denominator'].transform('max')
alldatayears['flag_financial'] = alldatayears.groupby(['seriesCode'])['flag_financial'].transform('max')
alldatayears['is_comparator'] = alldatayears['is_comparator'].fillna(0)

alldatayears = alldatayears.drop_duplicates()

# drop some duplicated indicators:
alldatayears = alldatayears.loc[~(alldatayears.seriesName == "Under 75 mortality rate from all cardiovascular diseases"),]
alldatayears = alldatayears.loc[~(alldatayears.seriesName == "Under 75 mortality rate from cancer"),]
alldatayears = alldatayears.loc[~(alldatayears.seriesName == "Under 75 mortality rate from liver disease"),]
alldatayears = alldatayears.loc[~(alldatayears.seriesName == "Under 75 mortality rate from respiratory disease"),]

# for the analysis of multiple deprivations we select and deduplicate to obtain 10 decile groups per indicator
# We select indicators stratified by levels of deprivation
deprivation_columns = ["County & UA deprivation deciles in England (IMD2015, pre 4/19 geog.)",  \
    "County & UA deprivation deciles in England (IMD2019, 4/19 and 4/20 geog.)",  \
    "County & UA deprivation deciles in England (IMD2019, 4/21 geography)",  \
    "LSOA11 deprivation deciles in England (IMD2015)", \
    "LSOA11 deprivation deciles in England (IMD2019)", \
    "LSOA11 deprivation deciles within area (IDACI)",  \
    "LSOA11 deprivation deciles within area (IMD  trend)", "national"]

alldatayears = alldatayears.loc[(alldatayears.subgroup.isin(deprivation_columns)),]

# exclude missing Value
alldatayears = alldatayears.loc[~(alldatayears.Value.isna()),]

# then we sort data and identify duplicates
alldatayears = alldatayears.sort_values(['seriesCode','subgroup','group','Age'])
alldatayears['is_dup'] = alldatayears[['seriesCode','subgroup','group','Age']].duplicated(keep=False).astype(int)

# we drop cases with wider timeperiod (3years) that have a duplicate of 1year
alldatayears = alldatayears.loc[~((alldatayears.is_dup == 1) & (alldatayears['Time period range']=="3y")),]

# Prefer IMD based on county statistics as they also allocate budget
# but would only drop other indicators if duplicated with county
alldatayears = alldatayears.loc[~(alldatayears.subgroup.str.contains('LSOA') &  \
    ~alldatayears['Indicator ID'].isin([20601, 20602, 90319, 90801, 93078, 93079, 93195, 93580])),]
alldatayears = alldatayears.loc[~(alldatayears.Age == 'All ages') &  \
    ~alldatayears['Indicator ID'].isin([92489, 92490]),]
alldatayears = alldatayears.loc[~(alldatayears.Age == '45+ yrs') &  \
    ~alldatayears['Indicator ID'].isin([91262]),]

# to identify duplicates by Age column and select the broader age groups.
alldatayears['is_dup_age1'] = alldatayears[['seriesCode','subgroup','group', 'Time period Sortable']].duplicated(keep=False).astype(int)
alldatayears['is_dup_age2'] = alldatayears[['seriesCode','subgroup','group','Age','Time period Sortable']].duplicated(keep=False).astype(int)
alldatayears['is_dup_age'] = np.where(alldatayears.is_dup_age1 != alldatayears.is_dup_age2, 1,0)

# Column to identify the most comprehensive age group (i.e. All ages or other)
alldatayears['large_agegroup'] = np.where(((alldatayears.is_dup_age == 1) & 
                     (alldatayears.Age == "All ages")), 1, 0)

# a few exceptions by indicator, as 'All ages' is not always the best option
alldatayears.loc[(((alldatayears.is_dup_age == 1) & \
                                       (alldatayears.Age == "18+ yrs")) & \
                                    (alldatayears.seriesCode == 90638)), 'large_agegroup'] = 1
alldatayears.loc[(((alldatayears.is_dup_age == 1) & \
                                       (alldatayears.Age == "All ages")) & \
                                    (alldatayears.seriesCode == 90638)), 'large_agegroup'] = 0
alldatayears.loc[(((alldatayears.is_dup_age == 1) & \
                                       (alldatayears.Age == "16-64 yrs")) & \
                                    (alldatayears.seriesCode == 92313)), 'large_agegroup'] = 1
alldatayears.loc[(((alldatayears.is_dup_age == 1) & (alldatayears.Age == "18-64 yrs")) & \
                                    (alldatayears.seriesCode.isin([90283,92445]))), 'large_agegroup'] = 1

    
# identify those indicators with the largest age group defined already
alldatayears['maxageg'] = alldatayears.groupby(['seriesCode'])['large_agegroup'].transform('max') 

# and a loop retrieves the most comprehensive age group for the remaining of indicators
for i in ["10", "12" ,"16", "17" ,"18", "19", "30", "35", "40", "50", "60","65", "80", "85", "90"]:
    alldatayears.loc[((alldatayears.is_dup_age == 1) & \
                     (alldatayears.Age.str.contains(i+"\+")) & \
                     (alldatayears.maxageg == 0) & (alldatayears.large_agegroup == 0)), 'large_agegroup'] = 1 
    alldatayears['temp_col'] = alldatayears.groupby(['seriesCode'])['large_agegroup'].transform('max') 
    alldatayears.loc[(alldatayears.is_dup_age == 1) & \
                     (alldatayears.temp_col == 1) & \
                     (alldatayears.maxageg == 0), 'maxageg'] = 1 


# First exclusion : drop least comprehensive age groups per indicator 
alldatayears['drop_agedup'] = np.where(((alldatayears.large_agegroup == 1) & \
                                       (alldatayears.is_dup_age == 1)) | \
                                       (alldatayears.is_dup_age == 0) | \
                                       (alldatayears.large_agegroup.isna()), 0, 1)

alldatayears = alldatayears.loc[~(alldatayears.drop_agedup == 1),]

# drop additional duplicates for indicators 90638 & 1211 at national level
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 90638) & \
                                (alldatayears.Age == "65+ yrs") & \
                                (alldatayears.is_dup == 1)),]
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 90638) & \
                                alldatayears.Age.isin(["All ages","18-64 yrs"]) & \
                                (alldatayears.is_dup_age == 1)),]
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 1211) & \
                                (alldatayears.Age != "16+ yrs") & \
                                (alldatayears.is_dup_age == 1)),]
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 358) & \
                                (alldatayears.Age != "16+ yrs")),]
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 93183) & \
                                (alldatayears.Age != "18-64 yrs")),]
alldatayears = alldatayears.loc[~((alldatayears.seriesCode.isin([40401,40501,40701])) & \
                                (alldatayears.value_type == "Directly standardised rate")),]

# assert duplicates are no longer due to age group
alldatayears['is_dup_indicator'] = alldatayears[['seriesCode','group','Time period Sortable']].duplicated(keep = False).astype(int)
alldatayears['is_dup_age2'] = alldatayears[['seriesCode','Age','group','Time period Sortable']].duplicated(keep = False).astype(int)
assert alldatayears['is_dup_indicator'].equals(alldatayears['is_dup_age2'])


# standardise group column
alldatayears['group'] = alldatayears['group'].str.replace(" \(IMD2015\)","")
alldatayears['group'] = alldatayears['group'].str.replace(" \(IMD2019\)","")

# we expect 10 rows per indicator but some have difference due to changes in 2019 and 2020 county areas
# I assume ranking of IMD is sufficiently similar and standardized both measures.
# a column to identify duplicates 
alldatayears = alldatayears.sort_values(by = ['seriesCode','Age','subgroup','group','Time period Sortable'])
alldatayears['is_dup'] = alldatayears[['seriesCode','group','Time period Sortable']].duplicated(keep = False).astype(int)
alldatayears.loc[alldatayears.subgroup == "national", 'is_dup'] = 0

# a column to identify the lenght of each IMD within indicator, and identify this longest serie at indicator level
alldatayears['count_dup_indicator1'] = alldatayears.groupby(['seriesCode','Age','subgroup','group']).cumcount().add(1) 
alldatayears['longer'] = alldatayears.groupby(['seriesCode','Age','subgroup','group'])['count_dup_indicator1'].transform('max')
alldatayears['longest'] = alldatayears.groupby(['seriesCode','Age'])['count_dup_indicator1'].transform('max')
alldatayears['is_longest'] = np.where(alldatayears.longest == alldatayears.longer, 1, 0)

# correcting for this indicator, which has missing 2020 data for 3 deciles which makes it less convenient than 15 contigous years
alldatayears.loc[(alldatayears.seriesCode == 20602), 'longest'] = 15
alldatayears.loc[((alldatayears.seriesCode == 20602) & alldatayears.subgroup.str.contains('LSOA11')), 'is_longest'] = 1
alldatayears.loc[((alldatayears.seriesCode == 20602) & alldatayears.subgroup.str.contains('County')), 'is_longest'] = 0

#identify the latest series, in case there are duplicates in longest so we select the latest
latest_order = [2,1,0,4,3,5,6,7]
alldatayears['latest_subgroup'] = alldatayears.subgroup.map(dict(zip(deprivation_columns, latest_order)))
alldatayears['latest'] = alldatayears.groupby(['seriesCode','Age'])['latest_subgroup'].transform('min')
alldatayears['is_latest'] = np.where(alldatayears.latest == alldatayears.latest_subgroup,1,0)

# mark exception for seriescode 91101 after deduplicating for age 
alldatayears.loc[((alldatayears.seriesCode == 91101) & \
                 (alldatayears.subgroup == "County & UA deprivation deciles in England (IMD2019, 4/21 geography)") & \
                 (alldatayears['Time period Sortable'] == 2017)), 'is_dup'] = 0 

# 2nd exclusion rule: drop duplicates which are not the longest in the series
alldatayears = alldatayears.loc[~((alldatayears.is_longest == 0) & \
                                (alldatayears.is_dup == 1)),]

# 2nd exclusion rule: drop duplicates which come from comparators in the series
alldatayears = alldatayears.loc[~((alldatayears.is_comparator == 1) & \
                                (alldatayears.is_dup_indicator == 1)),]

# exception: drop 30314 which are longest but not the latest in the series
alldatayears = alldatayears.loc[~((alldatayears.seriesCode == 30314) & \
                    (alldatayears.is_latest == 0) & (alldatayears.is_longest == 1) & \
                    (alldatayears.is_dup == 1)),] 

# identify duplicates at this point
alldatayears = alldatayears.sort_values(['seriesCode','Age','subgroup','group','Time period Sortable'])
alldatayears['is_dup2'] = alldatayears[['seriesCode','Time period Sortable','group']].duplicated(keep = False).astype(int)
alldatayears.loc[alldatayears.subgroup == "national", 'is_dup2'] = 0

# 3rd exclusion rule: drop the oldest (not latest) series when 2 or more series are the longest
alldatayears = alldatayears.loc[~((alldatayears.is_latest == 0) & \
                                (alldatayears.is_longest == 1) & \
                                (alldatayears.is_dup2 == 1)),] 

#check for no dups
alldatayears.duplicated(subset =  ['seriesCode','group','Time period Sortable']).unique()
assert alldatayears.duplicated(subset =  ['seriesCode','group','Time period Sortable']).unique().all() == False


# after standardising, we create relevant columns. 
# polarity of the following indicators is stated as 'Not applicable' yet 
# they should be inverted as their increase reflects a loss of wellbeing
toinvert = ["A&E attendances (0-4 years) (previous method)",
"Statutory homelessness - households in temporary accommodation", # not in data
'Fraction of mortality attributable to particulate air pollution (old method)',
'Number in treatment at specialist alcohol misuse services',
'Re-offending levels - average number of re-offences per re-offender',
'Domestic abuse related incidents and crimes',
'Violent crime - violence offences per 1,000 population',
'Violent crime - sexual offences per 1,000 population',
'Re-offending levels - percentage of offenders who re-offend',
'First time offenders',
'Adults in treatment at specialist alcohol misuse services: rate per 1000 population',
'Under 18s conceptions leading to abortion (%)',
'Under 18s abortions rate / 1,000',
'Adults in treatment at specialist drug misuse services: rate per 1000 population',
'Diabetes: QOF prevalence (17+ yrs)',
'CKD: QOF prevalence (18+ yrs)',
'Depression: QOF prevalence (18+ yrs)',
'Osteoporosis: QOF prevalence (50+ yrs)',
'Antidepressant prescribing: average daily quantities (ADQs) per STAR-PU',
'Hypnotics prescribing: average daily quantities (ADQs) per STAR-PU',
'Depression and anxiety among social care users: % of social care users',
'Depression: QOF incidence (18+ yrs) - new diagnosis',
'Number of people receiving RRT',
'The proportion of patients receiving home dialysis (Home HD and PD combined)',
'The percentage of all people receiving RRT on the different modality types: Kidney transplant',
'Number in treatment at specialist drug misuse services',
'Patients (75+ yrs) with a fragility fracture treated with a bone-sparing agent (den. incl. exc.) - retired after 2018/19',
'Rheumatoid Arthritis: QOF prevalence (16+ yrs)',
'The percentage of all people receiving RRT on the different modality types: Home dialysis',
'The percentage of all people receiving RRT on the different modality types: Hospital dialysis',
'Personalised Care Adjustment (PCA) rate for depression indicator',
'C. difficile all rates by reporting acute trust and financial year',
'Smoking prevalence in adults (18+) - ex smoker (GPPS)',
'Reception: Prevalence of overweight',
'Year 6: Prevalence of overweight',
'Smoking Prevalence in adults (18+) - ex smokers (APS)',
'Obesity: QOF prevalence (18+ yrs)',
'C. difficile infection community-onset counts and rates, by CCG and financial year',
'Percentage of deaths that occur in hospital',
'Percentage of deaths that occur in care homes',
"Percentage of deaths that occur in 'other places'",
'Percentage of deaths that occur in hospice',
'Percentage of homes fail the Decent Homes Standard (ENGLAND)',
'Percentage of homes fail the minimum housing standard (ENGLAND)',
'People aged 65-74 registered blind or partially sighted',
'People aged 75+ registered blind or partially sighted',
'Mixed anxiety and depressive disorder: estimated % of population aged 16-74',
'Generalised anxiety disorder: estimated % of population aged 16-74',
'Depressive episode: estimated % of population aged 16-74',
'Children on child protection plans: Rate per 10,000 children <18',
"Alzheimer's disease: Direct standardised rate of inpatient admissions (aged 65 years and over)",
'Vascular dementia:  Direct standardised rate of inpatient admissions (aged 65 years and over)',
'Concurrent contact with mental health services and substance misuse services for drug misuse',
'Concurrent contact with mental health services and substance misuse services for alcohol misuse',
'Suicide crude rate 10-34 years: per 100,000 (5 year average)',
'Suicide crude rate 35-64 years: per 100,000 (5 year average)',
'Suicide crude rate 65+ years: per 100,000  (5 year average)',
'Stroke admissions with history of atrial fibrillation not prescribed anticoagulation prior to stroke',
'Stroke admissions (Sentinel Stroke National Audit Programme)',
'Stroke patients who are assessed at 6 months',
'Unspecified dementia: Direct standardised rate of inpatient admissions (aged 65 years and over)',
'Dementia: Recorded prevalence (aged 65 years and over)',
'Place of death - care home: People with dementia (aged 65 years and over)',
'Place of death - hospital: People with dementia (aged 65 years and over)',
'Sole registered births: % births registered by one parent only',
'Mental health detection at antenatal booking: % valid completion',
'Substance use recorded at antenatal booking: % valid completion',
'Support status recorded at antenatal booking: % valid completion',
'Alcohol consumption recorded at antenatal booking: % valid completion',
'Complex social factors recorded at antenatal booking: % valid completion',
'Complex social factors: % of pregnant women',
'Demand for Debt Advice: rate per 10,000 adults',
'Percentage of pedestrians killed or seriously injured in road traffic accidents taking place on a 30mph road (aged 0-24)',
'Percentage of pedal cyclists killed or seriously injured in road traffic accidents taking place on a 30mph road (aged 0-24)',
'Percentage of motorcyclists killed or seriously injured in road traffic accidents taking place on a 30mph road (aged 0-24)',
'Percentage of car occupants killed or seriously injured in road traffic accidents taking place on a 30mph road (aged 0-24)',
'Rate of newly diagnosed dementia registrations (Experimental)',
'Dementia (aged under 65 years) as a proportion of total dementia (all ages) per 100',
'Individuals with learning disabilities involved in Section 42 safeguarding enquiries (per 1,000 people on the GP learning disability register)',
'Percentage of people with type 1 diabetes aged under 40',
'Percentage of people with type 1 diabetes aged 40 to 64',
'Percentage of people with type 1 diabetes aged 65 to 79',
'Percentage of people with type 1 diabetes aged 80 and over',
'Percentage of people with type 2 diabetes aged 80 and over',
'Percentage of people with type 1 diabetes who are female',
'Percentage of people with type 1 diabetes who are white',
'Percentage of people with type 1 diabetes who are of minority ethnic origin',
"Alzheimer's disease: Direct standardised rate of inpatient admissions (aged 65 years and over) - CCG responsibility",
'Vascular dementia:  Direct standardised rate of inpatient admissions (aged 65 years and over) - CCG responsibility',
'Unspecified dementia: Direct standardised rate of inpatient admissions (aged 65 years and over) - CCG responsibility',
'Hospital admissions for dental caries (0 to 5 years) - CCG',
'C. difficile infection Community-Onset Healthcare Associated (COHA) counts and rates, by CCG and financial year',
'C. difficile infection Community-Onset Community Associated (COCA) counts and rates, by CCG and financial year',
'C. difficile infection Hospital-Onset Healthcare Associated (HOHA) counts and rates, by acute trust and financial year',
'C. difficile infection community-Onset Healthcare Associated (COHA) counts and rates, by acute trust and financial year',
'Attended contacts with community and outpatient mental health services, per 100,000',
'New referrals to secondary mental health services, per 100,000',
'Odds ratio of reporting a mental health condition among people with and without an MSK condition',
'Fraction of mortality attributable to particulate air pollution (new method)'
       ] 

# we generate the invert column based on the polarity column from metadata
alldatayears['invert'] = 0
alldatayears = alldatayears.merge(polarity, on = ['seriesCode'], validate = 'many_to_one', how = 'left')
alldatayears.loc[(alldatayears.polarity1 == "RAG - Low is good   ") |
                 (alldatayears.polarity1.str.contains("Low")) |
                 (alldatayears.polarity2.str.contains("Low")) |
                 (alldatayears.seriesName.isin(toinvert)),'invert'] = 1

alldatayears['toinvert'] = 0
alldatayears.loc[(alldatayears.seriesName.isin(toinvert)),'toinvert'] = 1

# retrieve best and worst bounds based on data
max_value = alldatayears.groupby(['Indicator ID'], as_index = False, observed = True)['Value'].max().rename(columns = {'Value':'bestBound'})
min_value = alldatayears.groupby(['Indicator ID'], as_index = False, observed = True)['Value'].min().rename(columns = {'Value':'worstBound'})

# retrieve first and last value of time series by deduplicating on key columns
start_value = alldatayears.loc[alldatayears.Value.notnull(),['seriesCode','seriesName','group','Time period Sortable','Value']]
start_value = start_value.loc[~start_value.Value.isna(),]
start_value = start_value.sort_values(['seriesCode','seriesName','group','Time period Sortable'])
last_value = start_value.copy()
start_value = start_value.drop_duplicates(subset = ['seriesCode','seriesName','group'], \
                keep = 'first').rename(columns = {'Value':'start_value'}).drop('Time period Sortable', axis = 1)
last_value = last_value.drop_duplicates(subset = ['seriesCode','seriesName','group'], \
                keep = 'last').rename(columns = {'Value':'end_value'}).drop('Time period Sortable', axis = 1)

# merge these columns onto the main database
alldatayears = alldatayears.merge(max_value, on = 'Indicator ID', validate = 'many_to_one', how = 'outer')
alldatayears = alldatayears.merge(min_value, on = 'Indicator ID', validate = 'many_to_one', how = 'outer')
alldatayears = alldatayears.merge(start_value, on = ['seriesCode','seriesName','group'], validate = 'many_to_one', how = 'outer')
alldatayears = alldatayears.merge(last_value, on = ['seriesCode','seriesName','group'], validate = 'many_to_one', how = 'outer')


# create theoretical bounds based on empirical and adjust proportions which bound to (0,100) 
alldatayears['worstTheoretical'] = alldatayears['worstBound']
alldatayears['bestTheoretical'] = alldatayears['bestBound']
alldatayears.loc[alldatayears.value_type.isin(["Percentage point", "Proportion"]),'worstTheoretical'] = 0
alldatayears.loc[alldatayears.value_type.isin(["Percentage point", "Proportion"]),'bestTheoretical'] = 100
alldatayears.loc[alldatayears.seriesCode.isin([90360,90361,90642,90641,91195]),'worstTheoretical'] = 0 # 
alldatayears.loc[alldatayears.seriesCode.isin([90360,90361,90642,90641,91195]),'bestTheoretical'] = 100

alldatayears.loc[alldatayears.seriesCode.isin([90360,90361,90642,90641]),'worstTheoretical'] = 0 #  indices
alldatayears.loc[alldatayears.seriesCode.isin([90360,90361,90642,90641]),'bestTheoretical'] = 100 # indices

# Population vaccination coverage, screening, which would bound (0,100)
alldatayears.loc[alldatayears.seriesName.str.contains('screening','coverage'),'worstTheoretical'] = 0 
alldatayears.loc[alldatayears.seriesName.str.contains('screening','coverage'),'bestTheoretical'] = 100

# Preventable mortality which would bound (1 to denominator)
alldatayears.loc[alldatayears.seriesName.str.contains('preventable') & 
                 (alldatayears.seriesCode != 93721),'worstTheoretical'] = 0 
#alldatayears['alt_best'] = alldatayears['Count'] / alldatayears['Denominator'] *100000
alldatayears.loc[alldatayears.seriesName.str.contains('preventable') & 
                 (alldatayears.seriesCode != 93721),'bestTheoretical'] = 100 #alldatayears['bestNat'] 

# theoretical life expectancy to bound to historical data
alldatayears.loc[alldatayears.seriesCode.isin([90366]),'worstTheoretical'] = 22 # life expectancy
alldatayears.loc[alldatayears.seriesCode.isin([90366]),'bestTheoretical'] = 85.3

                                
# assume all indicators are instrumental 
alldatayears['instrumental'] = 1
#non_instrumental = []
#alldatayears.loc[(alldatayears.seriesName.isin(non_instrumental)),'instrumental'] = 0


# create a flag for frequency
alldatayears['flag_non_annual'] = 1
alldatayears.loc[alldatayears.frequency.str.contains("nnual") == True,
                 'flag_non_annual'] = 0
alldatayears.loc[alldatayears['Time period range'] == '1y',
                 'flag_non_annual'] = 0
alldatayears.loc[alldatayears.year_type == 'Financial',
                 'flag_non_annual'] = 0


alldatayears = alldatayears.loc[alldatayears['Time period Sortable'].isin([20130000, 20140000, 20150000, 20160000,
       20170000, 20180000, 20190000, 20200000, 20210000, 20220000]),]
# create a column that identifies gaps in titmeseries
alldatayears['gap'] = alldatayears.groupby(['seriesCode','group'])['Time period Sortable'].diff()
alldatayears['gap'] = alldatayears['gap'] / 10000
alldatayears['gap_inseries'] = (alldatayears.groupby(['seriesCode','group'])['gap'].transform('max') -1)

# seriesCode recoded for different deciles and national
batch_dict = {'national':0,"Fifth less deprived decile":6,"Fifth more deprived decile":5,
              "Fourth less deprived decile":7,"Fourth more deprived decile":4,
              "Least deprived decile":10,"Most deprived decile":1,
              "Second least deprived decile":9,"Second most deprived decile":2,
              "Third less deprived decile":8,"Third more deprived decile":3}

alldatayears['seriesCode'] = alldatayears['seriesCode'].astype(str) + "_" + alldatayears.group.map(batch_dict).astype(str)

alldatayears = alldatayears[['seriesCode','seriesName','Age','Sex','group', 'Area Code', 'Indicator ID','invert', \
             'bestBound','worstBound', 'Count','Denominator','start_value','end_value', \
            'bestTheoretical','worstTheoretical', \
            'value_type','polarity1', 'polarity2','unit','instrumental','flag_non_annual','gap_inseries','flag_financial', \
                'Time period Sortable','Value']].reset_index()
alldatayears = alldatayears.sort_values(by = ['seriesCode','seriesName','Age','Sex','group','Time period Sortable']).reset_index()
assert alldatayears.duplicated().any()==False

data_wide = pd.pivot(alldatayears, index = ['seriesCode','seriesName','Age','Sex','group', \
            'Area Code', 'Indicator ID','invert','bestBound','worstBound', 'Count','Denominator','start_value','end_value', \
            'bestTheoretical','worstTheoretical', \
            'value_type','polarity1','polarity2','unit','instrumental','flag_non_annual','gap_inseries','flag_financial'], \
                        columns = 'Time period Sortable',
                        values = 'Value').reset_index()

years = [column_name for column_name in data_wide.columns if str(column_name).isnumeric()]
data_wide['count_valid'] = data_wide[years].notnull().sum(axis = 'columns')

# add columns that identify the categories of expenditure (Health profiles)
data_wide = data_wide.merge(categories, on = 'Indicator ID', validate = 'many_to_one', how = 'left')
data_wide.loc[data_wide['category1'].isna(),['category1']] = data_wide.seriesCode.map(national_category_dic)
#data_wide = data_wide.merge(profiles, on = ['seriesCode'], validate = 'many_to_one', how = 'left')
#data_wide.loc[data_wide['category1'].isna(),['category1']] = data_wide.profile1.map(profile_category_dic)
#data_wide.loc[data_wide['category2'].isna(),['category2']] = data_wide.profile2.map(profile_category_dic)
#data_wide.loc[data_wide['category3'].isna(),['category3']] = data_wide.profile2.map(profile_category_dic)

# export raw deduplicated data
data_wide.to_csv(home + '/data/england/deprivation/clean_data/data_indicators_raw.csv', index=False)

