import os
import csv
import pandas as pd

def process_govt_budget_file():
    # open budget csv
    fp_budget = os.path.join('datasets','england','raw_datasets','budget_description.csv')
    df_budget = pd.read_csv(fp_budget, header=0)

    # rename/create columns
    dict_rename ={
        'description': 'name',
        'final_year': 'final_date',
        'profile': 'group_name',
        'code': 'code'
    }
    df_budget_processed = df_budget.rename( dict_rename, axis=1 )
    df_budget_processed = df_budget_processed.drop( [c for c in df_budget_processed.columns if c not in dict_rename.values() ], axis=1 )
    df_budget_processed['period'] = '1Y'

    # reformat columns
    print("Current dataset lists budget items 'last year'.\
        This script assumes the last date is 31-12 of the last year")

    df_budget_processed['final_date'] = df_budget_processed['final_date'].map( lambda year: '31-12-'+str(year) )
    df_budget_processed = df_budget_processed.replace('â€“','-', regex=False)
    df_budget_processed = df_budget_processed.replace('â€™',"'", regex=False)
    df_budget_processed['group_name'] = df_budget_processed['group_name'].str.capitalize()
    
    # Handling case where some rows are summations of previous rows
    df_budget_processed = process_total_rows(df_budget_processed)

    # Removing specific categories
    df_budget_processed = df_budget_processed[ ~ df_budget_processed.group_name.str.contains('Central services')]
    df_budget_processed = df_budget_processed[ ~ df_budget_processed.group_name.str.contains('Precepts and levies')]
  
    # save to file
    df_budget_processed.to_csv( os.path.join('datasets','england','processed_datasets','budget_items.csv'), index=False )

def process_total_rows(df_budget):
    """Handling case where some rows are summations of previous rows
        Two methods to find sums - if it starts with 'TOTAL' and has '(total of lines 210 to 280)'
        Use 'TOTAL' at start to filter lines and the presence of '(total of lines 210 to 280)'
            strip the two starting and ending indices e.g. 210 and 280
            For all budget entries with code between 210 and 280 then add this group budget info as group_code, group_name
            """
    # If 'total of lines' is present then remove
    df_budget = df_budget[ ~ df_budget['name'].str.contains( 'total of lines', regex=False) ]

    # If 'TOTAL' is present and text is all UPPER and convert text to captialize
    bool_update = df_budget['name'].str.isupper() & df_budget['name'].str.contains( 'TOTAL', regex=False)
    df_budget['name'][bool_update] = df_budget['name'][bool_update].str.capitalize()

    return df_budget    

    

def process_health_indicator_file():
    # open budget csv
    fp_health = os.path.join('datasets','england','raw_datasets','indicator_description.csv')
    df_health = pd.read_csv(fp_health, header=0)

    # rename columns
    dict_rename ={
        'indicator_id': 'code',
        'indicator_name': 'name',
        'final_year': 'final_date',
        'profile_id': 'group_code',
        'profile_name': 'group_name'
    }

    df_health_processed = df_health.rename( dict_rename, axis=1 )
    df_health_processed = df_health_processed.drop( [c for c in df_health_processed.columns if c not in dict_rename.values() ], axis=1 )
    # reformat columns
    print("Current dataset lists budget items 'last year'.\
        This script assumes the last date is 31-12 of the last year")

    df_health_processed['final_date'] = df_health_processed['final_date'].map( lambda year: '31-12-'+str(int(year)) )
    df_health_processed['period'] = '1Y'
    # save to file
    df_health_processed.to_csv(os.path.join('datasets','england', 'processed_datasets','health_indicators.csv'), index=False )

if __name__ == '__main__':
    process_govt_budget_file()
    process_health_indicator_file()