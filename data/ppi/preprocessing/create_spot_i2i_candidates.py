# Goal of this script is to create the candidates for the indicator to indicator matching
# The input file "pipeline_indicators_sample_raw.csv" contains rows where each row contains information about a socio-economic indicator 
# The input file is a .csv with the following header columnns: [seriesCode,seriesName,Age,Sex,group,Area Code,Indicator ID,invert,bestBound,worstBound,Count,Denominator,start_value,end_value,bestTheoretical,worstTheoretical,value_type,polarity1,polarity2,unit,instrumental,flag_non_annual,gap_inseries,flag_financial,2013,2014,2015,2016,2017,2018,2019,count_valid,index,category1,category2,category3]
# The output file "i2i_candidates.csv" contains rows where each row contains information about a pair of indicators that are candidates for matching
# The resulting file should have the following columns = [ indicator1_name, indicator1, seriesCode1, idx1 , seriesName1, Age1, Sex1, group1, indicator2_name, indicator2, seriesCode2, idx2 , seriesName2, Age2,Sex2, group2, indicator2 ]
# indicator1_name and indicator2_name are the original indicators that are being matched
# indicator1 is an intuitive concatenation of the strings in indicator1_name and Age1 

import csv
import os

# Function to create intuitive concatenation of indicator name and age
def concatenate_name_age(name, age):
    # If age is not in the indicator name, then concatenate the name and age
    if age in name:
        return name
    else:
        return f"{name} ({age})"



# Reading the input file
with open( os.path.join('data','ppi','pipeline_indicators_sample_raw.csv'), "r") as input_file:
    reader = csv.DictReader(input_file)
    indicators = list(reader)

candidates = []

# Comparing each indicator with every other indicator to create pairs
for i in range(len(indicators)):
    for j in range(len(indicators)):
        if i == j:
            continue
        indicator1 = indicators[i]
        indicator2 = indicators[j]
        
        candidate = {
            "indicator1_name": indicator1["seriesName"],
            "indicator1": concatenate_name_age(indicator1["seriesName"], indicator1["Age"]),
            "seriesCode1": indicator1["seriesCode"],
            "idx1": i,
            "seriesName1": indicator1["seriesName"],
            "Age1": indicator1["Age"],
            "Sex1": indicator1["Sex"],
            "group1": indicator1["group"],
            
            "indicator2_name": indicator2["seriesName"],
            "indicator2": concatenate_name_age(indicator2["seriesName"], indicator2["Age"]),
            "seriesCode2": indicator2["seriesCode"],
            "idx2": j,
            "seriesName2": indicator2["seriesName"],
            "Age2": indicator2["Age"],
            "Sex2": indicator2["Sex"],
            "group2": indicator2["group"]
        }
        
        candidates.append(candidate)

# Writing the output to i2i_candidates.csv
with open(os.path.join('data','ppi','preprocessing',"i2i_candidates.csv"), "w", newline="") as output_file:
    fieldnames = ["indicator1_name", "indicator1", "seriesCode1", "idx1", "seriesName1", "Age1", "Sex1", "group1",
                  "indicator2_name", "indicator2", "seriesCode2", "idx2", "seriesName2", "Age2", "Sex2", "group2"]
    
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(candidate)
