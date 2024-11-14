import pandas as pd

def count_indicators_per_budget():
    # Read the CSV file
    df = pd.read_csv('./spot_b2i_broad.csv')
    
    # Group by budget_item and count the number of indicators
    indicator_counts = df.groupby('budget_item').size().sort_values(ascending=False)
    
    # Print the results
    print("\nNumber of indicators per bupydget item:")
    print("=====================================")
    for budget_item, count in indicator_counts.items():
        print(f"{budget_item}: {count}")
    
    print(f"\nTotal number of budget items: {len(indicator_counts)}")
    print(f"Total number of indicators: {len(df)}")

if __name__ == "__main__":
    count_indicators_per_budget()