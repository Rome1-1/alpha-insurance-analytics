import pandas as pd
from data_cleaning import clean_missing_values, fix_data_types
from outlier_detection import detect_outliers
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with fixes for dtype issues
file_path = 'C:\\Users\\teble\\alpha-insurance-analytics\\my_project\\data\\insurance_data.csv'
df = pd.read_csv(file_path, low_memory=False, dtype={
    'Citizenship': str, 
    'NumberOfVehiclesInFleet': str, 
    'Converted': str
})

# 1. Data Cleaning
df = clean_missing_values(df)
df = fix_data_types(df)

# 2. Check missing values summary
print("Missing Values After Cleaning:")
print(df.isnull().sum())

# 3. Summary Statistics
print("Basic Summary Statistics:")
print(df.describe())

# 4. Outlier Detection
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

print("Boxplots for Numeric Columns:")
for col in numeric_cols:
    detect_outliers(df, col)  # Plot boxplots for numeric columns

# 5. Save cleaned data
cleaned_file_path = 'C:\\Users\\teble\\alpha-insurance-analytics\\my_project\\data\\insurance_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")