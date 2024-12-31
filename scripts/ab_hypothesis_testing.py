import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
data = pd.read_csv(r'C:\Users\teble\alpha-insurance-analytics\my_project\data\cleanedinsurance_data.csv')

# Calculate ProfitMargin
data['ProfitMargin'] = data['TotalPremium'] - data['TotalClaims']

# Null Hypotheses Definitions
null_hypotheses = {
    "province": "There are no risk differences across provinces",
    "zipcode": "There are no risk differences between zip codes",
    "margin": "There are no significant margin (profit) differences between zip codes",
    "gender": "There are no significant risk differences between Women and Men"
}

results = []

# 1. Risk differences across provinces (ANOVA)
province_model = stats.f_oneway(*[group['TotalClaims'].values for _, group in data.groupby('Province')])
results.append(("Risk Across Provinces", province_model.statistic, province_model.pvalue))

# 2. Risk differences between zip codes (ANOVA)
zipcode_model = stats.f_oneway(*[group['TotalClaims'].values for _, group in data.groupby('PostalCode')])
results.append(("Risk Between Zip Codes", zipcode_model.statistic, zipcode_model.pvalue))

# 3. Profit margin differences between zip codes (ANOVA)
margin_model = stats.f_oneway(*[group['ProfitMargin'].values for _, group in data.groupby('PostalCode')])
results.append(("Profit Margin Between Zip Codes", margin_model.statistic, margin_model.pvalue))

# 4. Risk differences by gender (t-test)
male_risk = data[data['Gender'] == 'Male']['TotalClaims']
female_risk = data[data['Gender'] == 'Female']['TotalClaims']
gender_model = stats.ttest_ind(male_risk, female_risk, equal_var=False)
results.append(("Risk Between Genders", gender_model.statistic, gender_model.pvalue))

# Ensure the directory exists
output_dir = r'C:\Users\teble\alpha-insurance-analytics\my_project\results'
os.makedirs(output_dir, exist_ok=True)

# Save results to a text file
with open(os.path.join(output_dir, 'task_3_results.txt'), 'w') as f:
    f.write("A/B Hypothesis Testing Results\n")
    f.write("====================================\n")
    for name, f_stat, p_value in results:
        result = "Reject Null Hypothesis" if p_value < 0.05 else "Fail to Reject Null Hypothesis"
        f.write(f"{name}:\n  F-stat: {f_stat:.4f}, p-value: {p_value:.4f} => {result}\n\n")

# Visualizations
sns.boxplot(x='Province', y='TotalClaims', data=data)
plt.title('Total Claims by Province')
plt.savefig(os.path.join(output_dir, 'claims_by_province.png'))

sns.boxplot(x='PostalCode', y='TotalClaims', data=data)
plt.title('Total Claims by Zip Code')
plt.savefig(os.path.join(output_dir, 'claims_by_zipcode.png'))

sns.boxplot(x='Gender', y='TotalClaims', data=data)
plt.title('Total Claims by Gender')
plt.savefig(os.path.join(output_dir, 'claims_by_gender.png'))

# Print summary to console
print("Task 3 completed. Results saved to 'task_3_results.txt' and visualizations exported.")
