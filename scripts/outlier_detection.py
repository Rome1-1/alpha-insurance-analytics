import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def detect_outliers(df, column):
    """
    Detect and visualize outliers using the Interquartile Range (IQR) method.
    Handles skewed data with log or square root transformations.
    """
    clean_data = df[column].dropna()

    if clean_data.empty:
        print(f"No data available for column {column}")
        return

    # Check for highly skewed data
    skewness = clean_data.skew()
    transformed_data = clean_data

    # Apply log transformation for highly skewed data
    if abs(skewness) > 1:
        print(f"Data is highly skewed. Applying log transformation to {column}.")
        # Log transformation handles zeros gracefully
        transformed_data = np.log1p(clean_data)  # Using log1p handles zeros correctly
    else:
        print(f"Data is not highly skewed. Using raw values for {column}.")

    # Handle extreme values by clipping (optional step)
    if column in ['TotalClaims', 'CustomValueEstimate']:
        print(f"Clipping extreme values for {column}.")
        # Clip extreme values at the 99th percentile to remove outliers
        transformed_data = transformed_data.clip(upper=transformed_data.quantile(0.99))

    # IQR method for outlier detection
    Q1 = transformed_data.quantile(0.25)
    Q3 = transformed_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = transformed_data[(transformed_data < lower_bound) | (transformed_data > upper_bound)]

    # Print outliers if found
    if not outliers.empty:
        print(f"Outliers detected in column {column}:\n{outliers.describe()}")
    else:
        print(f"No outliers detected in column {column}.")

    # Plot boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=transformed_data)
    plt.title(f'Boxplot for {column} (Transformed)')
    plt.show()