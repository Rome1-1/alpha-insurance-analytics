import numpy as np
import pandas as pd

def clean_missing_values(df):
    """
    Handle missing values appropriately.
    """
    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical columns with 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Fill boolean columns with False
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].fillna(False)

    return df


def fix_data_types(df):
    """
    Fix inconsistent data types and handle skewed data.
    """
    # Convert dates
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Treat `mmcode` as a string identifier
    if 'mmcode' in df.columns:
        df['mmcode'] = df['mmcode'].astype(str)

    # Convert discrete numeric features to categorical
    for col in ['Cylinders', 'NumberOfDoors']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('category')

    # Handle `CustomValueEstimate`: Clip and categorize bins
    if 'CustomValueEstimate' in df.columns:
        # Clip large values
        df['CustomValueEstimate'] = df['CustomValueEstimate'].clip(upper=500000)

        # Categorize into bins
        bins = [0, 50000, 100000, 200000, 500000]
        labels = ['Low', 'Medium', 'High', 'Very High']
        df['CustomValueCategory'] = pd.cut(df['CustomValueEstimate'], bins=bins, labels=labels)

    # Handle `TotalClaims`: Remove negative and zero values, then log-transform
    if 'TotalClaims' in df.columns:
        # Replace negatives with zero and apply log transformation
        df['TotalClaims'] = df['TotalClaims'].clip(lower=0)  # Replace negative with 0
        df['TotalClaims'] = np.log1p(df['TotalClaims'])  # Log transformation

    return df