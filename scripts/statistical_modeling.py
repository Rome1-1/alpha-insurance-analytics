import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import lime
from lime.lime_tabular import LimeTabularExplainer
import shap

# Data Loading with Improved Handling
def load_data(file_path):
    """Load the CSV data while handling warnings and setting appropriate data types."""
    try:
        df = pd.read_csv(file_path, low_memory=False)  # Avoid dtype warning by reading in chunks
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data Cleaning
def clean_data(df):
    """Clean the dataset by handling missing values and irrelevant columns."""
    print("Columns in the dataset:")
    print(df.columns.tolist())  # List all column names
    
    # Drop irrelevant columns (example columns, replace with actual)
    irrelevant_columns = ['irrelevant_column1', 'irrelevant_column2']  # Adjust as per the output
    df.drop(columns=[col for col in irrelevant_columns if col in df.columns], inplace=True)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    
    # Impute numeric columns with the mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Impute categorical columns with the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
    
    # Handle date columns (e.g., if 'VehicleIntroDate' is a date column)
    date_cols = df.select_dtypes(include=['object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime, invalid values become NaT
            date_imputer = SimpleImputer(strategy='most_frequent')
            df[col] = date_imputer.fit_transform(df[[col]])  # Impute with most frequent date value
        except Exception as e:
            print(f"Error processing date column {col}: {e}")
    
    df.dropna(axis=0, inplace=True)  # Drop rows with missing target variables if any
    
    print(f"Data cleaned. Shape: {df.shape}")
    return df

# Feature Engineering
def feature_engineering(df):
    """Create new features or transform existing ones for better predictive power."""
    # Example of creating new feature: Car age (Current year - Vehicle Registration Year)
    current_year = pd.Timestamp.now().year
    df['CarAge'] = current_year - df['RegistrationYear']
    
    # Encoding categorical features if needed (e.g., one-hot encoding for categorical columns)
    df = pd.get_dummies(df, drop_first=True)
    
    print("Feature Engineering completed.")
    return df

# Model Building: Linear Regression, Random Forest, XGBoost
def build_models(X_train, y_train, X_test, y_test):
    """Build and evaluate linear regression, random forest, and XGBoost models."""
    # Initialize the models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the models
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Model evaluation
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"Linear Regression - RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")
    print(f"Random Forest - RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")
    
    return lr_model, rf_model

# Model Interpretation with LIME
def interpret_with_lime(model, X_train, X_test):
    """Use LIME to interpret the model's predictions."""
    explainer = LimeTabularExplainer(X_train.values, training_labels=y_train.values, mode='regression')
    
    # Choose a sample to explain
    i = 5  # Example index
    exp = explainer.explain_instance(X_test.iloc[i].values, model.predict)
    
    exp.show_in_notebook(show_table=True, show_all=False)

# Model Interpretation with SHAP
def interpret_with_shap(model, X_train):
    """Use SHAP to explain the feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Visualize SHAP values for the first prediction
    shap.summary_plot(shap_values, X_train)

# Main Function
def main():
    # Load the dataset
    df = load_data(r'C:\Users\teble\alpha-insurance-analytics\my_project\data\cleanedinsurance_data.csv')
    if df is None:
        return
    
    # Clean the data
    df = clean_data(df)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Select target and features
    target = 'TotalPremium'  # Example target column
    X = df.drop(columns=[target])
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and evaluate models
    lr_model, rf_model = build_models(X_train, y_train, X_test, y_test)
    
    # Interpret the models with LIME and SHAP
    interpret_with_lime(rf_model, X_train, X_test)
    interpret_with_shap(rf_model, X_train)

# Run the main function
if __name__ == "__main__":
    main()
