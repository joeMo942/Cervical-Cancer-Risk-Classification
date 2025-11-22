import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def load_data(path=RAW_DATA_PATH):
    """Loads the dataset from the specified path."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None

def preprocess_data(df):
    """Preprocesses the dataset."""
    # Replace '?' with NaN
    df = df.replace('?', pd.NA)
    
    # Convert to numeric
    df = df.apply(pd.to_numeric)
    
    # Drop unnecessary columns
    df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], inplace=True, axis=1)
    
    # Fill missing values for numerical features with mean
    for feature in NUMERICAL_FEATURES:
        if feature in df.columns:
            feature_mean = round(df[feature].mean(), 1)
            df[feature] = df[feature].fillna(feature_mean)
            
    # Fill missing values for categorical features with mode
    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            feature_mode = df[feature].mode()[0]
            df[feature] = df[feature].fillna(feature_mode)
            
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs'])
    
    # Drop redundant columns after encoding (if they exist)
    cols_to_drop = ['Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)',
                    'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
                    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
                    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
                    'STDs:HPV']
    
    df.drop([c for c in cols_to_drop if c in df.columns], inplace=True, axis=1)
    
    return df

def save_data(df, path=PROCESSED_DATA_PATH):
    """Saves the processed dataframe to a CSV file."""
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        save_data(df)
