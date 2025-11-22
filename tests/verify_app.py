import os
import joblib
import pandas as pd
import sys
from src.config import MODELS_DIR, GUI_FEATURES, PROCESSED_DATA_PATH, TARGET_COLUMNS

def verify_app_logic():
    print("Verifying App Logic...")

    # 1. Verify Feature Alignment
    print("\n1. Verifying Feature Alignment...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        # Drop target columns to get feature columns
        train_features = df.drop(TARGET_COLUMNS, axis=1).columns.tolist()
        
        print(f"Number of training features: {len(train_features)}")
        print(f"Number of GUI features: {len(GUI_FEATURES)}")
        
        missing_in_gui = set(train_features) - set(GUI_FEATURES)
        missing_in_train = set(GUI_FEATURES) - set(train_features)
        
        if missing_in_gui:
            print(f"WARNING: Features in training data but missing in GUI: {missing_in_gui}")
        if missing_in_train:
            print(f"WARNING: Features in GUI but missing in training data: {missing_in_train}")
            
        if not missing_in_gui and not missing_in_train:
            print("SUCCESS: Features are perfectly aligned.")
        else:
            # If there are mismatches, we need to see if they are critical
            # For example, One-Hot Encoding might produce different columns if not handled carefully
            pass

    except Exception as e:
        print(f"FAILED to load processed data: {e}")
        return

    # 2. Verify Model Prediction
    print("\n2. Verifying Model Prediction...")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    if not model_files:
        print("FAILED: No models found.")
        return

    # Create dummy input
    input_values = {feature: 0.0 for feature in GUI_FEATURES}
    input_df = pd.DataFrame([input_values])
    
    # Ensure input_df has the same columns as training data (if there were mismatches, this might fail)
    # We might need to reorder or add missing columns with 0
    for feature in train_features:
        if feature not in input_df.columns:
            input_df[feature] = 0.0
    input_df = input_df[train_features] # Reorder to match training

    print(f"Testing prediction with {len(model_files)} models...")
    success_count = 0
    for f in model_files:
        try:
            model_path = os.path.join(MODELS_DIR, f)
            model = joblib.load(model_path)
            prediction = model.predict(input_df)
            # print(f"  {f}: Prediction = {prediction[0]}")
            success_count += 1
        except Exception as e:
            print(f"  FAILED {f}: {e}")

    if success_count == len(model_files):
        print(f"\nSUCCESS: All {success_count} models predicted successfully.")
    else:
        print(f"\nWARNING: Only {success_count}/{len(model_files)} models predicted successfully.")

if __name__ == "__main__":
    verify_app_logic()
