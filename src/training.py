import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import PROCESSED_DATA_PATH, MODELS_DIR, TARGET_COLUMNS

def train_models(data_path=PROCESSED_DATA_PATH):
    """Trains models for each target column and saves them."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Processed data not found at {data_path}. Run data_processing.py first.")
        return

    df_features = df.drop(TARGET_COLUMNS, axis=1)
    df_labels = df[TARGET_COLUMNS]

    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm_model', SVC(probability=True)),
        ('clf', GaussianNB()),
        ('rf_model', RandomForestClassifier()),
        ('Dt_model', DecisionTreeClassifier()),
        ('knn_model', KNeighborsClassifier())
    ]

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    results = {}

    for column in TARGET_COLUMNS:
        print(f"Training models for target: {column}")
        sm = SMOTE(random_state=42)
        # Handle cases where SMOTE might fail due to class distribution
        try:
            df_features_res, df_labels_res = sm.fit_resample(df_features, df_labels[column])
        except ValueError as e:
             print(f"Skipping SMOTE for {column} due to error: {e}")
             df_features_res, df_labels_res = df_features, df_labels[column]

        X_train, X_test, y_train, y_test = train_test_split(df_features_res, df_labels_res, test_size=0.25, random_state=42)

        for model_name, model in models:
            pipeline = Pipeline([
                ('over_sampling', SMOTE(random_state=42)),
                ('under_sampling', RandomUnderSampler(random_state=42)),
                ('model', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Save model
            model_filename = f"{model_name}_{column}_model.pkl"
            joblib.dump(pipeline, os.path.join(MODELS_DIR, model_filename))
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            
            results[f"{column}_{model_name}"] = {'accuracy': acc, 'roc_auc': roc_auc}
            print(f"  {model_name}: Accuracy={acc:.4f}, ROC_AUC={roc_auc:.4f}")

    return results

if __name__ == "__main__":
    train_models()
