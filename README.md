# Cervical Cancer Risk Classification using Machine Learning

This project aims to classify the risk of cervical cancer using various machine learning and deep learning techniques. It analyzes a dataset of risk factors to predict the presence of cervical cancer indicators (Biopsy).

## Dataset

The dataset used in this project is the **Cervical Cancer Risk Classification** dataset available on Kaggle.

- **Dataset Link**: [Cervical Cancer Risk Classification Dataset](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)

The dataset contains demographic information, habits, and medical history records of patients. It includes features such as:
- Age
- Number of sexual partners
- First sexual intercourse
- Number of pregnancies
- Smoking habits
- Hormonal contraceptive use
- IUD use
- STDs history
- Diagnosis results (Hinselmann, Schiller, Citology, Biopsy)

## Project Overview

The project involves the following steps:

1.  **Data Preprocessing**:
    -   Handling missing values (filling with mean or mode).
    -   Converting columns to numeric types.
    -   Dropping irrelevant columns (e.g., timestamps).
    -   Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) and **RandomUnderSampler**.

2.  **Exploratory Data Analysis (EDA)**:
    -   Visualizing distributions of numerical and categorical features using Histograms, Box plots, and Count plots.
    -   Correlation analysis.

3.  **Model Implementation**:
    -   **Machine Learning Models**:
        -   Logistic Regression
        -   Support Vector Machine (SVM)
        -   Gaussian Naive Bayes
        -   Random Forest Classifier
        -   Decision Tree Classifier
        -   K-Nearest Neighbors (KNN)
    -   **Ensemble Methods**:
        -   Stacking Classifier
        -   Bagging Classifier
        -   AdaBoost Classifier
        -   XGBoost Classifier
    -   **Deep Learning Models**:
        -   Convolutional Neural Network (CNN)
        -   Recurrent Neural Network (RNN)
        -   Long Short-Term Memory (LSTM)

4.  **Feature Selection**:
    -   Chi-Square
    -   Mutual Information
    -   ANOVA F-value

5.  **Hyperparameter Tuning**:
    -   GridSearchCV for optimizing model parameters.

6.  **Evaluation**:
    -   Models are evaluated using Accuracy, Precision, Recall, F1-score, and ROC AUC score.
    -   SHAP (SHapley Additive exPlanations) is used for model interpretability.

## Installation and Requirements

To run this project, it is recommended to use the provided setup script to create a virtual environment and install dependencies.

```bash
./setup.sh
```

Alternatively, you can manually create a virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1.  **Activate the virtual environment**:
    ```bash
    source venv/bin/activate
    ```

2.  **Run the pipeline**:
    The project uses `main.py` as the entry point. You can run it with different modes:

    -   **Run everything (Processing + Training)**:
        ```bash
        python main.py --mode all
        ```
    -   **Run Data Processing only**:
        ```bash
        python main.py --mode process
        ```
    -   **Run Model Training only**:
        ```bash
        python main.py --mode train
        ```
    -   **Launch the GUI**:
        ```bash
        python main.py --mode gui
        ```

## Results

The project compares various models to find the best classifier for predicting cervical cancer risk.
-   **Processed Data**: Saved in `data/processed/`.
-   **Trained Models**: Saved in `data/models/`.
