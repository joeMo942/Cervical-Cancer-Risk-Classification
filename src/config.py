import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'data_set.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'kag_risk_factors_cervical_cancer_preprocessed.csv')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Features
NUMERICAL_FEATURES = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
                      'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 
                      'IUD (years)', 'STDs (number)']

CATEGORICAL_FEATURES = ['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis', 
                        'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 
                        'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 
                        'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 
                        'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 
                        'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 
                        'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 
                        'Citology', 'Biopsy']

TARGET_COLUMNS = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy']

GUI_FEATURES = ['Age', 'Number of sexual partners', 'First sexual intercourse',
                'Num of pregnancies', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN',
                'Dx:HPV', 'Dx', 'Smokes_0.0', 'Smokes_1.0', 'Hormonal Contraceptives_0.0',
                'Hormonal Contraceptives_1.0', 'IUD_0.0', 'IUD_1.0', 'STDs_0.0', 'STDs_1.0']
