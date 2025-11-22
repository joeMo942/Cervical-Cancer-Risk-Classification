import argparse
import sys
from src.data_processing import load_data, preprocess_data, save_data
from src.training import train_models
from gui.app import InputGUI

def main():
    parser = argparse.ArgumentParser(description="Cervical Cancer Risk Classification Pipeline")
    parser.add_argument('--mode', type=str, choices=['process', 'train', 'gui', 'all'], default='all',
                        help="Mode to run the pipeline: 'process' (data processing), 'train' (model training), 'gui' (launch GUI), or 'all' (process + train).")
    
    args = parser.parse_args()
    
    if args.mode in ['process', 'all']:
        print("Starting data processing...")
        df = load_data()
        if df is not None:
            df = preprocess_data(df)
            save_data(df)
            print("Data processing completed.")
        else:
            print("Data processing failed.")
            return

    if args.mode in ['train', 'all']:
        print("Starting model training...")
        results = train_models()
        print("Model training completed.")
        print("Results:", results)

    if args.mode == 'gui':
        print("Launching GUI...")
        app = InputGUI()
        app.mainloop()

if __name__ == "__main__":
    main()
