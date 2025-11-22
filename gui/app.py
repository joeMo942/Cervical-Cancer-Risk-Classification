import customtkinter as ctk
import joblib
import pandas as pd
import os
from src.config import MODELS_DIR, GUI_FEATURES

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class InputGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cervical Cancer Risk Prediction")
        self.geometry("600x800")

        self.feature_labels = GUI_FEATURES
        self.feature_inputs = {}

        # Grid configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self, text="Enter Patient Data", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Scrollable Frame for Inputs
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Features")
        self.scrollable_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        for i, feature in enumerate(self.feature_labels):
            # Create label for feature
            label = ctk.CTkLabel(self.scrollable_frame, text=feature)
            label.grid(row=i, column=0, padx=10, pady=10, sticky="w")
            
            # Create input box for feature
            entry = ctk.CTkEntry(self.scrollable_frame, placeholder_text="0.0")
            entry.grid(row=i, column=1, padx=10, pady=10, sticky="ew")
            self.feature_inputs[feature] = entry

        # Submit Button
        self.submit_button = ctk.CTkButton(self, text="Predict Risk", command=self.show_results, font=ctk.CTkFont(size=16, weight="bold"), height=40)
        self.submit_button.grid(row=2, column=0, padx=20, pady=20)

    def show_results(self):
        # Load saved models
        models = {}
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        except FileNotFoundError:
             # Fallback for testing if directory doesn't exist
             model_files = []

        if not model_files:
            # Using a simple dialog for error
            error_window = ctk.CTkToplevel(self)
            error_window.title("Error")
            error_window.geometry("300x150")
            label = ctk.CTkLabel(error_window, text="No trained models found.\nPlease run training first.")
            label.pack(pady=20)
            return

        for f in model_files:
            model_name = f.replace('.pkl', '')
            models[model_name] = joblib.load(os.path.join(MODELS_DIR, f))
        
        # Get input values from user
        try:
            input_values = {}
            for feature in self.feature_labels:
                val = self.feature_inputs[feature].get()
                if val == "":
                    val = 0.0 # Default to 0 if empty
                input_values[feature] = float(val)
            input_df = pd.DataFrame([input_values])
        except ValueError:
            error_window = ctk.CTkToplevel(self)
            error_window.title("Error")
            error_window.geometry("300x150")
            label = ctk.CTkLabel(error_window, text="Please enter valid numeric values.")
            label.pack(pady=20)
            return
        
        # Call each model and get the predictions
        results = {}
        for model_name, model in models.items():
            try:
                y_pred = model.predict(input_df)
                results[model_name] = y_pred[0]
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = "Error"
        
        # Display the results
        self.open_result_window(results)

    def open_result_window(self, results):
        result_window = ctk.CTkToplevel(self)
        result_window.title("Prediction Results")
        result_window.geometry("700x500")
        result_window.grid_columnconfigure(0, weight=1)
        result_window.grid_rowconfigure(1, weight=1)

        # Title
        title_label = ctk.CTkLabel(result_window, text="Analysis Results", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=20)

        # Scrollable Frame for Results
        result_frame = ctk.CTkScrollableFrame(result_window)
        result_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_columnconfigure(1, weight=1)

        # Headers
        ctk.CTkLabel(result_frame, text="Model", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5, padx=10, sticky="w")
        ctk.CTkLabel(result_frame, text="Prediction", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, pady=5, padx=10, sticky="w")

        for i, (model_name, prediction) in enumerate(results.items()):
            if prediction == "Error":
                prediction_text = "Error"
                text_color = "orange"
            else:
                prediction_text = "Has No Cancer" if prediction == 0 else "Has Cancer"
                text_color = "green" if prediction == 0 else "red"
            
            ctk.CTkLabel(result_frame, text=model_name).grid(row=i+1, column=0, pady=5, padx=10, sticky="w")
            ctk.CTkLabel(result_frame, text=prediction_text, text_color=text_color).grid(row=i+1, column=1, pady=5, padx=10, sticky="w")

if __name__ == "__main__":
    app = InputGUI()
    app.mainloop()
