import tkinter as tk
import tkinter.ttk as ttk
import joblib
import pandas as pd
import os
from src.config import MODELS_DIR, GUI_FEATURES

class InputGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Cervical Cancer Risk Prediction")
        self.root.geometry("500x700")

        self.feature_labels = GUI_FEATURES
        self.feature_inputs = {}
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        ttk.Label(scrollable_frame, text="Enter Patient Data", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=20)

        for i, feature in enumerate(self.feature_labels):
            # Create label for feature
            ttk.Label(scrollable_frame, text=feature).grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
            # Create input box for feature
            self.feature_inputs[feature] = ttk.Entry(scrollable_frame)
            self.feature_inputs[feature].grid(row=i+1, column=1, padx=10, pady=5)

        # Create submit button
        ttk.Button(scrollable_frame, text="Predict", command=self.show_results).grid(row=len(self.feature_labels)+2, column=0, columnspan=2, pady=20)

        self.root.mainloop()
        
    def show_results(self):
        # Load saved models
        models = {}
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        
        if not model_files:
            tk.messagebox.showerror("Error", "No trained models found. Please run training first.")
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
            tk.messagebox.showerror("Error", "Please enter valid numeric values.")
            return
        
        # Call each model and get the predictions
        results = {}
        for model_name, model in models.items():
            try:
                # Ensure input columns match model expectation (this might need more robust handling)
                # For now, we assume the model was trained on the same features or a subset
                # We might need to align columns if the model expects specific ones
                # But since we used the same feature set for training, it should be fine if the order matches or if we select
                # However, the training script drops target columns. The GUI features list seems to match the input features.
                # Let's try to predict.
                y_pred = model.predict(input_df)
                results[model_name] = y_pred[0]
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = "Error"
        
        # Display the results
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction Results")
        result_window.geometry("600x400")
        
        # Create a treeview widget
        tree = ttk.Treeview(result_window, columns=("Model", "Prediction"))
        tree.heading("#0", text="No.")
        tree.heading("Model", text="Model")
        tree.heading("Prediction", text="Prediction")
        tree.column("#0", width=50)
        tree.column("Model", width=300)
        tree.column("Prediction", width=150)
        
        # Configure tags for colors
        tree.tag_configure('no_cancer', foreground='green')
        tree.tag_configure('has_cancer', foreground='red')
        tree.tag_configure('error', foreground='orange')
        
        # Insert the results into the treeview
        for i, (model_name, prediction) in enumerate(results.items()):
            if prediction == "Error":
                prediction_text = "Error"
                tag = 'error'
            else:
                prediction_text = "Has No Cancer" if prediction == 0 else "Has Cancer"
                tag = 'no_cancer' if prediction == 0 else 'has_cancer'
            
            tree.insert("", "end", text=i+1, values=(model_name, prediction_text), tags=(tag,))
        
        tree.pack(padx=10, pady=10, fill="both", expand=True)

if __name__ == "__main__":
    InputGUI()
