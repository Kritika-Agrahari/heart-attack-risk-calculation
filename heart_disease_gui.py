#!/usr/bin/env python3
"""
Heart Disease Prediction System - GUI Version
A beautiful and user-friendly graphical interface for heart disease risk assessment
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import threading

class HeartDiseasePredictorGUI:
    """Heart Disease Prediction System with Modern GUI"""
    
    def __init__(self):
        self.model = None
        self.data_file = 'heart_disease_data.csv'
        self.setup_gui()
        self.load_model_async()
        
    def setup_gui(self):
        """Initialize the GUI components"""
        self.root = tk.Tk()
        self.root.title("Heart Disease Prediction System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Predict.TButton', font=('Arial', 12, 'bold'))
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏥 Heart Disease Risk Assessment", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Loading model...", 
                                     style='Info.TLabel', foreground='orange')
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Create input fields
        self.create_input_fields(main_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=20, column=0, columnspan=2, pady=20)
        
        # Predict button
        self.predict_button = ttk.Button(button_frame, text="🔍 Predict Risk", 
                                        command=self.predict_risk, style='Predict.TButton',
                                        state='disabled')
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="🗑️ Clear All", 
                                 command=self.clear_fields)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="📋 Assessment Results", padding="10")
        results_frame.grid(row=21, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        results_frame.columnconfigure(0, weight=1)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=70,
                                                     font=('Consolas', 10), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Initial results message
        self.results_text.insert(tk.END, "Enter patient information above and click 'Predict Risk' to see results.\n\n")
        self.results_text.insert(tk.END, "⚠️  IMPORTANT: This tool is for educational purposes only.\n")
        self.results_text.insert(tk.END, "Always consult healthcare professionals for medical advice.")
        self.results_text.config(state=tk.DISABLED)
        
    def create_input_fields(self, parent):
        """Create input fields for all features"""
        self.input_vars = {}
        
        # Feature definitions with descriptions and ranges
        features = [
            ('age', 'Age (years)', 'Patient age in years', 1, 120),
            ('sex', 'Sex', '0 = Female, 1 = Male', 0, 1),
            ('cp', 'Chest Pain Type', '0=Typical angina, 1=Atypical angina, 2=Non-anginal, 3=Asymptomatic', 0, 3),
            ('trestbps', 'Resting Blood Pressure', 'Resting blood pressure in mm Hg', 50, 300),
            ('chol', 'Cholesterol', 'Serum cholesterol in mg/dl', 100, 600),
            ('fbs', 'Fasting Blood Sugar', 'Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)', 0, 1),
            ('restecg', 'Resting ECG', '0=Normal, 1=ST-T abnormality, 2=LV hypertrophy', 0, 2),
            ('thalach', 'Max Heart Rate', 'Maximum heart rate achieved', 50, 250),
            ('exang', 'Exercise Angina', 'Exercise induced angina (0=No, 1=Yes)', 0, 1),
            ('oldpeak', 'ST Depression', 'ST depression induced by exercise', 0.0, 10.0),
            ('slope', 'ST Slope', 'Slope of peak exercise ST segment (0=Up, 1=Flat, 2=Down)', 0, 2),
            ('ca', 'Major Vessels', 'Number of major vessels colored by fluoroscopy (0-4)', 0, 4),
            ('thal', 'Thalassemia', '0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Not described', 0, 3)
        ]
        
        row = 2
        for feature, label, description, min_val, max_val in features:
            # Label
            ttk.Label(parent, text=f"{label}:", style='Heading.TLabel').grid(
                row=row, column=0, sticky=tk.W, pady=2)
            
            # Input field
            if feature == 'oldpeak':
                self.input_vars[feature] = tk.DoubleVar()
                spinbox = ttk.Spinbox(parent, from_=min_val, to=max_val, increment=0.1,
                                     textvariable=self.input_vars[feature], width=15)
            else:
                self.input_vars[feature] = tk.IntVar()
                spinbox = ttk.Spinbox(parent, from_=min_val, to=max_val, increment=1,
                                     textvariable=self.input_vars[feature], width=15)
            
            spinbox.grid(row=row, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Description
            desc_label = ttk.Label(parent, text=f"({description})", style='Info.TLabel',
                                  foreground='gray')
            desc_label.grid(row=row+1, column=1, sticky=tk.W, padx=10, pady=(0, 5))
            
            row += 2
    
    def load_model_async(self):
        """Load and train the model in a separate thread"""
        def load_model():
            try:
                if not os.path.exists(self.data_file):
                    self.root.after(0, lambda: self.update_status(
                        f"❌ Error: '{self.data_file}' not found!", 'red'))
                    return
                
                # Update status
                self.root.after(0, lambda: self.update_status("📊 Loading dataset...", 'blue'))
                
                # Load data
                heart_data = pd.read_csv(self.data_file)
                
                # Prepare data
                X = heart_data.drop(columns='target', axis=1)
                y = heart_data['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                
                # Update status
                self.root.after(0, lambda: self.update_status("🤖 Training model...", 'blue'))
                
                # Train model
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                self.model.fit(X_train, y_train)
                
                # Calculate accuracy
                test_predictions = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, test_predictions)
                
                # Update status and enable predict button
                self.root.after(0, lambda: self.update_status(
                    f"✅ Model ready! (Accuracy: {accuracy:.1%})", 'green'))
                self.root.after(0, lambda: self.predict_button.config(state='normal'))
                
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"❌ Error: {str(e)}", 'red'))
        
        # Start loading in background thread
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def update_status(self, message, color):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)
    
    def validate_inputs(self):
        """Validate all input fields"""
        try:
            # Check if all fields have values
            for feature, var in self.input_vars.items():
                value = var.get()
                if feature == 'oldpeak':
                    if not (0 <= value <= 10):
                        raise ValueError(f"ST Depression must be between 0 and 10")
                else:
                    # Define valid ranges for each feature
                    ranges = {
                        'age': (1, 120), 'sex': (0, 1), 'cp': (0, 3), 'trestbps': (50, 300),
                        'chol': (100, 600), 'fbs': (0, 1), 'restecg': (0, 2), 'thalach': (50, 250),
                        'exang': (0, 1), 'slope': (0, 2), 'ca': (0, 4), 'thal': (0, 3)
                    }
                    min_val, max_val = ranges[feature]
                    if not (min_val <= value <= max_val):
                        feature_names = {
                            'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
                            'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol',
                            'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
                            'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
                            'slope': 'ST Slope', 'ca': 'Major Vessels', 'thal': 'Thalassemia'
                        }
                        raise ValueError(f"{feature_names[feature]} must be between {min_val} and {max_val}")
            
            return True
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
    
    def predict_risk(self):
        """Make prediction and display results"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded yet. Please wait.")
            return
        
        if not self.validate_inputs():
            return
        
        try:
            # Get input values
            feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            input_data = [self.input_vars[feature].get() for feature in feature_order]
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            probability = self.model.predict_proba(input_array)[0]
            
            # Display results
            self.display_results(prediction, probability, input_data, feature_order)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
    
    def display_results(self, prediction, probability, input_data, feature_order):
        """Display prediction results in the text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        self.results_text.insert(tk.END, "📋 HEART DISEASE RISK ASSESSMENT RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Input summary
        self.results_text.insert(tk.END, "📊 Input Summary:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        feature_names = {
            'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
            'trestbps': 'Resting BP', 'chol': 'Cholesterol', 'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG', 'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
            'oldpeak': 'ST Depression', 'slope': 'ST Slope', 'ca': 'Major Vessels', 'thal': 'Thalassemia'
        }
        
        for i, feature in enumerate(feature_order):
            self.results_text.insert(tk.END, f"   • {feature_names[feature]}: {input_data[i]}\n")
        
        # Prediction results
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        
        risk_percentage = probability[1] * 100
        confidence = max(probability) * 100
        
        if prediction == 0:
            self.results_text.insert(tk.END, "✅ RESULT: LOW RISK\n")
            self.results_text.insert(tk.END, f"   The model indicates a LOW risk of heart disease.\n")
            self.results_text.insert(tk.END, f"   Confidence: {confidence:.1f}%\n")
        else:
            self.results_text.insert(tk.END, "⚠️  RESULT: HIGH RISK\n")
            self.results_text.insert(tk.END, f"   The model indicates a HIGH risk of heart disease.\n")
            self.results_text.insert(tk.END, f"   Confidence: {confidence:.1f}%\n")
        
        self.results_text.insert(tk.END, f"\n📊 Risk Score: {risk_percentage:.1f}%\n")
        
        # Risk interpretation
        self.results_text.insert(tk.END, f"\n📈 Risk Interpretation:\n")
        if risk_percentage < 25:
            self.results_text.insert(tk.END, "   🟢 Very Low Risk (0-25%)\n")
        elif risk_percentage < 50:
            self.results_text.insert(tk.END, "   🟡 Low Risk (25-50%)\n")
        elif risk_percentage < 75:
            self.results_text.insert(tk.END, "   🟠 Moderate Risk (50-75%)\n")
        else:
            self.results_text.insert(tk.END, "   🔴 High Risk (75-100%)\n")
        
        # Disclaimer
        self.results_text.insert(tk.END, f"\n{'='*60}\n")
        self.results_text.insert(tk.END, "⚠️  IMPORTANT DISCLAIMER:\n")
        self.results_text.insert(tk.END, "This prediction is for educational purposes only.\n")
        self.results_text.insert(tk.END, "Always consult with healthcare professionals for medical advice.\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def clear_fields(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set(0)
        
        # Clear results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Enter patient information above and click 'Predict Risk' to see results.\n\n")
        self.results_text.insert(tk.END, "⚠️  IMPORTANT: This tool is for educational purposes only.\n")
        self.results_text.insert(tk.END, "Always consult healthcare professionals for medical advice.")
        self.results_text.config(state=tk.DISABLED)
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

def main():
    """Main function to run the GUI application"""
    app = HeartDiseasePredictorGUI()
    app.run()

if __name__ == "__main__":
    main()
