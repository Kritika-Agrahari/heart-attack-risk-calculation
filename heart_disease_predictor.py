#!/usr/bin/env python3
"""
Heart Disease Prediction System
A user-friendly machine learning application for predicting heart disease risk
using logistic regression.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

class HeartDiseasePredictor:
    """Heart Disease Prediction System using Logistic Regression"""
    
    def __init__(self, data_file='heart_disease_data.csv'):
        self.data_file = data_file
        self.model = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.feature_descriptions = {
            'age': 'Age (years)',
            'sex': 'Sex (0: Female, 1: Male)',
            'cp': 'Chest Pain Type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Serum Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl (0: False, 1: True)',
            'restecg': 'Resting ECG Results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)',
            'thalach': 'Maximum Heart Rate Achieved',
            'exang': 'Exercise Induced Angina (0: No, 1: Yes)',
            'oldpeak': 'ST Depression Induced by Exercise',
            'slope': 'Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)',
            'ca': 'Number of Major Vessels Colored by Fluoroscopy (0-4)',
            'thal': 'Thalassemia (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Not described)'
        }
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset for training"""
        try:
            if not os.path.exists(self.data_file):
                print(f"❌ Error: Data file '{self.data_file}' not found!")
                print("Please ensure the heart disease dataset is in the current directory.")
                return False
                
            print("📊 Loading heart disease dataset...")
            self.heart_data = pd.read_csv(self.data_file)
            
            # Basic data info
            print(f"✅ Dataset loaded successfully!")
            print(f"   • Shape: {self.heart_data.shape}")
            print(f"   • Missing values: {self.heart_data.isnull().sum().sum()}")
            
            # Check target distribution
            target_counts = self.heart_data['target'].value_counts()
            print(f"   • Healthy hearts: {target_counts[0]}")
            print(f"   • Heart disease cases: {target_counts[1]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def train_model(self):
        """Train the logistic regression model"""
        print("\n🤖 Training the prediction model...")
        
        # Prepare features and target
        X = self.heart_data.drop(columns='target', axis=1)
        y = self.heart_data['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train the model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"✅ Model trained successfully!")
        print(f"   • Training accuracy: {train_accuracy:.3f}")
        print(f"   • Test accuracy: {test_accuracy:.3f}")
        
        return True
    
    def get_user_input(self):
        """Get user input with validation"""
        print("\n" + "="*60)
        print("🏥 HEART DISEASE RISK ASSESSMENT")
        print("="*60)
        print("Please provide the following information:")
        print()
        
        user_data = {}
        
        # Define input ranges for validation
        input_ranges = {
            'age': (1, 120),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (50, 300),
            'chol': (100, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (50, 250),
            'exang': (0, 1),
            'oldpeak': (0, 10),
            'slope': (0, 2),
            'ca': (0, 4),
            'thal': (0, 3)
        }
        
        for feature in self.feature_names:
            while True:
                try:
                    prompt = f"{self.feature_descriptions[feature]}: "
                    
                    if feature in ['oldpeak']:
                        value = float(input(prompt))
                    else:
                        value = int(input(prompt))
                    
                    # Validate range
                    min_val, max_val = input_ranges[feature]
                    if min_val <= value <= max_val:
                        user_data[feature] = value
                        break
                    else:
                        print(f"   ⚠️  Please enter a value between {min_val} and {max_val}")
                        
                except ValueError:
                    print("   ⚠️  Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    sys.exit(0)
        
        return user_data
    
    def make_prediction(self, user_data):
        """Make prediction based on user input"""
        # Convert to numpy array in the correct order
        input_array = np.array([user_data[feature] for feature in self.feature_names])
        input_reshaped = input_array.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(input_reshaped)[0]
        probability = self.model.predict_proba(input_reshaped)[0]
        
        return prediction, probability
    
    def display_results(self, prediction, probability, user_data):
        """Display prediction results in a user-friendly format"""
        print("\n" + "="*60)
        print("📋 ASSESSMENT RESULTS")
        print("="*60)
        
        # Display input summary
        print("Input Summary:")
        for feature, value in user_data.items():
            print(f"   • {self.feature_descriptions[feature]}: {value}")
        
        print("\n" + "-"*60)
        
        # Display prediction
        risk_percentage = probability[1] * 100
        
        if prediction == 0:
            print("✅ RESULT: LOW RISK")
            print(f"   The model indicates a LOW risk of heart disease.")
            print(f"   Confidence: {probability[0]:.1%}")
        else:
            print("⚠️  RESULT: HIGH RISK")
            print(f"   The model indicates a HIGH risk of heart disease.")
            print(f"   Confidence: {probability[1]:.1%}")
        
        print(f"\n📊 Risk Score: {risk_percentage:.1f}%")
        
        # Add disclaimer
        print("\n" + "="*60)
        print("⚠️  IMPORTANT DISCLAIMER:")
        print("This prediction is for educational purposes only.")
        print("Always consult with healthcare professionals for medical advice.")
        print("="*60)
    
    def run(self):
        """Main application loop"""
        print("🏥 Heart Disease Prediction System")
        print("=" * 50)
        
        # Load data and train model
        if not self.load_and_prepare_data():
            return
        
        if not self.train_model():
            return
        
        while True:
            try:
                # Get user input
                user_data = self.get_user_input()
                
                # Make prediction
                prediction, probability = self.make_prediction(user_data)
                
                # Display results
                self.display_results(prediction, probability, user_data)
                
                # Ask if user wants to continue
                print("\n" + "="*60)
                while True:
                    continue_choice = input("Would you like to make another prediction? (y/n): ").lower().strip()
                    if continue_choice in ['y', 'yes']:
                        break
                    elif continue_choice in ['n', 'no']:
                        print("\n👋 Thank you for using Heart Disease Prediction System!")
                        return
                    else:
                        print("   Please enter 'y' for yes or 'n' for no.")
                        
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                print("Please try again.")

def main():
    """Main function to run the application"""
    predictor = HeartDiseasePredictor()
    predictor.run()

if __name__ == "__main__":
    main()
