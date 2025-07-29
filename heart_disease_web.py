#!/usr/bin/env python3
"""
Heart Disease Prediction System - Web Version
A modern web-based interface for heart disease risk assessment
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import json

app = Flask(__name__)

class HeartDiseaseModel:
    """Heart Disease Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.accuracy = 0
        self.is_trained = False
        self.load_and_train()
    
    def load_and_train(self):
        """Load data and train the model"""
        try:
            data_file = 'heart_disease_data.csv'
            if not os.path.exists(data_file):
                print(f"Warning: {data_file} not found!")
                return False
            
            # Load and prepare data
            heart_data = pd.read_csv(data_file)
            X = heart_data.drop(columns='target', axis=1)
            y = heart_data['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Train model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            test_predictions = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, test_predictions)
            self.is_trained = True
            
            print(f"Model trained successfully! Accuracy: {self.accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict(self, input_data):
        """Make prediction"""
        if not self.is_trained:
            return None, None
        
        input_array = np.array(input_data).reshape(1, -1)
        prediction = self.model.predict(input_array)[0]
        probability = self.model.predict_proba(input_array)[0]
        
        return prediction, probability

# Initialize model
predictor = HeartDiseaseModel()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_ready=predictor.is_trained, 
                         accuracy=predictor.accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if not predictor.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 500
        
        # Get input data
        data = request.json
        
        # Validate inputs
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                          'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        input_data = []
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            input_data.append(float(data[field]))
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'risk_score': float(probability[1] * 100),
            'confidence': float(max(probability) * 100),
            'risk_level': get_risk_level(probability[1] * 100),
            'message': get_risk_message(prediction, probability)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_risk_level(risk_score):
    """Get risk level based on score"""
    if risk_score < 25:
        return 'Very Low'
    elif risk_score < 50:
        return 'Low'
    elif risk_score < 75:
        return 'Moderate'
    else:
        return 'High'

def get_risk_message(prediction, probability):
    """Get risk message"""
    if prediction == 0:
        return f"The model indicates a LOW risk of heart disease with {probability[0]:.1%} confidence."
    else:
        return f"The model indicates a HIGH risk of heart disease with {probability[1]:.1%} confidence."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
