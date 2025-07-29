# Heart Disease Prediction System

A user-friendly machine learning application for predicting heart disease risk using logistic regression.

## Features

✅ **Improved Efficiency:**
- Object-oriented design with clean class structure
- Optimized data handling and model training
- Proper variable naming and code organization
- Eliminated redundant code from original script

✅ **Enhanced User Experience:**
- Interactive command-line interface with clear prompts
- Input validation with helpful error messages
- Detailed feature descriptions for each input
- Professional result formatting with confidence scores
- Continuous prediction loop with exit option

✅ **Better Error Handling:**
- File existence validation
- Input range validation
- Graceful error recovery
- Keyboard interrupt handling

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Usage

1. Ensure you have the `heart_disease_data.csv` file in the same directory
2. Run the application:

```bash
python heart_disease_predictor.py
```

3. Follow the interactive prompts to enter patient information
4. View the prediction results with confidence scores
5. Choose to make additional predictions or exit

## Input Parameters

The system will prompt you for the following information:

| Parameter | Description | Valid Range |
|-----------|-------------|-------------|
| Age | Patient's age in years | 1-120 |
| Sex | 0 = Female, 1 = Male | 0-1 |
| Chest Pain Type | 0-3 (see detailed descriptions in app) | 0-3 |
| Resting BP | Resting blood pressure (mm Hg) | 50-300 |
| Cholesterol | Serum cholesterol (mg/dl) | 100-600 |
| Fasting Blood Sugar | >120 mg/dl (0=False, 1=True) | 0-1 |
| Resting ECG | ECG results (0-2) | 0-2 |
| Max Heart Rate | Maximum heart rate achieved | 50-250 |
| Exercise Angina | Exercise induced angina (0=No, 1=Yes) | 0-1 |
| ST Depression | ST depression by exercise | 0-10 |
| ST Slope | Slope of peak exercise ST segment | 0-2 |
| Major Vessels | Number colored by fluoroscopy | 0-4 |
| Thalassemia | Thalassemia type | 0-3 |

## Improvements Over Original Script

### Code Efficiency
- ✅ Fixed variable naming issues (`x.train_predict` → proper variables)
- ✅ Removed commented-out exploration code
- ✅ Organized code into logical class structure
- ✅ Optimized imports and removed unused code

### User Experience
- ✅ Added input validation with clear error messages
- ✅ Provided detailed descriptions for each medical parameter
- ✅ Formatted output with emojis and clear sections
- ✅ Added confidence scores and risk percentages
- ✅ Included medical disclaimer for safety

### Error Handling
- ✅ File existence checking
- ✅ Input type and range validation
- ✅ Graceful handling of interrupts
- ✅ Exception handling with user-friendly messages

## Sample Output

```
🏥 Heart Disease Prediction System
==================================================
📊 Loading heart disease dataset...
✅ Dataset loaded successfully!
   • Shape: (303, 14)
   • Missing values: 0
   • Healthy hearts: 165
   • Heart disease cases: 138

🤖 Training the prediction model...
✅ Model trained successfully!
   • Training accuracy: 0.869
   • Test accuracy: 0.852

============================================================
🏥 HEART DISEASE RISK ASSESSMENT
============================================================
Please provide the following information:

Age (years): 45
Sex (0: Female, 1: Male): 1
...

============================================================
📋 ASSESSMENT RESULTS
============================================================
Input Summary:
   • Age (years): 45
   • Sex (0: Female, 1: Male): 1
   ...

------------------------------------------------------------
✅ RESULT: LOW RISK
   The model indicates a LOW risk of heart disease.
   Confidence: 78.5%

📊 Risk Score: 21.5%

============================================================
⚠️  IMPORTANT DISCLAIMER:
This prediction is for educational purposes only.
Always consult with healthcare professionals for medical advice.
============================================================
```

## Note

This is an educational tool and should not be used as a substitute for professional medical advice. Always consult healthcare professionals for medical concerns.
