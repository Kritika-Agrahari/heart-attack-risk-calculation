# Heart Disease Prediction using Logistic Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('heart_disease_data.csv')
# heart_data.head()
# heart_data.tail()
# heart_data.shape
# heart_data.info()
# heart_data.isnull().sum()
heart_data.describe()

# checking distribution of target variable
heart_data['target'].value_counts()

"""1 problem in heart
0 healthy heart
"""

# spliting the feature and target
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

print(x)

print(y)

# splitting the data into specfic way
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

# traning
model = LogisticRegression()

model.fit(x_train, y_train)

#accuracy on prediction data
x.train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x.train_predict, y_train)

print('Accuracy on Training data : ', training_data_accuracy)

#accuracy on test data
x.test_predict = model.predict(x_test)
training_data_accuracy = accuracy_score(x.test_predict, y_test)

print('Accuracy on Test data : ', training_data_accuracy)

# # building model prediction
# input_data = (61,1,0,148,203,0,1,161,0,0,2,1,3)
# #changing input_data to numpy
# input_data_as_numpy_array = np.asarray(input_data)
# # reshape the numpy array as we are predicting for only one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# prediction = model.predict(input_data_reshaped)
# print(prediction)
# # print(prediction)
# if (prediction[0]== 0):
#   print('The Person does not have a Heart Disease')
# else:
#   print('The Person has Heart Disease')

# Get input from the user for each feature
age = int(input("Enter age: "))
sex = int(input("Enter sex (0 for female, 1 for male): "))
cp = int(input("Enter chest pain type (0-3): "))
trestbps = int(input("Enter resting blood pressure: "))
chol = int(input("Enter serum cholestoral in mg/dl: "))
fbs = int(input("Enter fasting blood sugar > 120 mg/dl (0 for false, 1 for true): "))
restecg = int(input("Enter resting electrocardiographic results (0-2): "))
thalach = int(input("Enter maximum heart rate achieved: "))
exang = int(input("Enter exercise induced angina (0 for no, 1 for yes): "))
oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
ca = int(input("Enter number of major vessels (0-4) colored by flourosopy: "))
thal = int(input("Enter thal (0-3): "))


# building model prediction
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

#changing input_data to numpy
input_data_as_numpy_array = np.asarray(input_data)
# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
# print(prediction)
if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')