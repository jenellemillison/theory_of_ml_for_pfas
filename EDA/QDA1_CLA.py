# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:03:48 2024

@author: hotco
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load the dataset
pfas_data = pd.read_csv(r'C:\Users\hotco\OneDrive\Desktop\Theory of Machine Learning\Group Project\PFAs.csv') # Use the PFAS_ENV data

# Create a binary target variable based on the median of 'PFOA-VA'
median_pfoa_va = pfas_data['PFOA-VA'].median()
pfas_data['PFOA_Target'] = (pfas_data['PFOA-VA'] > median_pfoa_va).astype(int)

# Select predictor variables (we will use the other PFAS measurement columns ending in '-VA')
predictor_columns = [col for col in pfas_data.columns if col.endswith('-VA') and col != 'PFOA-VA']

X = pfas_data[predictor_columns]
y = pfas_data['PFOA_Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Quadratic Discriminant Analysis (QDA)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = qda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"QDA Model Accuracy: {accuracy * 100:.2f}%")



