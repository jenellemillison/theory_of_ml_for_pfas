# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:17:49 2024

@author: hotco
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import t

# Step 1: Load and preprocess the data
file_path =r'C:\Users\hotco\OneDrive\Desktop\Theory of Machine Learning\Mod6\bonedensity.txt'
data = pd.read_csv(file_path, sep='\t')


data = data.dropna()
data = data.sort_values(by='age')


# Extract the relevant columns (age and spinal BMD)
X = data['age']
y = data['spnbmd']

# Step 2: Define function to calculate the optimal smoothing factor
def cross_validate_spline(X, y, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=1)
    smoothing_factors = np.linspace(0.1, 10, 100)  # Range of smoothing factors to test
    best_smoothing = None
    best_mse = float('inf')
    
    for s in smoothing_factors:
        mse_fold = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            spline = UnivariateSpline(X_train, y_train, s=s)
            y_pred = spline(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_fold.append(mse)
        
        avg_mse = np.mean(mse_fold)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_smoothing = s
    
    return best_smoothing

# Step 3: Fit the spline with the optimal smoothing factor
optimal_smoothing = cross_validate_spline(X, y)
spline = UnivariateSpline(X, y, s=optimal_smoothing)

# Step 4: Construct pointwise 90% confidence bands
n = len(X)
y_pred = spline(X)
residuals = y - y_pred
variance = np.var(residuals)
stderr = np.sqrt(variance)
t_value = t.ppf(0.95, df=n-1)

confidence_interval = t_value * stderr * np.sqrt(1 + 1/n)

# Upper and lower confidence bands
upper_band = y_pred + confidence_interval
lower_band = y_pred - confidence_interval

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'bo', label='Observed Data')
plt.plot(X, y_pred, 'r-', label='Fitted Spline')
plt.fill_between(X, lower_band, upper_band, color='gray', alpha=0.2, label='90% Confidence Band')
plt.xlabel('Age')
plt.ylabel('Relative Change in Spinal BMD')
plt.title('Cubic Smoothing Spline with 90% Confidence Bands')
plt.legend()
plt.show()
