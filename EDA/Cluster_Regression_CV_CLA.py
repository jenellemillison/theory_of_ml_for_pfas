# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:30:58 2024

@author: hotco
"""
import os

# Set OMP_NUM_THREADS environment variable to avoid memory leak in KMeans on Windows
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

     #1. LOAD AND EXPLORE DATA
pfa_data = pd.read_csv(r'C:\Users\hotco\OneDrive\Desktop\Theory of Machine Learning\Group Project\PFAs.csv')
pfa_dict = pd.read_csv(r'C:\Users\hotco\OneDrive\Desktop\Theory of Machine Learning\Group Project\PFAS_Data_Dictionary.csv',encoding='ISO-8859-1')

# Display the first few rows of each dataset for exploration
print("PFAs Data:")
print(pfa_data.head())

print("\nPFAs Data Dictionary:")
print(pfa_dict.head())

# Basic information about the datasets
print("\nPFAs Data Info:")
print(pfa_data.info())

print("\nPFAs Data Dictionary Info:")
print(pfa_dict.info())

# Summary statistics of the numerical columns in the PFAs dataset
print("\nSummary Statistics of PFAs Data:")
print(pfa_data.describe())

# Check for missing values in both datasets
print("\nMissing Values in PFAs Data:")
print(pfa_data.isnull().sum())

print("\nMissing Values in PFAs Data Dictionary:")
print(pfa_dict.isnull().sum())

# Convert 'DATE' column to datetime format
pfa_data['DATE'] = pd.to_datetime(pfa_data['DATE'], format='%m/%d/%Y')


numeric_columns = pfa_data.select_dtypes(include=['float64', 'int64']).columns

    #2 VISUALIZE AND FIND TRENDS

# Plot the distribution of numeric variables
plt.figure(figsize=(12, 8))
pfa_data[numeric_columns].hist(bins=15, figsize=(20, 15), layout=(5, 5))
plt.tight_layout()
plt.show()

#Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(pfa_data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#Correlation analysis
correlation_matrix = pfa_data[numeric_columns].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

#Trend of PFAs measurements over time (example with PFOS-VA)
plt.figure(figsize=(12, 6))
sns.lineplot(data=pfa_data, x='DATE', y='PFOS-VA', ci=None)
plt.title('PFOS-VA Trend Over Time')
plt.show()

    #3 APPLY CLUSTERING

# Select numeric columns for clustering
numeric_columns = pfa_data.select_dtypes(include=['float64', 'int64']).columns

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
pfa_data_imputed = imputer.fit_transform(pfa_data[numeric_columns])

# Scale the data
scaler = StandardScaler()
pfa_scaled = scaler.fit_transform(pfa_data_imputed)

# Elbow method to find the optimal number of clusters
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pfa_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

#K-means clustering
optimal_clusters = 3  # Assume 3 clusters based on elbow method
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
pfa_data['Cluster'] = kmeans.fit_predict(pfa_scaled)

# Analyze cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_columns)
print("Cluster Centers (Scaled):")
print(cluster_centers)

numeric_columns_no_time = numeric_columns.drop('TIME')

# Compare PFAs concentrations by cluster
pfa_data.groupby('Cluster')[numeric_columns_no_time].mean().T.plot(kind='bar', figsize=(15, 8))
plt.title('Average PFAs Concentrations by Cluster (Excluding TIME)')
plt.show()

# Visualize clusters by selecting two numeric features (you can adjust these to any meaningful pair)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pfa_data[numeric_columns[0]], y=pfa_data[numeric_columns[1]], hue='Cluster', data=pfa_data, palette='Set2')
plt.title('K-means Clusters')
plt.show()

    #4.REGRESSION 

# Define features and target variable
target = 'PFOA-VA'
features = pfa_data.select_dtypes(include=['float64', 'int64']).columns
features = [col for col in features if col != target]  # Exclude the target variable from features

# Impute missing values with the median for both features and target
X = imputer.fit_transform(pfa_data[features])
y = imputer.fit_transform(pfa_data[[target]]).ravel()

# Split the data into train and test sets (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (standardize them)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Get the coefficients from the linear model
coefficients = pd.DataFrame(lr_model.coef_, index=features, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Initialize and train Ridge regression model
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression - Mean Squared Error (MSE): {mse_ridge}")
print(f"Ridge Regression - R-squared (R²): {r2_ridge}")

        #5. CROSS VALIDATION
        
from sklearn.model_selection import cross_val_score
import numpy as np

# Define the scorer for MSE
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform 5-fold cross-validation for Linear Regression
cv_mse_lr = cross_val_score(lr_model, X, y, cv=5, scoring=mse_scorer)
cv_r2_lr = cross_val_score(lr_model, X, y, cv=5, scoring='r2')

# Perform 5-fold cross-validation for Ridge Regression
cv_mse_ridge = cross_val_score(ridge_model, X, y, cv=5, scoring=mse_scorer)
cv_r2_ridge = cross_val_score(ridge_model, X, y, cv=5, scoring='r2')

# Convert negative MSE to positive for easier interpretation
cv_mse_lr = -cv_mse_lr
cv_mse_ridge = -cv_mse_ridge

# Print the cross-validation results
print(f"Linear Regression Cross-Validation MSE: {np.mean(cv_mse_lr):.4f} (Std: {np.std(cv_mse_lr):.4f})")
print(f"Linear Regression Cross-Validation R²: {np.mean(cv_r2_lr):.4f} (Std: {np.std(cv_r2_lr):.4f})\n")

print(f"Ridge Regression Cross-Validation MSE: {np.mean(cv_mse_ridge):.4f} (Std: {np.std(cv_mse_ridge):.4f})")
print(f"Ridge Regression Cross-Validation R²: {np.mean(cv_r2_ridge):.4f} (Std: {np.std(cv_r2_ridge):.4f})")
