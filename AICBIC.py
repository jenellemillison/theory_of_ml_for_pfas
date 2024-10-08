# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:59:50 2024

@author: alexandercl
"""

import os

# Set OMP_NUM_THREADS environment variable to avoid memory leak in KMeans on Windows
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
import numpy as np

# 1. LOAD AND EXPLORE DATA
pfa_data = pd.read_csv(('\PFAs.csv') #filepath
pfa_dict = pd.read_csv('\PFAS_Data_Dictionary.csv', encoding='ISO-8859-1') #filepath

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

# 2. VISUALIZE AND FIND TRENDS
numeric_columns = pfa_data.select_dtypes(include=['float64', 'int64']).columns

# Plot the distribution of numeric variables
plt.figure(figsize=(12, 8))
pfa_data[numeric_columns].hist(bins=15, figsize=(20, 15), layout=(5, 5))
plt.tight_layout()
plt.show()

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(pfa_data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Correlation analysis
correlation_matrix = pfa_data[numeric_columns].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Trend of PFAs measurements over time (example with PFOS-VA)
plt.figure(figsize=(12, 6))
sns.lineplot(data=pfa_data, x='DATE', y='PFOS-VA', ci=None)
plt.title('PFOS-VA Trend Over Time')
plt.show()

# 3. APPLY CLUSTERING USING GMM AND BIC/AIC

# Select numeric columns for clustering
numeric_columns = pfa_data.select_dtypes(include=['float64', 'int64']).columns

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
pfa_data_imputed = imputer.fit_transform(pfa_data[numeric_columns])

# Scale the data
scaler = StandardScaler()
pfa_scaled = scaler.fit_transform(pfa_data_imputed)

# Apply Gaussian Mixture Model and compute BIC and AIC for different numbers of clusters
bic_scores = []
aic_scores = []
K = range(1, 11)

for k in K:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(pfa_scaled)
    bic_scores.append(gmm.bic(pfa_scaled))
    aic_scores.append(gmm.aic(pfa_scaled))

# Plot the BIC and AIC scores
plt.figure(figsize=(8, 5))
plt.plot(K, bic_scores, label='BIC', marker='o')
plt.plot(K, aic_scores, label='AIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('BIC and AIC Scores for Optimal k')
plt.legend()
plt.show()

# Find the optimal number of clusters based on BIC
optimal_clusters = np.argmin(bic_scores) + 1
print(f"Optimal number of clusters based on BIC: {optimal_clusters}")

# Apply GMM with the optimal number of clusters
gmm_optimal = GaussianMixture(n_components=optimal_clusters, random_state=42)
pfa_data['Cluster'] = gmm_optimal.fit_predict(pfa_scaled)

# Analyze cluster centers
cluster_centers = pd.DataFrame(gmm_optimal.means_, columns=numeric_columns)
print("Cluster Centers (Scaled):")
print(cluster_centers)
