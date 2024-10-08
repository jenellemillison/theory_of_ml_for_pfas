# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:33:51 2024

@author: alexandercl
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture  # Using GMM for clustering
from sklearn.manifold import LocallyLinearEmbedding  # For dimensionality reduction
import numpy as np


pfa_data = pd.read_csv(\PFAs.csv') #filepath for PFAS_ENV.csv
pfa_dict = pd.read_csv(\PFAS_Data_Dictionary.csv', encoding='ISO-8859-1')

# Display the first few rows of each dataset for exploration
print("PFAs Data:")
print(pfa_data.head())

print("\nPFAs Data Dictionary:")
print(pfa_dict.head())

# Convert 'DATE' column to datetime format
pfa_data['DATE'] = pd.to_datetime(pfa_data['DATE'], format='%m/%d/%Y')

#CLUSTERING MODEL (GMM with 8 Clusters from AIC/BIC)

# Select numeric columns for clustering
numeric_columns = pfa_data.select_dtypes(include=['float64', 'int64']).columns

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
pfa_data_imputed = imputer.fit_transform(pfa_data[numeric_columns])

# Scale the data
scaler = StandardScaler()
pfa_scaled = scaler.fit_transform(pfa_data_imputed)

# Apply Gaussian Mixture Model with the optimal number of clusters (8 clusters)
gmm_final = GaussianMixture(n_components=8, random_state=42)
pfa_data['Cluster'] = gmm_final.fit_predict(pfa_scaled)

# Analyze cluster centers
cluster_centers = pd.DataFrame(gmm_final.means_, columns=numeric_columns)
print("Cluster Centers (Scaled):")
print(cluster_centers)

#VISUALIZE THE CLUSTERS USING LLE (2D Plot)

# Apply LLE to reduce the dimensionality to 2 components
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
lle_components = lle.fit_transform(pfa_scaled)

# Create a DataFrame for visualization
lle_df = pd.DataFrame(data=lle_components, columns=['LLE1', 'LLE2'])
lle_df['Cluster'] = pfa_data['Cluster']

# Plot the LLE components with cluster labels
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LLE1', y='LLE2', hue='Cluster', data=lle_df, palette='tab10', legend='full')
plt.title('LLE Plot of Clusters (2D Visualization)')
plt.show()

#CLUSTER STATISTICS AND INSIGHTS

# Cluster size: How many points fall into each cluster
print("\nCluster Sizes:")
print(pfa_data['Cluster'].value_counts())

# Summary statistics for each cluster
print("\nCluster Summary Statistics:")
cluster_summary = pfa_data.groupby('Cluster').mean()
print(cluster_summary)

#Save the clustered data
pfa_data.to_csv('Clustered_PFAs_Data.csv', index=False)


#parameter tuning of n_neighbors

from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
import seaborn as sns

# Try different values of n_neighbors for LLE
n_neighbors_values = [5, 10, 15, 20, 30]  # Example values to tune

#Loop through the different values and plot the LLE results
for n_neighbors in n_neighbors_values:
    # Apply LLE with the current n_neighbors value
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=42)
    lle_components = lle.fit_transform(pfa_scaled)
    
    # Create a DataFrame for visualization
    lle_df = pd.DataFrame(data=lle_components, columns=['LLE1', 'LLE2'])
    lle_df['Cluster'] = pfa_data['Cluster']
    
    # Plot the LLE components with cluster labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='LLE1', y='LLE2', hue='Cluster', data=lle_df, palette='tab10', legend='full')
    plt.title(f'LLE Plot of Clusters (n_neighbors = {n_neighbors})')
    plt.show()
    
    
import matplotlib.pyplot as plt
import seaborn as sns

#CLUSTER PROFILING

#Summary statistics for each cluster
print("\nCluster Summary Statistics:")
cluster_summary = pfa_data.groupby('Cluster').mean()
print(cluster_summary)

#Visualize the summary statistics for each cluster (mean values of features)
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='coolwarm')
plt.title('Cluster Summary Statistics (Mean Values)')
plt.show()

#CLUSTER COMPARISON USING BOX PLOTS

# Select key features to compare across clusters
key_features = ['PFOS-VA', 'PFOA-VA', 'PFBA-VA', 'PFHxS-VA']  # Example features related to PFAS

# Plot box plots for each feature by cluster
for feature in key_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=feature, data=pfa_data, palette='tab10')
    plt.title(f'Box Plot of {feature} by Cluster')
    plt.show()

#CORRELATION WITHIN CLUSTERS

# Calculate and visualize correlations within each cluster
for cluster in sorted(pfa_data['Cluster'].unique()):
    cluster_data = pfa_data[pfa_data['Cluster'] == cluster]
    cluster_corr = cluster_data[numeric_columns].corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for Cluster {cluster}')
    plt.show()

#FEATURE IMPORTANCE (Using Random Forest to Analyze Important Features for Clustering)


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Ensure 'numeric_columns' is a list of valid column names (drop 'TIME' if it exists)
if 'TIME' in numeric_columns:
    numeric_columns_no_time = numeric_columns.drop('TIME', errors='ignore')
else:
    numeric_columns_no_time = numeric_columns

# Convert to list in case it is not
numeric_columns_no_time = list(numeric_columns_no_time)

# Prepare data for classification model (using clusters as target variable)
X = pfa_data[numeric_columns_no_time]  # Ensure valid column selection, excluding 'TIME'
y = pfa_data['Cluster']  # Cluster labels

# Fit a random forest classifier to determine feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances from the random forest model
feature_importances = pd.Series(rf_model.feature_importances_, index=numeric_columns_no_time)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='skyblue')
plt.title('Feature Importance for Clustering')
plt.xlabel('Importance')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

# Selected top features for the predictive model
top_features = ['PFOS-VA', 'PFBA-VA', 'PFHxS-VA', 'PFOA-VA']

# Prepare the feature matrix (X) and the target variable (y)
X = pfa_data[top_features]  # Using the top features as the predictors
y = pfa_data['Cluster']  # Cluster labels as the target

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Perform cross-validation to check the model's generalization performance
cross_val_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores) * 100:.2f}%")

# Plot feature importance from the Random Forest model
feature_importances = pd.Series(rf_model.feature_importances_, index=top_features)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='skyblue')
plt.title('Feature Importance from Random Forest (Top Features)')
plt.xlabel('Importance')
plt.show()