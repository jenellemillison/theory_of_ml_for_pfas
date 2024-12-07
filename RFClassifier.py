# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 04:51:17 2024

@author: hotco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompletePFASAnalyzer:
    def __init__(self, data_path, dict_path=None, scaler_type='standard', imputer_type='median'):
        """
        Initialize the PFAS analyzer with enhanced preprocessing options.
        
        Parameters:
        -----------
        data_path : str
            Path to the PFAS data CSV file
        dict_path : str, optional
            Path to the data dictionary CSV file
        scaler_type : str, optional
            Type of scaler ('standard' or 'robust')
        imputer_type : str, optional
            Type of imputer ('median', 'mean', or 'knn')
        """
        self.data = pd.read_csv(data_path)
        if dict_path:
            self.data_dict = pd.read_csv(dict_path, encoding='ISO-8859-1')
        self.scaler_type = scaler_type
        self.imputer_type = imputer_type
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess the PFAS data with enhanced preprocessing options."""
        try:
            self.data['DATE'] = pd.to_datetime(self.data['DATE'], format='%m/%d/%Y')
        except Exception as e:
            print(f"Warning: Date conversion failed - {e}")
            
        self.numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        constant_columns = [col for col in self.numeric_columns 
                          if self.data[col].nunique() == 1]
        self.numeric_columns = self.numeric_columns.drop(constant_columns)
        
        if self.imputer_type == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=self.imputer_type)
        
        self.data_imputed = imputer.fit_transform(self.data[self.numeric_columns])
        
        if self.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        self.data_scaled = scaler.fit_transform(self.data_imputed)

    def analyze_distributions(self):
        """Analyze feature distributions with detailed statistics."""
        stats_df = pd.DataFrame({
            'skewness': self.data[self.numeric_columns].skew(),
            'kurtosis': self.data[self.numeric_columns].kurtosis(),
            'missing_pct': self.data[self.numeric_columns].isnull().mean() * 100,
            'unique_values': self.data[self.numeric_columns].nunique(),
            'std_dev': self.data[self.numeric_columns].std()
        }).sort_values('skewness', ascending=False)
        
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(stats_df.head(6).index):
            plt.subplot(2, 3, i+1)
            sns.histplot(self.data[feature].dropna(), bins=30)
            plt.title(f'{feature}\nSkewness: {stats_df.loc[feature, "skewness"]:.2f}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return stats_df

    def analyze_correlations(self):
        """Analyze and visualize feature correlations."""
        corr_matrix = self.data[self.numeric_columns].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        return corr_matrix, high_corr_pairs

    def find_optimal_clusters(self, max_clusters=15):
        """Find optimal number of clusters using multiple metrics."""
        n_components_range = range(2, max_clusters + 1)
        metrics = {'aic': [], 'bic': [], 'silhouette': []}
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(self.data_scaled)
            labels = gmm.predict(self.data_scaled)
            
            metrics['aic'].append(gmm.aic(self.data_scaled))
            metrics['bic'].append(gmm.bic(self.data_scaled))
            metrics['silhouette'].append(silhouette_score(self.data_scaled, labels))
            
        self._plot_cluster_metrics(n_components_range, metrics)
        return metrics

    def hierarchical_cluster_analysis(self, primary_clusters=4, secondary_clusters=8):
        """Perform hierarchical clustering analysis."""
        gmm_primary = GaussianMixture(n_components=primary_clusters, random_state=42)
        primary_labels = gmm_primary.fit_predict(self.data_scaled)
        self.data['Primary_Cluster'] = primary_labels
        
        secondary_labels = np.zeros(len(self.data), dtype=int)
        subcluster_info = {}
        
        for i in range(primary_clusters):
            mask = primary_labels == i
            cluster_size = np.sum(mask)
            
            if cluster_size > secondary_clusters:
                cluster_data = self.data_scaled[mask]
                n_subclusters = min(secondary_clusters, cluster_size - 1)
                
                gmm_secondary = GaussianMixture(n_components=n_subclusters, random_state=42)
                sub_labels = gmm_secondary.fit_predict(cluster_data)
                
                secondary_labels[mask] = sub_labels + i * n_subclusters
                
                subcluster_info[f'Primary_Cluster_{i}'] = {
                    'size': cluster_size,
                    'subclusters': np.unique(sub_labels, return_counts=True)[1]
                }
            else:
                secondary_labels[mask] = i
                subcluster_info[f'Primary_Cluster_{i}'] = {
                    'size': cluster_size,
                    'subclusters': np.array([cluster_size])
                }
        
        self.data['Secondary_Cluster'] = secondary_labels
        return self._analyze_hierarchical_clusters(primary_clusters, secondary_clusters, subcluster_info)

    def analyze_with_random_forest(self):
        """Comprehensive Random Forest analysis of PFAS clusters."""
        pfas_compounds = [col for col in self.numeric_columns if col.endswith('-VA')]
        X = self.data[pfas_compounds]
        y = self.data['Primary_Cluster']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        importance_df = pd.DataFrame({
            'compound': pfas_compounds,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        cv_scores = cross_val_score(rf_model, X, y, cv=5)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self._plot_random_forest_results(rf_model, X_test, importance_df)
        
        interactions = self._calculate_feature_interactions(rf_model, X, pfas_compounds)
        self._plot_feature_interactions(interactions, importance_df)
        
        return {
            'feature_importance': importance_df,
            'cv_scores': cv_scores,
            'accuracy': accuracy,
            'interactions': interactions,
            'model': rf_model
        }

    def _plot_cluster_metrics(self, n_components_range, metrics):
        """Plot clustering metrics."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.plot(n_components_range, metrics['aic'], 'bo-')
        plt.title('AIC Score vs. N Clusters')
        plt.xlabel('N Clusters')
        plt.ylabel('AIC Score')
        
        plt.subplot(132)
        plt.plot(n_components_range, metrics['bic'], 'ro-')
        plt.title('BIC Score vs. N Clusters')
        plt.xlabel('N Clusters')
        plt.ylabel('BIC Score')
        
        plt.subplot(133)
        plt.plot(n_components_range, metrics['silhouette'], 'go-')
        plt.title('Silhouette Score vs. N Clusters')
        plt.xlabel('N Clusters')
        plt.ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.show()

    def _analyze_hierarchical_clusters(self, primary_clusters, secondary_clusters, subcluster_info):
        """Analyze hierarchical clustering results."""
        primary_silhouette = silhouette_score(self.data_scaled, self.data['Primary_Cluster'])
        
        pfas_compounds = [col for col in self.numeric_columns if col.endswith('-VA')]
        primary_profiles = self.data.groupby('Primary_Cluster')[pfas_compounds].mean()
        
        plt.figure(figsize=(20, 10))
        
        plt.subplot(131)
        primary_sizes = self.data['Primary_Cluster'].value_counts().sort_index()
        sns.barplot(x=primary_sizes.index, y=primary_sizes.values)
        plt.title('Primary Cluster Sizes')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        
        plt.subplot(132)
        sns.heatmap(primary_profiles, annot=True, cmap='YlOrRd', fmt='.2f',
                   xticklabels=True, yticklabels=True)
        plt.title('Primary Cluster PFAS Profiles')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(133)
        actual_subclusters = []
        for i in range(primary_clusters):
            if f'Primary_Cluster_{i}' in subcluster_info:
                actual_subclusters.append(len(subcluster_info[f'Primary_Cluster_{i}']['subclusters']))
            else:
                actual_subclusters.append(0)
        
        plt.bar(range(len(actual_subclusters)), actual_subclusters)
        plt.title('Number of Subclusters per Primary Cluster')
        plt.xlabel('Primary Cluster')
        plt.ylabel('Number of Subclusters')
        
        plt.tight_layout()
        plt.show()
        
        self._plot_hierarchical_lle()
        
        return {
            'primary_silhouette': primary_silhouette,
            'primary_profiles': primary_profiles,
            'subcluster_info': subcluster_info,
            'primary_sizes': primary_sizes.to_dict()
        }

    def _plot_hierarchical_lle(self):
        """Plot LLE visualization of hierarchical clusters."""
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
        lle_components = lle.fit_transform(self.data_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.scatterplot(
            x=lle_components[:, 0],
            y=lle_components[:, 1],
            hue=self.data['Primary_Cluster'],
            palette='deep',
            ax=ax1
        )
        ax1.set_title('Primary Clusters')
        
        sns.scatterplot(
            x=lle_components[:, 0],
            y=lle_components[:, 1],
            hue=self.data['Secondary_Cluster'],
            palette='tab20',
            ax=ax2
        )
        ax2.set_title('Secondary Clusters')
        
        plt.tight_layout()
        plt.show()

    def _plot_random_forest_results(self, rf_model, X_test, importance_df):
        """Plot Random Forest analysis results."""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(211)
        sns.barplot(x='importance', y='compound', data=importance_df.head(10))
        plt.title('Top 10 Most Important PFAS Compounds for Cluster Prediction')
        plt.xlabel('Importance Score')
        
        plt.subplot(212)
        pred_probs = rf_model.predict_proba(X_test)
        max_probs = np.max(pred_probs, axis=1)
        sns.histplot(max_probs, bins=30)
        plt.title('Distribution of Prediction Confidence')
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

    def _calculate_feature_interactions(self, model, X, feature_names):
        """Calculate feature interactions."""
        feature_importances = pd.DataFrame(
            model.feature_importances_,
            index=feature_names,
            columns=['importance']
        ).sort_values('importance', ascending=False)
            
        top_features = feature_importances.head(10).index
        n_top = len(top_features)
        interactions = np.zeros((n_top, n_top))
    
        for tree in model.estimators_:
            for path in self._generate_tree_paths(tree.tree_):
                features_in_path = set()
                for node in path:
                    if node[0] != -2:
                       feature = tree.tree_.feature[node[0]]
                       if feature_names[feature] in top_features:
                           features_in_path.add(feature_names[feature])
            
                features_list = list(features_in_path)
                for i in range(len(features_list)):
                    for j in range(i+1, len(features_list)):
                        if features_list[i] in top_features and features_list[j] in top_features:
                            idx1 = np.where(top_features == features_list[i])[0][0]
                            idx2 = np.where(top_features == features_list[j])[0][0]
                            interactions[idx1, idx2] += 1
                            interactions[idx2, idx1] += 1
    
        return pd.DataFrame(interactions, index=top_features, columns=top_features)

    def _generate_tree_paths(self, tree, node_id=0, path=None):
        """Generate paths through trees."""
        if path is None:
            path = []
    
        if tree.feature[node_id] != -2:
            paths = []
            paths_left = self._generate_tree_paths(tree, tree.children_left[node_id], 
                                             path + [(node_id, "left")])
            paths_right = self._generate_tree_paths(tree, tree.children_right[node_id], 
                                              path + [(node_id, "right")])
            paths.extend(paths_left)
            paths.extend(paths_right)
            return paths
        else:
            return [path]
        
    def _plot_feature_interactions(self, interactions, importance_df):
            """Plot feature interactions heatmap."""
            plt.figure(figsize=(12, 8))
            sns.heatmap(interactions, 
                annot=True, 
                cmap='YlOrRd',
                xticklabels=importance_df['compound'].head(10),
                yticklabels=importance_df['compound'].head(10))
            plt.title('Top 10 PFAS Compound Interactions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
# Initialize analyzer
analyzer = CompletePFASAnalyzer('PFAs.csv', 'PFAS_Data_Dictionary.csv')

# 1. Analyze Data Distributions
print("Analyzing PFAS compound distributions...")
distribution_stats = analyzer.analyze_distributions()
print("\nDistribution Statistics:")
print(distribution_stats.head())

# 2. Analyze Correlations
print("\nAnalyzing correlations between PFAS compounds...")
corr_matrix, high_corr_pairs = analyzer.analyze_correlations()
print("\nHighly correlated PFAS pairs (correlation > 0.8):")
for compound1, compound2, corr in high_corr_pairs:
    print(f"{compound1} - {compound2}: {corr:.3f}")

# 3. Find Optimal Number of Clusters
print("\nFinding optimal number of clusters...")
cluster_metrics = analyzer.find_optimal_clusters(max_clusters=15)

# 4. Perform Hierarchical Clustering
print("\nPerforming hierarchical clustering analysis...")
cluster_results = analyzer.hierarchical_cluster_analysis(primary_clusters=4, secondary_clusters=8)

print("\nPrimary Cluster Silhouette Score:", cluster_results['primary_silhouette'])
print("\nCluster Sizes:")
for cluster, size in cluster_results['primary_sizes'].items():
    print(f"Cluster {cluster}: {size} samples")

print("\nPrimary Cluster PFAS Profiles:")
print(cluster_results['primary_profiles'])

# 5. Random Forest Analysis
print("\nPerforming Random Forest analysis...")
rf_results = analyzer.analyze_with_random_forest()

print("\nRandom Forest Results:")
print(f"Model Accuracy: {rf_results['accuracy']:.3f}")
print("\nCross-validation scores:", rf_results['cv_scores'])
print(f"Mean CV Score: {rf_results['cv_scores'].mean():.3f} (+/- {rf_results['cv_scores'].std() * 2:.3f})")

print("\nTop 10 Most Important PFAS Compounds:")
print(rf_results['feature_importance'].head(10))

# Save results
print("\nSaving results...")
# Save cluster assignments
analyzer.data.to_csv('pfas_clustered_data.csv', index=False)

# Save feature importance
rf_results['feature_importance'].to_csv('pfas_feature_importance.csv', index=False)

# Save cluster profiles
cluster_results['primary_profiles'].to_csv('pfas_cluster_profiles.csv')

print("\nAnalysis complete. Results have been saved to CSV files.")