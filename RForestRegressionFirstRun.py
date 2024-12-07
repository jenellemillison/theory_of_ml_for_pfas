# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:25:55 2024

@author: hotco
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 05:36:33 2024

@author: hotco
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

def process_censored_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle censored values marked with '<'"""
    for col in df.select_dtypes(include=['object']).columns:
        if col.endswith('-VA'):
            remark_col = col.replace('-VA', '-RMK')
            if remark_col in df.columns:
                mask = df[remark_col] == '<'
                values = pd.to_numeric(df[col], errors='coerce')
                df[col] = np.where(mask, values/2, values)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features with temporal and spatial components"""
    # Convert date to cyclical features
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Extract site information
    df['site'] = df['NAWQA_ID'].str[:8]
    
    return df

class GroupRandomizedSearchCV:
    """Custom RandomizedSearchCV that respects group structure"""
    def __init__(self, estimator, param_distributions, n_iter, groups, n_splits=5):
        self.base_estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.groups = groups
        self.n_splits = n_splits
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.cv_results_ = []
        
    def fit(self, X, y):
        group_kfold = GroupKFold(n_splits=self.n_splits)
        
        for iteration in range(self.n_iter):
            print(f"\nIteration {iteration + 1}/{self.n_iter}")
            
            # Sample parameters
            params = {k: v.rvs() if hasattr(v, 'rvs') else np.random.choice(v) 
                     for k, v in self.param_distributions.items()}
            
            # Initialize model with sampled parameters
            model = RandomForestRegressor(**params, random_state=42)
            
            # Perform group k-fold CV
            fold_scores = []
            fold_predictions = []
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, self.groups)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_squared_error(y_val, y_pred, squared=False)
                
                fold_scores.append({
                    'r2': score,
                    'rmse': rmse,
                    'mae': mae
                })
                fold_predictions.append({
                    'true': y_val.values,
                    'pred': y_pred
                })
            
            # Calculate mean and std of scores
            mean_scores = {metric: np.mean([s[metric] for s in fold_scores]) 
                         for metric in fold_scores[0].keys()}
            std_scores = {metric: np.std([s[metric] for s in fold_scores]) 
                        for metric in fold_scores[0].keys()}
            
            # Store detailed results
            result = {
                'iteration': iteration,
                'params': params,
                'mean_scores': mean_scores,
                'std_scores': std_scores,
                'fold_scores': fold_scores,
                'fold_predictions': fold_predictions
            }
            
            self.cv_results_.append(result)
            
            # Update best parameters if necessary
            if mean_scores['r2'] > self.best_score_:
                self.best_score_ = mean_scores['r2']
                self.best_params_ = params
            
            # Print current iteration results
            print(f"Parameters: {params}")
            for metric, value in mean_scores.items():
                print(f"{metric}: {value:.3f} ± {std_scores[metric]:.3f}")
        
        # Sort results by mean R² score
        self.cv_results_ = sorted(self.cv_results_, 
                                key=lambda x: x['mean_scores']['r2'], 
                                reverse=True)
        return self

def train_spatial_temporal_rf(
    X: pd.DataFrame, 
    y: pd.Series, 
    groups: pd.Series, 
    n_splits: int = 5,
    scale_features: bool = True,
    tune_hyperparameters: bool = True,
    n_iter_search: int = 20
) -> Tuple[List[Dict], RandomForestRegressor]:
    """Train RF with spatial-temporal cross-validation and optional parameter tuning"""
    scaler = StandardScaler() if scale_features else None
    
    if scale_features:
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(10, 50)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None] + [0.3, 0.5, 0.7],
        'bootstrap': [True, False]
    }
    
    if tune_hyperparameters:
        print("Starting hyperparameter tuning...")
        group_random_search = GroupRandomizedSearchCV(
            estimator=RandomForestRegressor,
            param_distributions=param_distributions,
            n_iter=n_iter_search,
            groups=groups,
            n_splits=n_splits
        )
        group_random_search.fit(X, y)
        
        best_params = group_random_search.best_params_
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    else:
        final_model = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    
    # Perform final cross-validation with best model
    group_kfold = GroupKFold(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        
        scores.append({
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_squared_error(y_test, y_pred, squared=False)
        })
    
    # Train final model on all data
    final_model.fit(X, y)
    
    return scores, final_model

def analyze_model_variations(X, y, groups, n_runs=10):
    """Analyze variation in model performance and feature importance across multiple runs"""
    all_scores = []
    all_importances = []
    
    print(f"Running {n_runs} iterations...\n")
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        cv_scores, final_model = train_spatial_temporal_rf(
            X, y, groups,
            scale_features=True,
            tune_hyperparameters=True,
            n_iter_search=20
        )
        
        # Store results
        mean_scores = {metric: np.mean([score[metric] for score in cv_scores]) 
                      for metric in cv_scores[0].keys()}
        
        # Filter out temporal features from importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': final_model.feature_importances_
        })
        importance_df = importance_df[~importance_df['feature'].isin(['day_sin', 'day_cos'])]
        
        all_scores.append(mean_scores)
        all_importances.append(importance_df)
    
    # Create scores DataFrame
    scores_df = pd.DataFrame(all_scores)
    
    # Calculate feature importance variations (excluding temporal features)
    importance_variations = {}
    for feature in X.columns:
        if feature not in ['day_sin', 'day_cos']:
            importances = [df.loc[df['feature'] == feature, 'importance'].iloc[0] 
                         for df in all_importances]
            importance_variations[feature] = {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'min': np.min(importances),
                'max': np.max(importances)
            }
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Performance metrics distribution
    plt.subplot(2, 1, 1)
    scores_df.boxplot()
    plt.title('Distribution of Performance Metrics Across Runs')
    plt.ylabel('Score')
    
    # Feature importance variations
    plt.subplot(2, 1, 2)
    importance_df = pd.DataFrame(importance_variations).T
    feature_order = importance_df['mean'].sort_values(ascending=False).index
    
    # Plot mean importance with error bars
    plt.errorbar(
        x=importance_df.loc[feature_order, 'mean'],
        y=range(len(feature_order)),
        xerr=importance_df.loc[feature_order, 'std'],
        fmt='o',
        capsize=5
    )
    plt.yticks(range(len(feature_order)), feature_order)
    plt.title('Feature Importance Variations')
    plt.xlabel('Mean Importance (with std dev)')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nPerformance Metrics Statistics:")
    print(scores_df.describe())
    
    print("\nFeature Importance Statistics:")
    print("Top 5 most variable features (by coefficient of variation):")
    importance_df['cv'] = importance_df['std'] / importance_df['mean']
    print(importance_df.sort_values('cv', ascending=False).head())
    
    return scores_df, importance_variations

def main():
    # Load and process data
    train_data = pd.read_csv('train_PFAS_ENV.csv')
    test_data = pd.read_csv('test_PFAS_ENV.csv')
    data = pd.concat([train_data, test_data])
    
    # Process data
    data = process_censored_data(data)
    data = prepare_features(data)
    
    # Select features and target (excluding temporal features)
    pfas_cols = [col for col in data.columns if col.endswith('-VA')]
    feature_cols = pfas_cols[:-1]  # Use all but one PFAS
    target_col = pfas_cols[-1]
    
    X = data[feature_cols]
    y = data[target_col]
    groups = data['site']
    
    # Run variation analysis
    scores_df, importance_variations = analyze_model_variations(X, y, groups, n_runs=10)

if __name__ == "__main__":
    main()
    
    
#scores_df, importance_variations = analyze_model_variations(X, y, groups, n_runs=10)