import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import yaml
import os
import pickle

def load_config():
    """Load configuration file"""
    with open('/home/code/lstm_if_one_dim/final/src/config.yaml') as f:
        config = yaml.safe_load(f)
    return config

def load_train_data(train_path):
    """Load training set"""
    print(f"Loading training set: {train_path}")
    df = pd.read_csv(train_path)
    # Training set may not have labels, fill with 0
    if 'label' not in df.columns:
        df['label'] = 0
    return df

def load_test_data(test_path):
    """Load test set""" 
    print(f"Loading test set: {test_path}")
    df = pd.read_csv(test_path)
    # Ensure test set has label column
    if 'label' not in df.columns:
        raise ValueError("Test set must contain 'label' column")
    return df

def preprocess_data(train_data, test_data):
    """Data preprocessing"""
    # Automatically get numeric feature columns (exclude label and non-numeric columns)
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != 'label']
    
    if not features:
        raise ValueError("No numeric feature columns found")
    
    # Separate features and labels
    X_train = train_data[features].values.astype(np.float64)
    y_train = train_data['label'].values
    X_test = test_data[features].values.astype(np.float64)
    y_test = test_data['label'].values
    
    # Check and handle NaN values
    if np.isnan(X_train.astype(np.float64)).any():
        print("Warning: Training data contains NaN values, will remove rows with NaN")
        # Get indices of non-NaN rows
        non_nan_idx = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[non_nan_idx]
        y_train = y_train[non_nan_idx]
        
    if np.isnan(X_test).any():
        print("Warning: Test data contains NaN values, will remove rows with NaN")
        # Get indices of non-NaN rows
        non_nan_idx = ~np.isnan(X_test).any(axis=1)
        X_test = X_test[non_nan_idx]
        y_test = y_test[non_nan_idx]
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def train_ocsvm(X_train, nu, kernel='rbf', gamma='scale'):
    """Train One-Class SVM model"""
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train)
    
    # Save trained model
    model_path = '/home/code/lstm_if_one_dim/final/src/svm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    # Convert One-Class SVM output (-1,1) to (0,1)
    y_pred = np.where(y_pred == -1, 1, 0)  # -1 means anomaly, convert to 1
    y_true = y_test
    
    # Ensure test set has anomaly samples
    if np.sum(y_true) == 0:
        print("Warning: No anomaly samples found in test set")
        return y_pred
    
    # Calculate and print full classification report
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Calculate and print anomaly class metrics
    try:
        report = classification_report(y_true, y_pred, output_dict=True, digits=4)
        if '1.0' in report:
            anomaly_report = report['1.0']
            print("\nAnomaly detection performance:")
            print(f"Precision: {anomaly_report['precision']:.4f}")
            print(f"Recall: {anomaly_report['recall']:.4f}")
            print(f"F1-score: {anomaly_report['f1-score']:.4f}")
    except:
        print("Cannot calculate anomaly class metrics")
    
    # Calculate AUC-ROC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"\nAUC-ROC: {auc:.4f}")
    except:
        print("\nAUC-ROC: Requires both positive and negative samples")
    
    return y_pred


def main():
    # Load configuration
    config = load_config()
    
    # Load training and test sets
    train_data = load_train_data(config['shared']['data']['train_data'])
    test_data = load_test_data(config['shared']['data']['test_data'])
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
    
    # Train model
    model = train_ocsvm(X_train, 
                       nu=0.005,
                       kernel=config.get('svm_kernel', 'rbf'),
                       gamma=config.get('svm_gamma', 'scale'))
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
   
if __name__ == "__main__":
    main()
