import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
import yaml
import os
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, config):
    """Train autoencoder model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('ae_lr', 0.001))
    
    model.train()
    for epoch in range(config.get('ae_epochs', 100)):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Save trained model
    model_path = config.get('model_path', 'best_autoencoder_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model

def detect_anomalies(model, data_loader, threshold):
    """Detect anomalies"""
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data in data_loader:
            outputs = model(data)
            mse = torch.mean((outputs - data)**2, dim=1)
            anomalies.extend((mse > threshold).numpy())
    return np.array(anomalies)

def evaluate(y_true, error_scores, target_recall=None, quantile=0.60):
    """Evaluate model performance and select threshold
    Args:
        y_true: True labels
        error_scores: Error scores
        target_recall: Target recall rate
        quantile: Anomaly percentile (e.g. 0.60 means 60th percentile)
    """
    # Print error score statistics
    print("\nError score statistics:")
    print(f"Min: {np.min(error_scores):.4f}")
    print(f"25th percentile: {np.percentile(error_scores, 25):.4f}")
    print(f"Median: {np.median(error_scores):.4f}") 
    print(f"75th percentile: {np.percentile(error_scores, 75):.4f}")
    print(f"95th percentile: {np.percentile(error_scores, 95):.4f}")
    print(f"Max: {np.max(error_scores):.4f}")
    
    # If quantile is specified, calculate threshold directly
    if quantile is not None:
        threshold = np.percentile(error_scores, quantile * 100)
        preds = (error_scores > threshold).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"\nUsing {quantile*100:.0f}th percentile as threshold:")
        print(f"Threshold: {threshold:.4f}")
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        return threshold
        
    print("\nAnomaly detection performance:")
    
    # Calculate AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, error_scores)
    print(f"AUC-ROC: {auc(fpr, tpr):.4f}")
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, error_scores)
    
    # Find minimum threshold that meets target recall
    target_idx = np.where(recall >= target_recall)[0]
    if len(target_idx) > 0:
        best_idx = target_idx[0]
        best_threshold = thresholds[best_idx]
        best_precision = precision[best_idx]
        print(f"When recall >= {target_recall}:")
        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall: {recall[best_idx]:.4f}")
        preds = (error_scores > best_threshold).astype(int)
        f1 = f1_score(y_true, preds)
        print(f"F1-score: {f1:.4f}")
    else:
        print(f"Cannot achieve target recall {target_recall}")
    
    return best_threshold if 'best_threshold' in locals() else None

def load_data(file_path):
    """Load and preprocess data"""
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df

def main():
    # Load configuration
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Data loading and preprocessing
    train_data = load_data(config['shared']['data']['train_data'])
    test_data = load_data(config['shared']['data']['test_data'])
    
    # Preprocessing
    features = config['shared']['features']
    X_train = train_data[features].values
    X_test = test_data[features].values
    y_test = test_data['label'].values
    
    # Check if features exist
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        raise ValueError(f"Features specified in config not found in data: {missing_features}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)
    
    # Create model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim=config.get('ae_encoding_dim', 8))
    
    # Train model
    model = train_autoencoder(model, [train_tensor], config)
    
    # Calculate reconstruction errors on test set
    with torch.no_grad():
        test_recon = model(test_tensor)
        test_errors = torch.mean((test_recon - test_tensor)**2, dim=1).numpy()
    
    # Print current configuration
    print("\nCurrent configuration parameters:")
    print(f"ae_threshold_quantile: {config['autoencoder'].get('ae_threshold_quantile')}")
    
    # Evaluate and get best threshold
    best_threshold = evaluate(y_test, test_errors,
                            quantile=config['autoencoder'].get('ae_threshold_quantile'))
    
    if best_threshold is not None:
        # Detect anomalies using best threshold
        test_anomalies = (test_errors > best_threshold).astype(int)
        print("\nDetection results using best threshold:")
        print(f"Anomaly samples: {sum(test_anomalies)}")
        print(f"Normal samples: {len(test_anomalies) - sum(test_anomalies)}")

if __name__ == "__main__":
    main()
