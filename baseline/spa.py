import pandas as pd
import torch
import pickle
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler  # Additional import

sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics_csv import save_metrics_to_csv

# Load configuration parameters
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    spa_config = config["spa"]
    shared_config = config["shared"]

# Load training data (with standardization)
train_data = pd.read_csv(shared_config["data"]["train_data"])
# Start standardization process
scaler = StandardScaler()
train_features = train_data[shared_config["features"]]
train_data[shared_config["features"]] = scaler.fit_transform(train_features)  # Standardize training set
X_train = torch.tensor(train_data[shared_config["features"]].values, dtype=torch.float32)

# Load test data (apply same standardization)
test_data = pd.read_csv(shared_config["data"]["test_data"])
test_features = test_data[shared_config["features"]]
test_data[shared_config["features"]] = scaler.transform(test_features)  # Apply same transform to test set
X_test = torch.tensor(test_data[shared_config["features"]].values, dtype=torch.float32)
true_labels = test_data['label'].values

# Use part of training data as validation set
val_size = int(0.15 * len(X_train))
X_val = X_train[-val_size:]
X_train = X_train[:-val_size]


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")


# Define Isolation Tree Node
class IsolationTreeNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.size = 0  # Number of samples in node


# Define Isolation Tree
class IsolationTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, features):
        self.root = self._build_tree(X, depth=0, features=features)

    def _build_tree(self, X, depth, features):
        node = IsolationTreeNode(depth=depth)
        node.size = X.shape[0]

        # Stop condition: reach max depth or insufficient samples
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            return node

        # Randomly select feature and threshold
        if len(features) == 0:
            return node
        split_feature_idx = np.random.choice(len(features))
        split_feature = features[split_feature_idx]
        feature_values = X[:, split_feature]
        min_val, max_val = torch.min(feature_values), torch.max(feature_values)
        if min_val == max_val:
            return node
        split_threshold = torch.rand(1).item() * (max_val - min_val).item() + min_val.item()

        # Split data
        left_idx = feature_values < split_threshold
        right_idx = ~left_idx

        # Stop if split is invalid
        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return node

        # Recursively build subtrees
        node.split_feature = split_feature
        node.split_threshold = split_threshold
        node.left = self._build_tree(X[left_idx], depth + 1, features)
        node.right = self._build_tree(X[right_idx], depth + 1, features)

        return node

    def path_length(self, x):
        path_len = 0
        node = self.root
        while node.left or node.right:
            if node.split_feature is None:
                break
            path_len += 1
            if x[node.split_feature] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return path_len + self._avg_external_path_length(node.size)

    @staticmethod
    def _avg_external_path_length(n):
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


# Define Sparse-iForest
class SparseIsolationForest:
    def __init__(self, n_trees=50, max_depth=10, feature_subset_size=None,
                 min_samples_split=2, subsample_size=256):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.feature_subset_size = feature_subset_size
        self.min_samples_split = min_samples_split
        self.subsample_size = subsample_size
        self.trees = []
        self.feature_names = None  # New: store feature names

    def fit(self, X, feature_names=None):
        n_samples, n_features = X.shape
        if self.feature_subset_size is None:
            self.feature_subset_size = max(1, n_features // 2)  # At least 1 feature
        self.trees = []
        self.feature_names = feature_names  # Store feature names
        for _ in range(self.n_trees):
            # Random subsampling
            idx = torch.randint(0, n_samples, (self.subsample_size,))
            X_subsample = X[idx]
            # Randomly select feature subset
            features = np.random.choice(n_features, self.feature_subset_size, replace=False)
            # Build tree
            tree = IsolationTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subsample, features.tolist())
            self.trees.append(tree)

    def anomaly_score(self, X):
        scores = []
        c_n = self._avg_path_length(self.subsample_size)
        for x in X:
            path_lengths = []
            for tree in self.trees:
                path_len = tree.path_length(x)
                path_lengths.append(path_len)
            mean_path = np.mean(path_lengths)
            score = 2 ** (-mean_path / c_n)
            scores.append(score)
        return torch.tensor(scores)

    def feature_wise_anomaly_scores(self, X):
        """Calculate anomaly scores for each feature dimension separately"""
        if self.feature_names is None:
            raise ValueError("Feature names not set. Call fit() with feature_names first.")

        feature_scores = {name: [] for name in self.feature_names}
        c_n = self._avg_path_length(self.subsample_size)

        for x in X:
            # Initialize path length for each feature
            feature_paths = {name: [] for name in self.feature_names}

            # Collect path lengths for each feature
            for tree in self.trees:
                if tree.root.split_feature is None:
                    continue
                feature_idx = tree.root.split_feature
                feature_name = self.feature_names[feature_idx]
                path_len = tree.path_length(x)
                feature_paths[feature_name].append(path_len)

            # Calculate anomaly score for each feature
            for name in self.feature_names:
                if feature_paths[name]:
                    mean_path = np.mean(feature_paths[name])
                    score = 2 ** (-mean_path / c_n)
                    feature_scores[name].append(score)
                else:
                    feature_scores[name].append(0)  # Score 0 if no trees used this feature

        return {name: torch.tensor(scores) for name, scores in feature_scores.items()}

    @staticmethod
    def _avg_path_length(n):
        if n <= 1:
            return 1
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


# Training and prediction
if __name__ == "__main__":
    # Initialize model
    model = SparseIsolationForest(
        n_trees=spa_config["n_trees"],
        max_depth=spa_config["max_depth"],
        feature_subset_size=1,  # Fixed to 3 features, take 1
        min_samples_split=spa_config["min_samples_split"],
        subsample_size=spa_config["subsample_size"]
    )
    model.fit(X_train, feature_names=shared_config["features"])
    save_model(model, "sparse_isolation_forest_model.pkl")

    # === Change 1: Use comprehensive anomaly score for threshold calculation ===
    # Calculate validation set comprehensive anomaly scores

    val_scores = model.anomaly_score(X_val)
    
    # Debug output score range
    print(f"Validation score range: min={val_scores.min().item():.4f}, max={val_scores.max().item():.4f}")
    
    # Calculate global threshold
    global_threshold = 1.0*np.percentile(val_scores.numpy(),95
                                         )
    global_threshold_1 =1.2*global_threshold
     # Calculate normal threshold
    normal_threshold = 1*np.percentile(val_scores.numpy(), 20)
    print("global_threshold:", global_threshold)
    print("normal_threshold:", normal_threshold)
    
    # Calculate percentile for threshold 0.45
    target_threshold = 0.45
    percentile = np.mean(val_scores.numpy() <= target_threshold) * 100
    print(f"Percentile for threshold {target_threshold}: {percentile:.2f}%")
    
    # Save thresholds to config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["spa"]["GLOBAL_original_THRESHOLD"] = float(global_threshold)
    config["spa"]["GLOBAL_THRESHOLD"] = float(global_threshold_1)
    config["spa"]["NORMAL_THRESHOLD"] = float(normal_threshold)
    with open("config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # === Change 2: Use comprehensive score for test set ===
    # Calculate test set comprehensive anomaly scores
    test_scores = model.anomaly_score(X_test)
    print(f"Test score range: min={test_scores.min().item():.4f}, max={test_scores.max().item():.4f}")
    
    # Generate predictions
    final_predictions = (test_scores.numpy() > global_threshold).astype(int)
    true_labels = true_labels[:len(X_test)]  # Ensure label alignment

    # === Change 3: Evaluation metrics calculation ===
    accuracy = accuracy_score(true_labels, final_predictions)
    recall = recall_score(true_labels, final_predictions)
    precision = precision_score(true_labels, final_predictions)
    f1 = f1_score(true_labels, final_predictions)

    print(f"Global threshold: {global_threshold:.4f}")
    print(f"Test set anomaly count: {final_predictions.sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save metrics
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    save_metrics_to_csv("Sparse-iForest", metrics)

    # Print confusion matrix
    cm = confusion_matrix(true_labels, final_predictions)
    print("\nConfusion matrix:")
    print("          Predicted")
    print("          Normal Anomaly")
    print(f"Actual Normal  {cm[0,0]:<6} {cm[0,1]:<6}")
    print(f"       Anomaly {cm[1,0]:<6} {cm[1,1]:<6}")
