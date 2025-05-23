import csv
import os
from datetime import datetime

def save_metrics_to_csv(model_name, metrics):
    """Save evaluation metrics to CSV file"""
    # Ensure logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_name}_metrics.csv"
    filepath = os.path.join(log_dir, filename)
    
    # CSV file header
    fieldnames = [
        'timestamp', 'model', 'accuracy', 'precision', 
        'recall', 'f1'
    ]
    
    # Prepare data row
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'accuracy': metrics.get("Accuracy", 0),
        'precision': metrics.get("Precision", 0),
        'recall': metrics.get("Recall", 0),
        'f1': metrics.get("F1", 0),
       
    }
    
    # Write to CSV file
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"metrics_log saved in: {filepath}")
