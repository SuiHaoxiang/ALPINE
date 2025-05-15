import csv
import os
from datetime import datetime

def save_metrics_to_csv(model_name, metrics):
    """将评估指标保存到CSV文件"""
    # 确保logs目录存在
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{model_name}_metrics.csv"
    filepath = os.path.join(log_dir, filename)
    
    # CSV文件头
    fieldnames = [
        'timestamp', 'model', 'accuracy', 'precision', 
        'recall', 'f1'
    ]
    
    # 准备数据行
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': model_name,
        'accuracy': metrics.get("Accuracy", 0),
        'precision': metrics.get("Precision", 0),
        'recall': metrics.get("Recall", 0),
        'f1': metrics.get("F1", 0),
       
    }
    
    # 写入CSV文件
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"metrics_log已保存到: {filepath}")
