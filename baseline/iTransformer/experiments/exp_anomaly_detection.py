from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import numpy as np

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        # Modify output layer for binary classification
        in_features = model.projector.in_features if hasattr(model.projector, 'in_features') else model.projector.weight.shape[1]
        model.projector = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()  # Use MSE loss for sequence reconstruction

    def train(self, setting):
        import torch.nn.functional as F
        train_data, train_loader = self._get_data(flag='train')
        # For anomaly detection, validation set should also use normal data
        vali_data, vali_loader = self._get_data(flag='train')  
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, None, None)
                if self.args.is_training == 1:
                    # Use MSE reconstruction loss (normal data only)
                    loss = F.mse_loss(outputs, batch_x)
                else:
                    # Only use anomaly data during testing
                    recon_error = torch.mean(F.mse_loss(outputs, batch_x, reduction='none'), dim=[1,2])
                    # Use reconstruction error as anomaly score
                    anomaly_scores = recon_error.detach().cpu().numpy()
                    # Calculate evaluation metrics
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(batch_y.cpu().numpy(), anomaly_scores)
                    print(f'Test AUC: {auc:.4f}')
                    loss = torch.tensor(auc)  # Use AUC as evaluation metric
                train_loss.append(loss.item())
                
                loss.backward()
                optimizer.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Vali Loss: {vali_loss:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, None, None)
                loss = criterion(outputs, batch_x)  # Compare reconstruction output with original input
                total_loss.append(loss.item())
                
        return np.average(total_loss)

    def test(self, setting):
        import time
        test_data, test_loader = self._get_data(flag='test')
        threshold = getattr(self.args, 'anomaly_threshold', 0.4)
        print(f"\nUsing anomaly threshold: {threshold:.4f} (set via --anomaly_threshold)")
        
        # Start timing
        start_time = time.time()
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, None, None)
                # Get reconstruction error as anomaly score
                recon_error = torch.mean(F.mse_loss(outputs, batch_x, reduction='none'), dim=[1,2])
                pred = recon_error.cpu().numpy()  # [batch_size]
                true = batch_y.squeeze().cpu().numpy()  # [batch_size]
                
                preds.append(pred)
                trues.append(true)
        
        # Calculate evaluation metrics
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        y_true = np.hstack(trues) if len(trues) > 0 else np.array([])
        y_pred = np.hstack(preds) if len(preds) > 0 else np.array([])
        
        # Force binary classification mode, using threshold from args (default 0.5)
        threshold = getattr(self.args, 'anomaly_threshold', 0.4)
        y_pred_binary = y_pred > threshold
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        print("\n=== Anomaly Detection Metrics ===")
        print("="*40)
        print(f"=== CURRENT ANOMALY THRESHOLD: {threshold:.4f} ===")
        print("="*40)
        print(f"Threshold: {threshold:.4f} (adjust with --anomaly_threshold)")
        print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Regression Metrics - mse:{np.mean(y_pred):.4f}, mae:{np.median(y_pred):.4f}")
        print("===============================")
        # Calculate inference time per sample (ms)
        sample_count = len(y_true)
        total_time_ms = (time.time() - start_time) * 1000
        per_sample_time = total_time_ms / sample_count if sample_count > 0 else 0
        
        print(f"\n=== Inference time statistics ===")
        print(f"Inference time per sample: {per_sample_time:.6f}ms")
        print("==================")
        
        return {
            'AUC': auc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'InferenceTime': total_time_ms
        }
