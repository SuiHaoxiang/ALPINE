from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        # 先不初始化模型
        self.model = None

    def _init_model(self):
        """在获取数据后初始化模型"""
        if self.model is None:
            self.model = self._build_model()
            # 确保数据加载器在正确设备上
            if self.args.use_gpu:
                print(f"Moving model to device: {self.device}")
                self.model = self.model.to(self.device)

    def _build_model(self):
        from data_provider.data_factory import select_model
        model = select_model(self.args, self.args).float()
        if self.args.use_gpu:
            model = model.to(self.device)
            if self.args.use_multi_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f"Model initialized on device: {next(model.parameters()).device}")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        # 自动设置enc_in参数
        if 'SD' in self.args.data or 'FD' in self.args.data:
            sample = next(iter(data_loader))[0]
            if sample is not None:
                # 确保在模型构建前设置维度
                self.args.actual_enc_in = sample.shape[-1]  # 实际输入维度
                self.args.enc_in = sample.shape[-1]  # 兼容旧代码
                print(f"Detected input dimension: {sample.shape[-1]}")
        
        # 添加数据检查
        if flag == 'train':
            sample = next(iter(data_loader))
            #print(f"Data sample shape: {sample[0].shape if sample[0] is not None else 'None'}")
            #print(f"Data path: {self.args.root_path}{self.args.data_path}")
            
            if sample[0] is None:
                raise ValueError("Data loader returned None input!")
                
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # 确保模型和数据在相同设备上
                if str(next(self.model.parameters()).device) != str(batch_x.device):
                    self.model = self.model.to(batch_x.device)
                    print(f"Moving model to match data device: {batch_x.device}")

                # 测量单个样本推理时间
                sample_times = []
                for sample in batch_x:
                    start_time = time.perf_counter()
                    if 'SD' in self.args.data or 'FD' in self.args.data:
                        outputs = self.model(sample.unsqueeze(0))  # 3D模型只需要一个输入
                    else:
                        outputs = self.model(sample.unsqueeze(0), None, None, None)  # 原模型需要4个输入
                    end_time = time.perf_counter()
                    sample_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                # 计算并打印单个样本的平均推理时间
                avg_sample_time = sum(sample_times) / len(sample_times)
                print(f"Average inference time per sample: {avg_sample_time:.6e} ms")

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # 先获取数据并设置维度
        train_data, train_loader = self._get_data(flag='train')
        # 确保模型使用正确的维度初始化
        self._init_model()
        vali_data, vali_loader = self._get_data(flag='val')  # 使用训练集划分的验证集
        test_data, test_loader = self._get_data(flag='test')  # 保持测试集独立

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        print(f"Checkpoints will be saved to: {path}")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                if 'SD' in self.args.data or 'FD' in self.args.data:
                    outputs = self.model(batch_x)  # 3D模型只需要一个输入
                else:
                    outputs = self.model(batch_x, None, None, None)  # 原模型需要4个输入

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim,scheduler, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"Warning: No checkpoint found at {best_model_path}")

        return self.model

    def test(self, setting, test=0):
        # 记录推理开始时间
        inference_start_time = time.time()
        
        # 先获取数据并设置维度
        test_data, test_loader = self._get_data(flag='test')
        # 确保模型使用正确的维度初始化
        self._init_model()
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                if 'SD' in self.args.data or 'FD' in self.args.data:
                    outputs = self.model(batch_x)  # 3D模型只需要一个输入
                else:
                    outputs = self.model(batch_x, None, None, None)  # 原模型需要4个输入
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            if 'SD' in self.args.data or 'FD' in self.args.data:
                outputs = self.model(batch_x)  # 3D模型只需要一个输入
            else:
                outputs = self.model(batch_x, None, None, None)  # 原模型需要4个输入
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        print(f"Using anomaly_ratio: {self.args.anomaly_ratio}")
        threshold = np.percentile(combined_energy, 100 - (100 * self.args.anomaly_ratio))
        print(f"Calculated threshold ({100 - (100 * self.args.anomaly_ratio)}th percentile):", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import confusion_matrix
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        cm = confusion_matrix(gt, pred)
        print("Confusion Matrix:")
        print(cm)
        print("\nAccuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # 精确计算并打印推理时间(毫秒/样本)
        total_samples = len(test_loader.dataset)
        avg_time_ms = (time.time() - inference_start_time) * 1000 / total_samples
        print(f"\nInference Time Summary:")
        print(f"Total samples processed: {total_samples}")
        print(f"Average inference time: {avg_time_ms:.6f} ms per sample")
        return
