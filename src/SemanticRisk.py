import numpy as np
from collections import defaultdict
from math import log2

class SemanticRiskCalculator:
    def __init__(self, w1=0.5, w2=0.5, window_size=5):
        
        self.w1 = w1
        self.w2 = w2
        self.window_size = window_size
        
        # 定义数据敏感度映射 (可根据GDPR标准扩展)
        self.data_sensitivity_map = {
            'temperature': 0.3,  # 环境数据 - 低敏感度
            'humidity': 0.3,     # 环境数据 - 低敏感度
            'location': 1.0,     # 位置数据 - 高敏感度
            'health': 0.8        # 健康数据 - 高敏感度
        }
        
        # 历史数据存储(滑动窗口)
        self.history_data = defaultdict(lambda: {'window': [], 'count': 0})
        
        # 默认权重
        self.w1 = w1
        self.w2 = w2
    
    def calculate_data_sensitivity(self, field_name):
        return self.data_sensitivity_map.get(field_name, 0.5)  # 默认为中等敏感度

    def calculate_entropy(self, data, field_name=None, update_history=False):
        """计算数据的信息熵(支持滑动窗口)
        信息熵公式: H(X) = -Σ p(x_i) * log2(p(x_i))
        
        Args:
            data: 输入数据(单值或数组)
            field_name: 字段名(用于窗口计算)
            update_history: 是否更新历史数据
        Returns:
            float: 归一化后的信息熵(0-1)
        """
        if field_name and update_history:
            # 更新滑动窗口
            if isinstance(data, (list, np.ndarray)):
                new_data = list(data)
            else:
                new_data = [data]
                
            window_data = self.history_data[field_name]
            window_data['window'].extend(new_data)
            window_data['count'] += len(new_data)
            
            # 维护窗口大小
            if len(window_data['window']) > self.window_size:
                remove_count = len(window_data['window']) - self.window_size
                window_data['window'] = window_data['window'][remove_count:]
                window_data['count'] = self.window_size
                
        # 获取计算数据
        calc_data = []
        if field_name:
            window_data = self.history_data[field_name]
            if window_data['count'] > 0:
                calc_data = window_data['window']
        else:
            calc_data = data if isinstance(data, (list, np.ndarray)) else [data]
            
        # 处理空数据情况
        if not calc_data:
            return 0.0
            
        # 计算熵值
        value_counts = defaultdict(int)
        total = len(calc_data)
        for value in calc_data:
            value_counts[value] += 1
            
        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2(p)
                
        # 直接返回原始熵值
        return max(0.0, entropy)  # 确保不小于0
    
    def calculate_context_risk(self, fields, data):
        """计算上下文风险
        Args:
            fields (list): 字段名称列表
            data (DataFrame/Series): 包含字段数据的数据框或单行数据
        Returns:
            float: 上下文风险分数
        """
        total_risk = 0.0
        n = len(fields)
        
        for i in range(n):
            field_i = fields[i]
            # 模拟指示函数I: 80%概率为1，20%概率为0
            import random
            I = 1 if random.random() < 0.8 else 0
            
            # 处理单值和数组两种情况
            field_data = data[field_i]
            H = self.calculate_entropy(field_data)  # 统一使用calculate_entropy方法
                
            total_risk += I * H
            
        return total_risk / n if n > 0 else 0.0
    
    def calculate_semantic_risk(self, fields, data):
        """计算语义风险评分 R_sem = w1*DataSens + w2*ContextRisk
        Args:
            fields (list): 字段名称列表
            data (DataFrame): 包含所有字段数据的数据框
        Returns:
            float: 语义风险评分
        """
        # 计算平均数据敏感度
        data_sens = np.mean([self.calculate_data_sensitivity(f) for f in fields])
        
        # 计算上下文风险
        context_risk = self.calculate_context_risk(fields, data)
        
        # 计算综合语义风险
        return self.w1 * data_sens + self.w2 * context_risk

    def set_weights(self, w1, w2):
        """设置权重参数
        Args:
            w1 (float): 数据敏感度权重
            w2 (float): 上下文风险权重
        """
        self.w1 = w1
        self.w2 = w2
        
    def calculate(self, df, use_columns=None):
        
        import pandas as pd
        
        
        
        # 确定要使用的列
        if use_columns is None:
            use_columns = df.columns[:2]  # 默认使用前两列
            
        # 计算每行的语义风险
        results = {}
        for idx, row in df.iterrows():
            data_sens = np.mean([self.calculate_data_sensitivity(col) for col in use_columns])
            # 更新历史数据并计算上下文风险
            for col in use_columns:
                self.calculate_entropy(row[col], col, update_history=True)
            
            # 使用历史数据计算上下文风险
            context_risk = 0.0
            for col in use_columns:
                # 模拟指示函数I: 80%概率为1，20%概率为0
                import random
                I = 1 if random.random() < 0.8 else 0
                H = self.calculate_entropy(None, col, False)  # 使用历史数据计算熵
                context_risk += I * H
            context_risk /= len(use_columns)
            r_sem = self.w1 * data_sens + self.w2 * context_risk
            results[idx] = {
                'data_sensitivity': data_sens,
                'context_risk': context_risk,
                'semantic_risk': r_sem
            }
            
        return results

    def add_sensitivity_mapping(self, field_name, sensitivity):
        """添加自定义数据敏感度映射
        Args:
            field_name (str): 字段名称
            sensitivity (float): 敏感度值 (0-1)
        """
        self.data_sensitivity_map[field_name] = sensitivity
