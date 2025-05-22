import numpy as np

class RiskCalculator:
    """综合风险计算器，结合信道风险和语义风险"""
    
    def __init__(self, gamma1=0.9, gamma2=0.1):
        """初始化风险计算器
        
        Args:
            gamma1 (float): 信道风险权重 (默认0.5)
            gamma2 (float): 语义风险权重 (默认0.5)
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def sigmoid(self, x):
        """Sigmoid归一化函数
        
        Args:
            x (float): 输入值
            
        Returns:
            float: 归一化后的值(0-1)
        """
        return 1 / (1 + np.exp(-x))
    
    def calculate_combined_risk(self, r_channel, r_semantic):
        """计算综合风险评分 R_risk = σ(γ1*R_channel + γ2*R_semantic)
        
        Args:
            r_channel (float): 信道风险值
            r_semantic (float): 语义风险值
            
        Returns:
            float: 综合风险评分(0-1)
        """
        # 计算加权和并应用Sigmoid归一化
        weighted_sum = self.gamma1 * r_channel + self.gamma2 * r_semantic
        return self.sigmoid(weighted_sum)
    
    def update_weights(self, gamma1, gamma2):
        """更新权重参数
        
        Args:
            gamma1 (float): 新的信道风险权重
            gamma2 (float): 新的语义风险权重
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
