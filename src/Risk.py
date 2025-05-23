import numpy as np

class RiskCalculator:
    """Integrated risk calculator combining channel risk and semantic risk"""
    
    def __init__(self, gamma1=0.9, gamma2=0.1):
        """Initialize risk calculator
        
        Args:
            gamma1 (float): Channel risk weight (default 0.5)
            gamma2 (float): Semantic risk weight (default 0.5)
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    
    def sigmoid(self, x):
        """Sigmoid normalization function
        
        Args:
            x (float): Input value
            
        Returns:
            float: Normalized value (0-1)
        """
        return 1 / (1 + np.exp(-x))
    
    def calculate_combined_risk(self, r_channel, r_semantic):
        """Calculate combined risk score R_risk = σ(γ1*R_channel + γ2*R_semantic)
        
        Args:
            r_channel (float): Channel risk value
            r_semantic (float): Semantic risk value
            
        Returns:
            float: Combined risk score (0-1)
        """
        # Calculate weighted sum and apply sigmoid normalization
        weighted_sum = self.gamma1 * r_channel + self.gamma2 * r_semantic
        return self.sigmoid(weighted_sum)
    
    def update_weights(self, gamma1, gamma2):
        """Update weight parameters
        
        Args:
            gamma1 (float): New channel risk weight
            gamma2 (float): New semantic risk weight
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
