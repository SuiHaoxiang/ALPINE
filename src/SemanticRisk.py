import numpy as np
from collections import defaultdict
from math import log2

class SemanticRiskCalculator:
    def __init__(self, w1=0.5, w2=0.5, window_size=5):
        
        self.w1 = w1
        self.w2 = w2
        self.window_size = window_size
        
        # Define data sensitivity mapping (can be extended per GDPR)
        self.data_sensitivity_map = {
            'temperature': 0.3,  # Environmental data - low sensitivity
            'humidity': 0.3,     # Environmental data - low sensitivity
            'location': 1.0,     # Location data - high sensitivity
            'health': 0.8        # Health data - high sensitivity
        }
        
        # Historical data storage (sliding window)
        self.history_data = defaultdict(lambda: {'window': [], 'count': 0})
        
        # Default weights
        self.w1 = w1
        self.w2 = w2
    
    def calculate_data_sensitivity(self, field_name):
        return self.data_sensitivity_map.get(field_name, 0.5)  # Default to medium sensitivity

    def calculate_entropy(self, data, field_name=None, update_history=False):
        """Calculate information entropy (supports sliding window)
        Entropy formula: H(X) = -Σ p(x_i) * log2(p(x_i))
        
        Args:
            data: Input data (single value or array)
            field_name: Field name (for window calculation)
            update_history: Whether to update historical data
        Returns:
            float: Normalized information entropy (0-1)
        """
        if field_name and update_history:
            # Update sliding window
            if isinstance(data, (list, np.ndarray)):
                new_data = list(data)
            else:
                new_data = [data]
                
            window_data = self.history_data[field_name]
            window_data['window'].extend(new_data)
            window_data['count'] += len(new_data)
            
            # Maintain window size
            if len(window_data['window']) > self.window_size:
                remove_count = len(window_data['window']) - self.window_size
                window_data['window'] = window_data['window'][remove_count:]
                window_data['count'] = self.window_size
                
        # Get data for calculation
        calc_data = []
        if field_name:
            window_data = self.history_data[field_name]
            if window_data['count'] > 0:
                calc_data = window_data['window']
        else:
            calc_data = data if isinstance(data, (list, np.ndarray)) else [data]
            
        # Handle empty data case
        if not calc_data:
            return 0.0
            
        # Calculate entropy value
        value_counts = defaultdict(int)
        total = len(calc_data)
        for value in calc_data:
            value_counts[value] += 1
            
        entropy = 0.0
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * log2(p)
                
        # Return raw entropy value
        return max(0.0, entropy)  # Ensure not less than 0
    
    def calculate_context_risk(self, fields, data):
        """Calculate contextual risk
        Args:
            fields (list): List of field names
            data (DataFrame/Series): DataFrame or single row containing field data
        Returns:
            float: Contextual risk score
        """
        total_risk = 0.0
        n = len(fields)
        
        for i in range(n):
            field_i = fields[i]
            # Simulate indicator function I: 80% probability 1, 20% probability 0
            import random
            I = 1 if random.random() < 0.8 else 0
            
       
            field_data = data[field_i]
            H = self.calculate_entropy(field_data)  # Use calculate_entropy method uniformly
                
            total_risk += I * H
            
        return total_risk / n if n > 0 else 0.0
    
    def calculate_semantic_risk(self, fields, data):
        """Calculate semantic risk score R_sem = w1*DataSens + w2*ContextRisk
        Args:
            fields (list): List of field names
            data (DataFrame): DataFrame containing all field data
        Returns:
            float: Semantic risk score
        """
        # Calculate average data sensitivity
        data_sens = np.mean([self.calculate_data_sensitivity(f) for f in fields])
        
        # Calculate contextual risk
        context_risk = self.calculate_context_risk(fields, data)
        
        # Calculate comprehensive semantic risk
        return self.w1 * data_sens + self.w2 * context_risk

    def set_weights(self, w1, w2):
        """Set weight parameters
        Args:
            w1 (float): Data sensitivity weight
            w2 (float): Contextual risk weight
        """
        self.w1 = w1
        self.w2 = w2
        
    def calculate(self, df, use_columns=None):
        
        import pandas as pd
        
        
        
        # Determine columns to use
        if use_columns is None:
            use_columns = df.columns[:2]  # Default to first two columns
            
        # Calculate semantic risk for each row
        results = {}
        for idx, row in df.iterrows():
            data_sens = np.mean([self.calculate_data_sensitivity(col) for col in use_columns])
            # Update historical data and calculate contextual risk
            for col in use_columns:
                self.calculate_entropy(row[col], col, update_history=True)
            
            # Calculate contextual risk using historical data
            context_risk = 0.0
            for col in use_columns:
                # Simulate indicator function I: 80% probability 1, 20% probability 0
                import random
                I = 1 if random.random() < 0.8 else 0
                H = self.calculate_entropy(None, col, False)  # Calculate entropy using historical data
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
        """Add custom data sensitivity mapping
        Args:
            field_name (str): Field name
            sensitivity (float): Sensitivity value (0-1)
        """
        self.data_sensitivity_map[field_name] = sensitivity
