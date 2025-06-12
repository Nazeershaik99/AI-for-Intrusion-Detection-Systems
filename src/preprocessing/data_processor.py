import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class DataProcessor:
    """
    Class for handling data preprocessing tasks
    """
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from specified file"""
        return pd.read_csv(filepath)
    
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       target_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data
        
        Args:
            data: Input DataFrame
            target_column: Name of the target variable column
            
        Returns:
            Tuple of features and targets (if target_column provided)
        """
        # Handle missing values
        data = data.fillna(0)
        
        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y.values
        
        return self.scaler.fit_transform(data), None