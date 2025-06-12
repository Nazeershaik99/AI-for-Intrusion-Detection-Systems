import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """
    Handles loading and preprocessing of cybersecurity datasets
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_nsl_kdd(self, filepath: str) -> pd.DataFrame:
        """
        Load NSL-KDD dataset
        
        Args:
            filepath: Path to the NSL-KDD dataset
            
        Returns:
            Loaded DataFrame
        """
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'class'
        ]
        
        return pd.read_csv(filepath, names=columns)
        
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the dataset for training
        
        Args:
            data: Input DataFrame
            test_size: Proportion of dataset to include in the test split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Handle categorical variables
        categorical_columns = ['protocol_type', 'service', 'flag']
        data_encoded = pd.get_dummies(data, columns=categorical_columns)
        
        # Separate features and target
        X = data_encoded.drop('class', axis=1)
        y = data_encoded['class']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test