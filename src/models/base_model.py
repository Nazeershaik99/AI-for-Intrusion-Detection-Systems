from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class BaseModel(ABC):
    """
    Abstract base class for all models
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    @abstractmethod
    def build(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        pass