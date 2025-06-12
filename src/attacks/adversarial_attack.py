from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def generate_attack(self, 
                       model: Any, 
                       X: np.ndarray, 
                       y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples"""
        pass
    
    @abstractmethod
    def evaluate_attack(self, 
                       model: Any, 
                       X_original: np.ndarray, 
                       X_adversarial: np.ndarray, 
                       y: np.ndarray) -> Dict[str, float]:
        """Evaluate attack effectiveness"""
        pass