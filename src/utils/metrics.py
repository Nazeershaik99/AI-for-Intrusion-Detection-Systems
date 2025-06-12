import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing various metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1_score': 2 * tp / (2 * tp + fp + fn),
        'false_positive_rate': fp / (fp + tn),
        'detection_rate': tp / (tp + fn)
    }
    
    return metrics