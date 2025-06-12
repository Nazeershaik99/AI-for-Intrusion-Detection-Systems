import tensorflow as tf
from typing import Dict, Any, Tuple
import numpy as np

class IDSModel:
    """
    Base Intrusion Detection System model using deep learning
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the neural network model
        
        Args:
            input_shape: Shape of input features
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             validation_data: Tuple[np.ndarray, np.ndarray],
             epochs: int = 10,
             batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return {
            'loss': loss,
            'accuracy': accuracy
        }