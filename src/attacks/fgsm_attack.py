import numpy as np
from typing import Dict, Any
import tensorflow as tf

class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) attack implementation
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get('epsilon', 0.01)
        
    def generate_adversarial_examples(self, 
                                    model: tf.keras.Model,
                                    X: np.ndarray,
                                    y: np.ndarray) -> np.ndarray:
        """
        Generate adversarial examples using FGSM
        
        Args:
            model: Target model
            X: Input samples
            y: True labels
            
        Returns:
            Adversarial examples
        """
        X_tensor = tf.convert_to_tensor(X)
        y_tensor = tf.convert_to_tensor(y)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor)
            loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
            
        gradients = tape.gradient(loss, X_tensor)
        signed_gradients = tf.sign(gradients)
        adversarial_examples = X_tensor + self.epsilon * signed_gradients
        
        return adversarial_examples.numpy()