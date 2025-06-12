import numpy as np
import tensorflow as tf
import logging
import os
from datetime import datetime
from .data_processor import DataProcessor


class IntrusionDetector:
    def __init__(self):
        """Initialize the Intrusion Detector"""
        # Initialize logger first
        self.logger = self._setup_logging()

        # Initialize other components
        self.initialize_model()
        self.data_processor = DataProcessor()
        self.prediction_count = 0

    def _setup_logging(self):
        """Setup logging configuration and return logger"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join('monitoring', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        logger = logging.getLogger('detector')
        logger.setLevel(logging.INFO)

        # Create file handler
        log_file = os.path.join(
            log_dir,
            f'detector_{datetime.utcnow().strftime("%Y%m%d")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Intrusion Detector initialized")
        return logger

    def initialize_model(self):
        """Initialize a lightweight neural network model"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(41,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            if hasattr(self, 'logger'):
                self.logger.info("Model initialized successfully")

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, input_data):
        """Make predictions on input data"""
        try:
            # Preprocess the data
            processed_data = self.data_processor.preprocess_data(input_data)

            # Make predictions in batches
            predictions = self.model.predict(
                processed_data,
                batch_size=32,
                verbose=0
            )

            self.prediction_count += len(predictions)

            if hasattr(self, 'logger'):
                self.logger.debug(
                    f"Made predictions on {len(predictions)} samples. "
                    f"Total predictions: {self.prediction_count}"
                )

            return predictions.flatten()

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_statistics(self):
        """Get model statistics"""
        return {
            'total_predictions': self.prediction_count,
            'model_type': 'Sequential Neural Network',
            'input_shape': (41,),
            'last_prediction_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }