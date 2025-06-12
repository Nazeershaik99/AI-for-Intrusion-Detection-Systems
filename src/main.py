import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import json
import logging


class AICybersecuritySystem:
    def __init__(self):
        self.columns = [
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
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        self.numeric_cols = [col for col in self.columns if col not in self.categorical_cols + ['class']]

        self.numeric_transformer = StandardScaler()
        self.categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_cols),
                ('cat', self.categorical_transformer, self.categorical_cols)
            ])

        self.model = None
        self.setup_directories()

    def setup_directories(self):
        os.makedirs('results', exist_ok=True)
        os.makedirs('models/baseline', exist_ok=True)
        os.makedirs('models/adversarial', exist_ok=True)

    def load_and_preprocess_data(self):
        logging.info("Loading and preprocessing data...")
        try:
            train_data = pd.read_csv('data/raw/NSL_KDD_train.txt', names=self.columns)
            test_data = pd.read_csv('data/raw/NSL_KDD_test.txt', names=self.columns)

            for col in self.numeric_cols:
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
                test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

            train_data = train_data.fillna(0)
            test_data = test_data.fillna(0)

            X_train = self.preprocessor.fit_transform(train_data[self.numeric_cols + self.categorical_cols])
            X_test = self.preprocessor.transform(test_data[self.numeric_cols + self.categorical_cols])

            X_train = X_train.toarray() if not isinstance(X_train, np.ndarray) else X_train
            X_test = X_test.toarray() if not isinstance(X_test, np.ndarray) else X_test

            y_train = (train_data['class'] != 'normal').astype(int)
            y_test = (test_data['class'] != 'normal').astype(int)

            logging.info(f"Training data shape: {X_train.shape}")
            logging.info(f"Test data shape: {X_test.shape}")
            logging.info(f"Number of features: {X_train.shape[1]}")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_model(self, X_train, y_train, X_test, y_test):
        logging.info("Starting model training...")
        input_shape = X_train.shape[1]
        inputs = tf.keras.Input(shape=(input_shape,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath='models/baseline/best_model.keras', monitor='val_accuracy', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        self.model.save('models/baseline/final_model.keras')
        return history

    def evaluate_model(self, X_test, y_test):
        logging.info("Evaluating model performance...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        evaluation = {'loss': float(results[0]), 'accuracy': float(results[1])}

        with open('results/evaluation.json', 'w') as f:
            json.dump(evaluation, f, indent=4)

        return evaluation


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    logging.basicConfig(level=logging.INFO)

    try:
        system = AICybersecuritySystem()
        X_train, X_test, y_train, y_test = system.load_and_preprocess_data()

        logging.info(f"\nDataset Information:\n{'-'*50}")
        logging.info(f"Training samples: {X_train.shape[0]}")
        logging.info(f"Test samples: {X_test.shape[0]}")
        logging.info(f"Number of features: {X_train.shape[1]}")
        logging.info(f"Positive class ratio in training: {np.mean(y_train):.2%}")
        logging.info(f"Positive class ratio in testing: {np.mean(y_test):.2%}")

        history = system.train_model(X_train, y_train, X_test, y_test)
        results = system.evaluate_model(X_test, y_test)

        logging.info("\nFinal Results:\n" + "-" * 50)
        logging.info(f"Test Loss: {results['loss']:.4f}")
        logging.info(f"Test Accuracy: {results['accuracy']:.4f}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()