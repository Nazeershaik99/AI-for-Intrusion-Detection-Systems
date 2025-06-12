import time
import numpy as np
from datetime import datetime
import logging
import json
import os
from .prediction import IntrusionDetector


class NetworkMonitor:
    def __init__(self, user_login="Nazeershaik99"):
        """Initialize NetworkMonitor with configurations and setup"""
        # Store user information
        self.user_login = user_login
        self.start_time = datetime.utcnow()

        # Initialize detector and thresholds
        self.detector = IntrusionDetector()
        self.alert_threshold = 0.9
        self.high_severity_threshold = 0.95

        # Initialize counters and statistics
        self.total_packets_processed = 0
        self.total_alerts_generated = 0
        self.alert_history = []

        # Setup monitoring system
        self.setup_monitoring()

    def setup_monitoring(self):
        """Setup monitoring directories and logging configuration"""
        # Create necessary directories
        self.monitor_dir = 'monitoring'
        self.alerts_dir = os.path.join(self.monitor_dir, 'alerts')
        self.stats_dir = os.path.join(self.monitor_dir, 'statistics')
        self.logs_dir = os.path.join(self.monitor_dir, 'logs')

        for directory in [self.alerts_dir, self.stats_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Setup logging configuration
        log_file = os.path.join(
            self.logs_dir,
            f'monitor_{self.user_login}_{datetime.utcnow().strftime("%Y%m%d")}.log'
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info(f"Network Monitoring System Initialized for user: {self.user_login}")
        logging.info(f"Start Time (UTC): {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def calculate_severity(self, confidence):
        """Calculate alert severity based on confidence score"""
        if confidence > self.high_severity_threshold:
            return 'HIGH'
        elif confidence > self.alert_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'

    def generate_alert(self, prediction, confidence, timestamp, traffic_data=None):
        """Generate detailed security alert"""
        severity = self.calculate_severity(confidence)

        alert = {
            'alert_id': f"ALERT-{self.user_login}-{timestamp.replace(':', '')}-{self.total_alerts_generated}",
            'timestamp': timestamp,
            'user': self.user_login,
            'detection': {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'severity': severity
            },
            'status': {
                'action_required': severity == 'HIGH',
                'is_active': True,
                'created_at': timestamp
            }
        }

        if traffic_data is not None:
            alert['traffic_data'] = traffic_data

        # Save alert to file
        alert_file = os.path.join(
            self.alerts_dir,
            f"alert_{self.user_login}_{timestamp.replace(':', '-')}_{severity.lower()}.json"
        )

        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=4)

        self.total_alerts_generated += 1
        self.alert_history.append(alert)

        return alert

    def monitor_network(self, input_data):
        """Monitor network traffic and generate alerts"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        self.total_packets_processed += len(input_data)

        try:
            predictions = self.detector.predict(input_data)
            alerts = []

            for i, conf in enumerate(predictions):
                if conf > self.alert_threshold:
                    traffic_data = {
                        'packet_id': i,
                        'timestamp': timestamp,
                        'confidence': float(conf)
                    }
                    alert = self.generate_alert(1, conf, timestamp, traffic_data)
                    alerts.append(alert)

            return alerts

        except Exception as e:
            logging.error(f"Error in monitoring cycle: {str(e)}")
            raise

    def get_monitoring_summary(self):
        """Get summary of monitoring activities"""
        return {
            'status': 'active',
            'user': self.user_login,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': {
                'total_packets': self.total_packets_processed,
                'total_alerts': self.total_alerts_generated,
                'alert_rate': self.total_alerts_generated / max(1, self.total_packets_processed)
            },
            'alert_severity_distribution': {
                'HIGH': len([a for a in self.alert_history if a['detection']['severity'] == 'HIGH']),
                'MEDIUM': len([a for a in self.alert_history if a['detection']['severity'] == 'MEDIUM']),
                'LOW': len([a for a in self.alert_history if a['detection']['severity'] == 'LOW'])
            }
        }

    def cleanup(self):
        """Cleanup and save final statistics"""
        try:
            stats = {
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'user': self.user_login,
                'total_runtime': (datetime.utcnow() - self.start_time).total_seconds(),
                'total_packets': self.total_packets_processed,
                'total_alerts': self.total_alerts_generated,
                'alert_distribution': {
                    'HIGH': len([a for a in self.alert_history if a['detection']['severity'] == 'HIGH']),
                    'MEDIUM': len([a for a in self.alert_history if a['detection']['severity'] == 'MEDIUM']),
                    'LOW': len([a for a in self.alert_history if a['detection']['severity'] == 'LOW'])
                }
            }

            stats_file = os.path.join(
                self.stats_dir,
                f"final_stats_{self.user_login}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)

            logging.info("Monitoring session completed")
            return stats

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            raise