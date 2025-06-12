import time
from datetime import datetime
import pandas as pd
import numpy as np
import os
import logging
from src.monitor import NetworkMonitor
from src.visualization import MonitoringVisualizer


def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join('monitoring', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'monitoring_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_test_data(file_path):
    """Load and prepare test data"""
    try:
        # Define column names for NSL-KDD dataset
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

        # Read only first 1000 rows for faster processing
        data = pd.read_csv(file_path, names=columns, nrows=1000)

        # Convert numeric columns
        numeric_cols = [col for col in columns if col not in ['protocol_type', 'service', 'flag', 'class']]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Fill missing values with 0
        data = data.fillna(0)

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def simulate_real_time_traffic(data, batch_size=25):
    """Simulate real-time network traffic by splitting data into batches"""
    total_batches = len(data) // batch_size
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        current_batch = i // batch_size + 1
        yield batch, current_batch, total_batches


def create_monitoring_dirs():
    """Create necessary monitoring directories"""
    directories = [
        os.path.join('monitoring', dir_name)
        for dir_name in ['alerts', 'logs', 'statistics']
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_statistics(stats_file, runtime, processed_packets, total_alerts,
                    alerts_by_severity, start_time):
    """Save monitoring statistics to file"""
    try:
        with open(stats_file, 'w') as f:
            f.write("Network Security Monitoring Statistics\n")
            f.write("=" * 40 + "\n\n")

            # Session Information
            f.write("Session Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Duration: {runtime:.2f} seconds\n\n")

            # Processing Statistics
            f.write("Processing Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Packets Processed: {processed_packets:,d}\n")
            f.write(f"Processing Rate: {processed_packets / runtime:.1f} packets/second\n\n")

            # Alert Statistics
            f.write("Alert Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Alerts: {total_alerts:,d}\n")
            f.write(f"Alert Rate: {total_alerts / runtime:.1f} alerts/second\n\n")

            # Severity Distribution
            f.write("Alert Severity Distribution:\n")
            f.write("-" * 20 + "\n")
            for severity, count in alerts_by_severity.items():
                percentage = (count / total_alerts * 100) if total_alerts > 0 else 0
                f.write(f"{severity}: {count:,d} ({percentage:.1f}%)\n")

        return True
    except Exception as e:
        logging.error(f"Error saving statistics: {str(e)}")
        return False


def main():
    """Main execution function"""
    # Setup logging
    logger = setup_logging()

    # Record start time and user
    start_time = datetime.utcnow()
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: Nazeershaik99")

    # Create monitoring directories
    create_monitoring_dirs()

    # Initialize monitoring system
    try:
        monitor = NetworkMonitor(user_login="Nazeershaik99")
        visualizer = MonitoringVisualizer()
    except Exception as e:
        logger.error(f"Error initializing monitoring system: {str(e)}")
        return

    # Load test data
    try:
        test_data = load_test_data('data/raw/NSL_KDD_test.txt')
        print(f"\nLoaded {len(test_data):,d} records from test dataset")
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return

    print("\n🔍 Starting Network Security Analysis 🔍")
    print("=" * 50)

    # Initialize monitoring variables
    processed_packets = 0
    total_alerts = 0
    start_process_time = time.time()
    update_interval = 0.1  # 100ms between updates
    last_update_time = start_process_time

    try:
        # Main processing loop
        for batch_data, batch_num, total_batches in simulate_real_time_traffic(test_data):
            try:
                # Process current batch
                alerts = monitor.monitor_network(batch_data)
                processed_packets += len(batch_data)
                total_alerts += len(alerts)

                # Update progress
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    elapsed_time = current_time - start_process_time
                    packets_per_second = processed_packets / elapsed_time

                    print(f"\rProgress: {batch_num}/{total_batches} batches | "
                          f"Processed: {processed_packets:,d} packets | "
                          f"Rate: {packets_per_second:.1f} packets/s", end='')

                    # Update visualization data
                    summary = monitor.get_monitoring_summary()
                    visualizer.update_data(summary, alerts)
                    last_update_time = current_time

                # Stop after processing all data or 30 seconds
                if batch_num >= total_batches or (current_time - start_process_time) > 30:
                    break

            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\n\nAnalysis stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Calculate final statistics
        total_runtime = time.time() - start_process_time
        final_summary = monitor.get_monitoring_summary()

        # Show final results
        print("\n\nGenerating final analysis...")
        visualizer.show_final_results(total_runtime)

        # Save statistics
        stats_file = os.path.join(
            'monitoring',
            'statistics',
            f'stats_{start_time.strftime("%Y%m%d_%H%M%S")}.txt'
        )

        if save_statistics(
                stats_file,
                total_runtime,
                processed_packets,
                total_alerts,
                final_summary['alert_severity_distribution'],
                start_time
        ):
            print(f"\nStatistics saved to: {stats_file}")

        # Cleanup
        monitor.cleanup()
        visualizer.close()

        print("\nAnalysis completed successfully")


if __name__ == "__main__":
    main()