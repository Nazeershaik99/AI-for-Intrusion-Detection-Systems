import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    def __init__(self):
        """Initialize the data processor with necessary encoders and scalers"""
        # Initialize encoders and scaler
        self.label_encoders = {}
        self.scaler = StandardScaler()

        # Define column names by type
        self.categorical_columns = ['protocol_type', 'service', 'flag']
        self.numeric_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]

        # Initialize label encoders
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()

        # Pre-fit label encoders with known categories
        self.label_encoders['protocol_type'].fit(['tcp', 'udp', 'icmp'])
        self.label_encoders['flag'].fit(['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'SH', 'S1', 'S2', 'RSTOS0', 'S3', 'OTH'])

        # Initialize service categories (common services in NSL-KDD dataset)
        common_services = [
            'http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i',
            'ntp_u', 'ecr_i', 'other', 'private', 'pop_3', 'ftp_data', 'rje', 'time',
            'mtp', 'link', 'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain',
            'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443',
            'exec', 'printer', 'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo',
            'discard', 'systat', 'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2',
            'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn', 'netbios_dgm', 'sql_net',
            'vmnet', 'bgp', 'Z39_50', 'ldap', 'netstat', 'urh_i', 'X11', 'urp_i',
            'pm_dump', 'tftp_u', 'tim_i', 'red_i'
        ]
        self.label_encoders['service'].fit(common_services)

    def preprocess_data(self, data):
        """Preprocess the input data"""
        try:
            # Convert to DataFrame if necessary
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=self.categorical_columns + self.numeric_columns)

            # Create a copy to avoid modifying the original data
            processed_data = data.copy()

            # Handle categorical columns
            for col in self.categorical_columns:
                if col in processed_data.columns:
                    try:
                        # Transform known categories
                        processed_data[col] = self.label_encoders[col].transform(processed_data[col])
                    except ValueError as e:
                        # Handle unknown categories
                        unknown_categories = set(processed_data[col]) - set(self.label_encoders[col].classes_)
                        print(f"Warning: Unknown categories in {col}: {unknown_categories}")
                        # Map unknown categories to a new index
                        for cat in unknown_categories:
                            processed_data.loc[processed_data[col] == cat, col] = -1

            # Convert all numeric columns to float
            for col in self.numeric_columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='raise')

            # Combine all features
            feature_matrix = processed_data[self.categorical_columns + self.numeric_columns].values

            # Scale the features
            if not hasattr(self, 'is_fitted'):
                self.scaler.fit(feature_matrix)
                self.is_fitted = True

            scaled_data = self.scaler.transform(feature_matrix)

            return scaled_data

        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def get_feature_names(self):
        """Get the list of feature names after preprocessing"""
        return self.categorical_columns + self.numeric_columns