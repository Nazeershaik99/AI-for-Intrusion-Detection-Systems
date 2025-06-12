# AI-based Network Intrusion Detection System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time network intrusion detection system using artificial intelligence to monitor and analyze network traffic for potential security threats.

## 🔍 Overview

This project implements an intelligent network monitoring system that:
- Processes network traffic in real-time
- Detects potential security threats using AI
- Provides visual analytics of network security status
- Generates detailed security reports and statistics

## 🚀 Features

- **Real-time Monitoring**: Processes network packets in real-time with minimal latency
- **AI-powered Detection**: Uses neural networks to identify potential threats
- **Visual Analytics**: 
  - Live visualization of security metrics
  - Alert distribution analysis
  - Packet processing statistics
- **Comprehensive Reporting**:
  - Detailed alert logs
  - Statistical analysis
  - Performance metrics

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

```bash
# Core dependencies
numpy>=1.19.2
pandas>=1.2.0
matplotlib>=3.3.0
tensorflow>=2.5.0
scikit-learn>=0.24.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Nazeershaik99/AI-for-Intrusion-Detection-Systems.git
cd AI-for-Intrusion-Detection-Systems
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Prepare the data directory:
```bash
mkdir -p data/raw
```

## 🎮 Usage

1. Place your NSL-KDD test data in `data/raw/NSL_KDD_test.txt`

2. Run the monitoring system:
```bash
python run_project.py
```

3. View results in the monitoring directory:
- `monitoring/logs/`: Contains detailed system logs
- `monitoring/alerts/`: Stores detected security alerts
- `monitoring/statistics/`: Keeps performance and analysis statistics

## 📊 Output Example

The system provides real-time visualization of:
- Alert Distribution by Severity
- Packet Processing Rate
- Security Alert Feed

Final results include:
- Total packets processed
- Alert distribution statistics
- Processing performance metrics
- Detailed security analysis

## 🌳 Project Structure

```
AI-for-Intrusion-Detection-Systems/
├── README.md               # Project documentation
├── requirements.txt        # Package dependencies
├── run_project.py         # Main execution script
├── src/                   # Source code directory
│   ├── __init__.py
│   ├── monitor.py         # Network monitoring module
│   ├── prediction.py      # AI prediction module
│   ├── data_processor.py  # Data processing utilities
│   └── visualization.py   # Visualization components
├── data/                  # Data directory
│   └── raw/              # Raw input data
└── monitoring/           # Output directory
    ├── alerts/          # Security alerts
    ├── logs/            # System logs
    └── statistics/      # Analysis statistics
```

## 📝 Components

- **monitor.py**: Implements real-time network traffic monitoring
- **prediction.py**: Contains AI models for threat detection
- **data_processor.py**: Handles data preprocessing and transformation
- **visualization.py**: Manages real-time data visualization

## 🔑 Key Features in Detail

1. **Network Monitoring**
   - Real-time packet processing
   - Traffic pattern analysis
   - Protocol-level inspection

2. **AI Detection System**
   - Neural network-based threat detection
   - Pattern recognition
   - Anomaly detection

3. **Visualization System**
   - Real-time metrics display
   - Interactive data visualization
   - Alert distribution analysis

## 👥 Author

- **Nazeer Shaik**
  - GitHub: [@Nazeershaik99](https://github.com/Nazeershaik99)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📫 Contact

For questions or feedback, please:
- Create an issue in this repository
- Contact me through GitHub [@Nazeershaik99](https://github.com/Nazeershaik99)
