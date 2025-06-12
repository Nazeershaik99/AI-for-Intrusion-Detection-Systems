# Create main project directory
mkdir AI-Cybersecurity-Framework
cd AI-Cybersecurity-Framework

# Create directory structure
mkdir -p data/{raw,processed}
mkdir -p models/{baseline,adversarial}
mkdir -p src/{preprocessing,models,attacks,defense,utils,visualization}
mkdir -p tests
mkdir -p notebooks

# Create initial README files
touch README.md
touch requirements.txt
touch setup.py
touch .gitignore