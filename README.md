# ML/DL Project 2 - Perceptron Implementation

## 📋 Project Overview
This project implements a perceptron model with a Streamlit web interface for interactive demonstrations.

## 🏗️ Project Structure
ML_DL_Project2/
├── deployment/ # Docker deployment files
│ ├── Dockerfile
│ ├── deploy.sh
│ └── docker-compose.yml
├── notebooks/ # Jupyter notebooks for experimentation
│ └── Perceptron_Project2.ipynb
├── src/ # Source code
│ ├── App_streamlit.py # Main Streamlit application
│ ├── models/ # ML model implementations
│ ├── utils/ # Utility functions
│ └── data/ # Data processing modules
├── tests/ # Test cases
└── requirements.txt # Python dependencies

text

## 🚀 Installation

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/App_streamlit.py
Docker Deployment
bash
# Build and run with Docker Compose
docker-compose -f deployment/docker-compose.yml up --build

# Or use the deployment script
./deployment/deploy.sh
📊 Features
Perceptron machine learning model

Interactive Streamlit web interface

Docker containerization

Modular code structure

🛠️ Technologies Used
Python

Streamlit

Docker

Jupyter Notebooks

NumPy/Pandas

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📝 License
This project is for educational purposes.
