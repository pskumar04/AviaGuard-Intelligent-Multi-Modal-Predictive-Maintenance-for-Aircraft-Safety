# ✈️ AviaGuard: Intelligent Multi-Modal Predictive Maintenance for Aircraft Safety

AviaGuard is an AI-powered predictive maintenance system for aircraft safety that integrates data from four sensor modalities—**vibration, thermal, acoustic, and pressure sensors**—using a hybrid AI approach. It extends beyond conventional vibration-only models to provide comprehensive aircraft health monitoring with **96.2% multi-class accuracy**.

## 🚀 Key Features

- **Multi-Sensor Fusion**: Vibration, thermal, acoustic, pressure sensors
- **Hybrid CNN+Transformer Architecture**: Spatial feature extraction + temporal pattern learning
- **Multi-Class Fault Diagnosis**: 16 fault types with severity levels (low, medium, high, critical)
- **Real-Time Edge Deployment**: 8.3ms inference on NVIDIA Jetson Xavier (TensorRT optimized)
- **Explainable AI (XAI)**: SHAP + LIME for transparent predictions
- **Web Dashboard**: Flask-based interface with email notifications
- **Model Optimization**: 65% size reduction (120MB → 42MB)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Multi-Class Accuracy | 96.2% |
| Precision | 98.2% |
| Recall | 97.1% |
| F1-Score | 97.6% |
| Inference Latency | 8.3ms |
| False Negative Rate | 1.8% |

## 🛠️ Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Deep Learning**: TensorFlow, Keras
- **XAI**: SHAP, LIME
- **Edge Optimization**: TensorRT, ONNX
- **Database**: PostgreSQL / SQLite
- **Frontend**: HTML, CSS, Bootstrap, JavaScript

## 📁 Project Structure
├── aircraft_configs/ # JSON configs for aircraft models
├── data_processing/ # Data loading and preprocessing
├── models/ # CNN+Transformer implementation
├── website/ # Flask web application
│ ├── templates/ # HTML templates
│ ├── static/ # CSS, JS, images
│ └── utils/ # Email, prediction, survey utilities
├── xai/ # SHAP and LIME explainers
├── edge/ # Edge deployment optimization
├── deployment/ # TensorRT conversion
├── run.py # Main application
└── requirements.txt # Python dependencies

## 🚀 Installation & Running

```bash
# Clone the repository
git clone https://github.com/yourusername/AviaGuard.git
cd AviaGuard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

# Open browser and go to
http://localhost:5000


📊 Datasets Used
NASA Turbofan Degradation Simulation Dataset (NASA, 2008)

Rotor Blade Vibration Dataset (Lee et al., 2025)

Custom Multi-Sensor Dataset (50,000+ samples, 3 aircraft models)

🔬 Model Architecture
The hybrid model consists of:

CNN Encoders: Separate 1D CNNs for vibration, thermal, acoustic, pressure data

Transformer Encoder: Multi-head self-attention for temporal dependencies

Multi-Task Learning: Fault classification (16 classes), severity prediction, aircraft type

📧 Email Notifications
Management: Summary alerts with condition status

Maintenance Team: Detailed alerts with sensor readings and recommendations

🔮 Future Enhancements
Real sensor integration with aviation partners

Federated learning for privacy-preserving training

Digital twin integration for real-time health monitoring

Autonomous maintenance scheduling

DO-178C/ARINC 625 certification

👥 Team
Name	Roll No.	Role
T. Jagadeesh	 223J1A05H0	Data Collection & Preprocessing
R.G.R. Lalitha	 233J5A0516	Hybrid Model Development
P. Satish Kumar	 223J1A05D4	Web Application & Email Integration
S. Meghana	     223J1A05G5	XAI & Visualization
K. Vimala Nanda	 233J5A0520	Edge Deployment & Email Notifications
Guide: Mr. Satyabrata Patro, Associate Professor, CSE

📄 License
This project is for academic purposes. All rights reserved.

🙏 Acknowledgments
Raghu Institute of Technology, Department of CSE

Mr. Satyabrata Patro (Guide)

Dr. S. Srinadhraju (HOD, CSE)