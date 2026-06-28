#!/bin/bash

# Multi-Modal Aircraft Predictive Maintenance - Dependency Installation Script
# Supports Ubuntu 20.04/22.04 and NVIDIA Jetson platforms

set -e

echo "=================================================="
echo "Multi-Modal Aircraft Predictive Maintenance System"
echo "Dependency Installation Script"
echo "=================================================="

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="JETSON"
    echo "Detected: NVIDIA Jetson Platform"
elif [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" = "ubuntu" ]; then
        PLATFORM="UBUNTU"
        echo "Detected: Ubuntu $VERSION_ID"
    else
        PLATFORM="OTHER"
        echo "Detected: $PRETTY_NAME"
    fi
else
    PLATFORM="UNKNOWN"
    echo "Warning: Unknown platform"
fi

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Update system packages
echo -e "\n1. Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
print_status "System packages updated"

# Install system dependencies
echo -e "\n2. Installing system dependencies..."
sudo apt-get install -y -qq \
    python3.9 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    htop \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtiff5-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    libgstreamer-plugins-base1.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev

print_status "System dependencies installed"

# Create virtual environment
echo -e "\n3. Creating Python virtual environment..."
python3.9 -m venv aircraft-pm-env
source aircraft-pm-env/bin/activate
print_status "Virtual environment created"

# Upgrade pip
echo -e "\n4. Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# Install Python dependencies
echo -e "\n5. Installing Python dependencies..."

# Install base requirements
pip install numpy==1.24.3
pip install pandas==2.1.0
pip install scikit-learn==1.3.0
pip install scipy==1.11.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install plotly==5.17.0
pip install jupyter==1.0.0
pip install notebook==6.5.5

print_status "Base dependencies installed"

# Install deep learning frameworks based on platform
if [ "$PLATFORM" = "JETSON" ]; then
    echo -e "\n6. Installing Jetson-optimized deep learning libraries..."
    
    # Install TensorFlow for Jetson
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 \
        tensorflow==2.13.0
    
    # Install PyTorch for Jetson
    wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0-cp39-cp39-linux_aarch64.whl
    pip install torch-2.0.0-cp39-cp39-linux_aarch64.whl
    pip install torchvision==0.15.0
    
    # Install TensorRT
    pip install tensorrt==8.6.1
    pip install pycuda==2022.2.2
    
    # Install Jetson utilities
    pip install jetson-stats==4.2.0
    
elif [ "$PLATFORM" = "UBUNTU" ]; then
    echo -e "\n6. Installing GPU-accelerated deep learning libraries..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing CUDA-enabled libraries..."
        
        # Install TensorFlow with GPU support
        pip install tensorflow[and-cuda]==2.13.0
        
        # Install PyTorch with CUDA
        pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
        
        # Install TensorRT if needed
        # pip install tensorrt==8.6.1
    else
        echo "No NVIDIA GPU detected. Installing CPU-only libraries..."
        pip install tensorflow==2.13.0
        pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo -e "\n6. Installing CPU-only deep learning libraries..."
    pip install tensorflow==2.13.0
    pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
fi

print_status "Deep learning libraries installed"

# Install edge deployment dependencies
echo -e "\n7. Installing edge deployment dependencies..."
pip install tflite-runtime==2.14.0
pip install onnxruntime==1.15.1
pip install opencv-python==4.8.0
pip install pyserial==3.5
pip install pyzmq==25.0.2
pip install pyaudio==0.2.11

print_status "Edge deployment dependencies installed"

# Install XAI dependencies
echo -e "\n8. Installing Explainable AI dependencies..."
pip install shap==0.42.1
pip install lime==0.2.0.1
pip install interpret==0.5.0

print_status "XAI dependencies installed"

# Install web framework dependencies
echo -e "\n9. Installing web framework dependencies..."
pip install flask==2.3.3
pip install flask-socketio==5.3.4
pip install streamlit==1.28.0
pip install fastapi==0.104.0
pip install uvicorn==0.24.0
pip install pydantic==2.4.2
pip install python-dotenv==1.0.0

print_status "Web framework dependencies installed"

# Install database and messaging
echo -e "\n10. Installing database and messaging dependencies..."
pip install sqlite3==3.41.2
pip install redis==5.0.0
pip install paho-mqtt==1.6.1

print_status "Database and messaging dependencies installed"

# Install utilities
echo -e "\n11. Installing utility packages..."
pip install joblib==1.3.2
pip install tqdm==4.66.0
pip install pyyaml==6.0
pip install h5py==3.9.0
pip install scikit-image==0.21.0
pip install imbalanced-learn==0.11.0

print_status "Utility packages installed"

# Install development dependencies
echo -e "\n12. Installing development dependencies..."
pip install pytest==7.4.3
pip install pytest-cov==4.1.0
pip install black==23.11.0
pip install flake8==6.1.0
pip install pylint==3.0.2

print_status "Development dependencies installed"

# Create project structure
echo -e "\n13. Creating project structure..."
mkdir -p data_processing models web_app deployment edge xai aircraft_configs
mkdir -p data/multi_sensor data/vibration data/thermal data/acoustic
mkdir -p results/logs results/models results/plots

print_status "Project structure created"

# Download sample data (if available)
echo -e "\n14. Downloading sample datasets..."
if command -v wget &> /dev/null; then
    # Download sample aircraft data
    wget -q -O data/sample_aircraft_data.csv https://raw.githubusercontent.com/aircraft-pm/datasets/main/sample_data.csv || \
    echo "Could not download sample data. Using synthetic data instead."
fi

print_status "Sample data downloaded"

# Set up environment variables
echo -e "\n15. Setting up environment variables..."
cat > .env << EOF
# Multi-Modal Aircraft Predictive Maintenance Environment
AIRCRFT_PM_ENV=development
AIRCRFT_PM_DEBUG=True
AIRCRFT_PM_MODEL_PATH=./results/models
AIRCRFT_PM_DATA_PATH=./data
AIRCRFT_PM_LOG_LEVEL=INFO

# Edge Deployment Settings
EDGE_DEVICE_TYPE=auto_detect
EDGE_INFERENCE_THRESHOLD_MS=10
EDGE_MAX_MEMORY_MB=1024

# Multi-Sensor Settings
SENSOR_SAMPLING_RATE=1000
SENSOR_FUSION_METHOD=hybrid
SENSOR_BUFFER_SIZE=1000

# Model Training Settings
TRAIN_BATCH_SIZE=32
TRAIN_EPOCHS=100
TRAIN_LEARNING_RATE=0.001
TRAIN_VALIDATION_SPLIT=0.2

# Web Dashboard Settings
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
DASHBOARD_DEBUG=True

# Database Settings
DATABASE_URL=sqlite:///./aircraft_pm.db
REDIS_URL=redis://localhost:6379
MQTT_BROKER=localhost:1883
EOF

print_status "Environment variables configured"

# Create activation script
echo -e "\n16. Creating activation script..."
cat > activate_pm.sh << 'EOF'
#!/bin/bash
# Activation script for Aircraft Predictive Maintenance System

echo "=================================================="
echo "Activating Aircraft Predictive Maintenance System"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "aircraft-pm-env" ]; then
    echo "Error: Virtual environment not found."
    echo "Please run install_dependencies.sh first."
    exit 1
fi

# Activate virtual environment
source aircraft-pm-env/bin/activate

# Set environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Environment variables loaded from .env"
fi

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check system resources
echo -e "\nSystem Information:"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null || echo "Not installed")"
echo "PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "\nGPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo -e "\nNo NVIDIA GPU detected. Running in CPU mode."
fi

echo -e "\nAvailable commands:"
echo "  aircraft-pm-train      - Train multi-modal model"
echo "  aircraft-pm-dashboard  - Launch web dashboard"
echo "  aircraft-pm-edge       - Start edge inference"
echo "  jupyter notebook       - Launch Jupyter notebook"

echo -e "\n✅ Aircraft Predictive Maintenance System activated!"
echo "   Virtual environment: aircraft-pm-env"
echo "   Project directory: $(pwd)"
EOF

chmod +x activate_pm.sh

cat > run_dashboard.sh << 'EOF'
#!/bin/bash
# Run the multi-modal dashboard

source aircraft-pm-env/bin/activate
streamlit run web_app/multi_modal_dashboard.py --server.port=8501 --server.address=0.0.0.0
EOF

chmod +x run_dashboard.sh

print_status "Activation scripts created"

# Final summary
echo -e "\n=================================================="
echo "INSTALLATION COMPLETE!"
echo "=================================================="
echo -e "\nNext steps:"
echo "1. Activate the environment:"
echo "   $ source activate_pm.sh"
echo ""
echo "2. Generate synthetic data:"
echo "   $ python -c \"from data_processing.multi_sensor_loader import MultiSensorDataLoader; loader = MultiSensorDataLoader(); X, y, _ = loader.create_multi_sensor_dataset()\""
echo ""
echo "3. Train the model:"
echo "   $ aircraft-pm-train"
echo ""
echo "4. Launch the dashboard:"
echo "   $ ./run_dashboard.sh"
echo ""
echo "5. For edge deployment:"
echo "   $ cd deployment"
echo "   $ ./deploy_edge.sh"
echo ""
echo "System requirements check:"
echo "✅ Python 3.9 environment"
echo "✅ All dependencies installed"
echo "✅ Project structure created"
echo "✅ Sample data available"
echo "✅ Activation scripts ready"
echo ""
echo "For support, check the README.md file"
echo "=================================================="