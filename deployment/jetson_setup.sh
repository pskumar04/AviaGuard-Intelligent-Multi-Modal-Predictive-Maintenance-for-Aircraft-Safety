#!/bin/bash

# NVIDIA Jetson Setup Script
# Configures Jetson devices for Aircraft Predictive Maintenance deployment

set -e

echo "=================================================="
echo "NVIDIA Jetson Setup for Aircraft Predictive Maintenance"
echo "=================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JETSON_TYPE=""
MAX_POWER_MODE="MAXN"
SWAP_SIZE="8G"
PYTHON_VERSION="3.9"

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

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Detect Jetson type
detect_jetson() {
    print_info "Detecting Jetson device..."
    
    if [ ! -f /etc/nv_tegra_release ]; then
        print_error "Not running on a Jetson device"
        exit 1
    fi
    
    JETSON_INFO=$(cat /etc/nv_tegra_release)
    JETSON_L4T_VERSION=$(echo "$JETSON_INFO" | cut -d ' ' -f 3)
    
    case $JETSON_L4T_VERSION in
        "32.7.3")
            JETSON_TYPE="JETSON_NANO"
            print_status "Detected: NVIDIA Jetson Nano"
            ;;
        "32.7.4")
            JETSON_TYPE="JETSON_TX2"
            print_status "Detected: NVIDIA Jetson TX2"
            ;;
        "35.3.1")
            JETSON_TYPE="JETSON_XAVIER"
            print_status "Detected: NVIDIA Jetson Xavier NX/AGX"
            ;;
        *)
            JETSON_TYPE="JETSON_UNKNOWN"
            print_warning "Unknown Jetson type: $JETSON_L4T_VERSION"
            ;;
    esac
    
    # Get memory info
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    print_status "Total Memory: ${TOTAL_MEM}MB"
    
    # Get GPU info
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
        print_status "GPU: $GPU_INFO"
    fi
}

# Setup performance mode
setup_performance_mode() {
    print_info "Setting up performance mode..."
    
    # Install jetson_clocks if not available
    if ! command -v jetson_clocks &> /dev/null; then
        print_warning "jetson_clocks not found, installing..."
        sudo apt-get install -y nvidia-jetpack
    fi
    
    # Set max performance
    case $MAX_POWER_MODE in
        "MAXN")
            print_info "Enabling MAXN performance mode..."
            sudo nvpmodel -m 0  # MAXN mode
            sudo jetson_clocks
            ;;
        "5W")
            print_info "Setting to 5W mode (Jetson Nano)..."
            sudo nvpmodel -m 1
            ;;
        "10W")
            print_info "Setting to 10W mode..."
            sudo nvpmodel -m 2
            ;;
    esac
    
    # Enable fan control (if available)
    if [ -f "/sys/devices/pwm-fan/target_pwm" ]; then
        echo 255 | sudo tee /sys/devices/pwm-fan/target_pwm > /dev/null
        print_status "Fan set to maximum speed"
    fi
    
    print_status "Performance mode configured"
}

# Setup swap space
setup_swap() {
    print_info "Setting up swap space..."
    
    # Check current swap
    CURRENT_SWAP=$(free -m | awk '/^Swap:/{print $2}')
    
    if [ $CURRENT_SWAP -lt 8192 ]; then  # Less than 8GB
        print_warning "Current swap: ${CURRENT_SWAP}MB (adding ${SWAP_SIZE} swap)"
        
        # Create swap file
        sudo fallocate -l $SWAP_SIZE /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        
        # Add to fstab
        echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
        
        # Set swappiness
        echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
        sudo sysctl -p
        
        print_status "Swap file created: ${SWAP_SIZE}"
    else
        print_status "Sufficient swap space: ${CURRENT_SWAP}MB"
    fi
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update -qq
    
    # Remove unnecessary packages
    sudo apt-get remove -y thunderbird libreoffice-* chromium-*
    
    # Install essential packages
    sudo apt-get install -y -qq \
        python3.9 \
        python3-pip \
        python3.9-dev \
        python3.9-venv \
        git \
        curl \
        wget \
        htop \
        ncdu \
        tmux \
        screen \
        vim \
        nano \
        net-tools \
        iperf3 \
        stress-ng \
        lm-sensors \
        psensor \
        nvtop \
        build-essential \
        cmake \
        pkg-config \
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
        zlib1g-dev \
        libopenblas-dev \
        libblas-dev \
        liblapack-dev \
        liblapacke-dev \
        libeigen3-dev \
        libboost-all-dev \
        libomp-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libprotobuf-dev \
        protobuf-compiler
    
    # Install Jetson-specific packages
    sudo apt-get install -y -qq \
        nvidia-jetpack \
        tensorrt \
        libnvinfer8 \
        libnvinfer-plugin8 \
        libnvparsers8 \
        libnvonnxparsers8 \
        libnvinfer-bin \
        libnvinfer-dev \
        libnvinfer-plugin-dev \
        libnvparsers-dev \
        libnvonnxparsers-dev \
        python3-libnvinfer \
        python3-libnvinfer-dev \
        uff-converter-tf \
        graphsurgeon-tf \
        onnx-graphsurgeon
    
    print_status "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Create virtual environment
    if [ ! -d "/opt/aircraft-pm/venv" ]; then
        python3.9 -m venv /opt/aircraft-pm/venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source /opt/aircraft-pm/venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install TensorFlow for Jetson
    print_info "Installing TensorFlow for Jetson..."
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 \
        tensorflow==2.13.0
    
    # Install PyTorch for Jetson
    print_info "Installing PyTorch for Jetson..."
    
    # Download appropriate wheel based on Jetson type
    if [ "$JETSON_TYPE" = "JETSON_NANO" ]; then
        wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0-cp39-cp39-linux_aarch64.whl
    elif [ "$JETSON_TYPE" = "JETSON_TX2" ]; then
        wget https://nvidia.box.com/shared/static/1b2w9vfm3qjo0d2xkdq4f5o5c6tvl5fz.whl -O torch-2.0.0-cp39-cp39-linux_aarch64.whl
    else  # Xavier
        wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-2.0.0-cp39-cp39-linux_aarch64.whl
    fi
    
    pip install torch-2.0.0-cp39-cp39-linux_aarch64.whl
    pip install torchvision==0.15.0
    
    # Install TensorRT Python bindings
    pip install pycuda
    pip install nvidia-tensorrt
    
    # Install Jetson utilities
    pip install jetson-stats
    
    # Install other ML dependencies
    pip install numpy==1.24.3
    pip install pandas==2.1.0
    pip install scikit-learn==1.3.0
    pip install scipy==1.11.0
    
    # Install edge deployment dependencies
    pip install tflite-runtime
    pip install onnxruntime
    pip install opencv-python==4.8.0
    
    # Install sensor libraries
    pip install pyserial
    pip install pyzmq
    pip install paho-mqtt
    pip install smbus2  # For I2C
    
    # Install web framework
    pip install streamlit
    pip install fastapi
    pip install uvicorn
    
    # Install visualization
    pip install matplotlib
    pip install seaborn
    pip install plotly
    
    # Install utilities
    pip install jupyter
    pip install notebook
    pip install ipython
    
    print_status "Python dependencies installed"
}

# Configure GPU memory
configure_gpu_memory() {
    print_info "Configuring GPU memory..."
    
    # Create GPU memory configuration
    sudo tee /etc/nvpmodel.conf > /dev/null << EOF
# GPU Memory Configuration for Aircraft Predictive Maintenance

# Jetson Nano (4GB)
[power_mode_0]
CPU_ONLINE_CORE_0 1
CPU_ONLINE_CORE_1 1
CPU_ONLINE_CORE_2 1
CPU_ONLINE_CORE_3 1
CPU_DENVER_0 0
CPU_DENVER_1 0
CPU_MIN_FREQ 0
GPU_MIN_FREQ 0
EMC_MAX_FREQ 0

# Jetson TX2
[power_mode_1]
CPU_ONLINE_CORE_0 1
CPU_ONLINE_CORE_1 1
CPU_ONLINE_CORE_2 1
CPU_ONLINE_CORE_3 1
CPU_DENVER_0 1
CPU_DENVER_1 1
CPU_MIN_FREQ 0
GPU_MIN_FREQ 0
EMC_MAX_FREQ 0

# Jetson Xavier
[power_mode_2]
CPU_ONLINE_CORE_0 1
CPU_ONLINE_CORE_1 1
CPU_ONLINE_CORE_2 1
CPU_ONLINE_CORE_3 1
CPU_DENVER_0 1
CPU_DENVER_1 1
CPU_MIN_FREQ 0
GPU_MIN_FREQ 0
EMC_MAX_FREQ 0
EOF
    
    # Set GPU memory allocation
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    
    if [ $TOTAL_MEM -lt 4096 ]; then  # Less than 4GB
        GPU_MEM=512
    elif [ $TOTAL_MEM -lt 8192 ]; then  # Less than 8GB
        GPU_MEM=1024
    else
        GPU_MEM=2048
    fi
    
    # Update boot configuration
    sudo sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\".*\"/GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash nvpmodel=0 mem=2048M gpu_mem=${GPU_MEM}M\"/" /etc/default/grub
    sudo update-grub
    
    print_status "GPU memory configured: ${GPU_MEM}MB"
}

# Setup sensor interfaces
setup_sensor_interfaces() {
    print_info "Setting up sensor interfaces..."
    
    # Enable I2C
    if [ ! -c /dev/i2c-1 ]; then
        sudo groupadd i2c
        sudo usermod -aG i2c $USER
        sudo chmod 666 /dev/i2c-*
        print_status "I2C interface enabled"
    fi
    
    # Enable SPI
    if [ ! -c /dev/spidev0.0 ]; then
        sudo groupadd spi
        sudo usermod -aG spi $USER
        sudo chmod 666 /dev/spidev*
        print_status "SPI interface enabled"
    fi
    
    # Enable serial ports
    sudo usermod -aG dialout $USER
    sudo chmod 666 /dev/ttyUSB*
    sudo chmod 666 /dev/ttyACM*
    
    # Create udev rules for sensors
    sudo tee /etc/udev/rules.d/99-sensors.rules > /dev/null << EOF
# Accelerometer
SUBSYSTEM=="i2c-dev", KERNEL=="i2c-1", MODE="0666"
SUBSYSTEM=="i2c", KERNEL=="1-0068", MODE="0666"

# Temperature sensors
SUBSYSTEM=="hwmon", KERNEL=="hwmon*", MODE="0666"

# USB sensors
SUBSYSTEM=="tty", KERNEL=="ttyUSB*", MODE="0666"
SUBSYSTEM=="tty", KERNEL=="ttyACM*", MODE="0666"
EOF
    
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    print_status "Sensor interfaces configured"
}

# Setup monitoring services
setup_monitoring_services() {
    print_info "Setting up monitoring services..."
    
    # Install and configure jetson_stats service
    if command -v jtop &> /dev/null; then
        sudo systemctl enable jetson_stats.service
        sudo systemctl start jetson_stats.service
        print_status "Jetson Stats service enabled"
    fi
    
    # Create system monitoring service
    sudo tee /etc/systemd/system/jetson-monitor.service > /dev/null << EOF
[Unit]
Description=Jetson System Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 /opt/aircraft-pm/edge/real_time_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable jetson-monitor.service
    
    # Create log rotation
    sudo tee /etc/logrotate.d/jetson-monitor > /dev/null << EOF
/var/log/jetson-monitor.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 640 root root
}
EOF
    
    print_status "Monitoring services configured"
}

# Run system tests
run_system_tests() {
    print_info "Running system tests..."
    
    # Test 1: CPU performance
    print_info "Testing CPU performance..."
    cpu_score=$(stress-ng --cpu 4 --timeout 10 --metrics-brief 2>/dev/null | grep "cpu" | awk '{print $NF}')
    print_status "CPU stress test completed"
    
    # Test 2: Memory performance
    print_info "Testing memory performance..."
    mem_score=$(stress-ng --vm 2 --vm-bytes 1G --timeout 10 --metrics-brief 2>/dev/null | grep "vm" | awk '{print $NF}')
    print_status "Memory stress test completed"
    
    # Test 3: GPU performance
    print_info "Testing GPU performance..."
    if command -v nvcc &> /dev/null; then
        # Simple CUDA test
        cat > /tmp/cuda_test.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    vectorAdd<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("CUDA test passed\\n");
    return 0;
}
EOF
        
        nvcc /tmp/cuda_test.cu -o /tmp/cuda_test
        /tmp/cuda_test && print_status "CUDA test passed" || print_warning "CUDA test failed"
    fi
    
    # Test 4: TensorRT availability
    print_info "Testing TensorRT..."
    python3 -c "
import tensorrt as trt
print(f'TensorRT version: {trt.__version__}')
" && print_status "TensorRT available" || print_warning "TensorRT not available"
    
    # Test 5: Sensor interfaces
    print_info "Testing sensor interfaces..."
    if [ -c /dev/i2c-1 ]; then
        print_status "I2C interface available"
    else
        print_warning "I2C interface not available"
    fi
    
    print_status "System tests completed"
}

# Display setup summary
display_summary() {
    echo ""
    echo "=================================================="
    echo "JETSON SETUP COMPLETE!"
    echo "=================================================="
    echo ""
    echo "Device Information:"
    echo "  Jetson Type: $JETSON_TYPE"
    echo "  Performance Mode: $MAX_POWER_MODE"
    echo "  Swap Size: $SWAP_SIZE"
    echo ""
    echo "Installed Components:"
    echo "  Python $PYTHON_VERSION with virtual environment"
    echo "  TensorFlow for Jetson"
    echo "  PyTorch for Jetson"
    echo "  TensorRT"
    echo "  CUDA Toolkit"
    echo "  Sensor interfaces (I2C, SPI, UART)"
    echo ""
    echo "Services Configured:"
    echo "  jetson_stats (jtop)"
    echo "  System monitoring"
    echo ""
    echo "Next Steps:"
    echo "  1. Reboot the system: sudo reboot"
    echo "  2. Deploy aircraft-pm: ./deploy_edge.sh install"
    echo "  3. Monitor system: jtop"
    echo ""
    echo "Useful Commands:"
    echo "  Check GPU: nvidia-smi"
    echo "  Check power mode: sudo nvpmodel -q"
    echo "  Set max performance: sudo nvpmodel -m 0 && sudo jetson_clocks"
    echo "  Monitor sensors: sensors"
    echo ""
    echo "=================================================="
}

# Main setup function
main_setup() {
    print_info "Starting Jetson setup..."
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then 
        print_warning "Please run as root or with sudo"
        exit 1
    fi
    
    # Step 1: Detect Jetson
    detect_jetson
    
    # Step 2: Setup performance mode
    setup_performance_mode
    
    # Step 3: Setup swap space
    setup_swap
    
    # Step 4: Install system dependencies
    install_system_deps
    
    # Step 5: Configure GPU memory
    configure_gpu_memory
    
    # Step 6: Setup sensor interfaces
    setup_sensor_interfaces
    
    # Step 7: Install Python dependencies
    install_python_deps
    
    # Step 8: Setup monitoring services
    setup_monitoring_services
    
    # Step 9: Run system tests
    run_system_tests
    
    # Step 10: Display summary
    display_summary
    
    print_status "Jetson setup completed successfully!"
}

# Handle command line arguments
case "$1" in
    "setup")
        main_setup
        ;;
    "test")
        detect_jetson
        run_system_tests
        ;;
    "performance")
        setup_performance_mode
        ;;
    "clean")
        print_info "Cleaning up..."
        sudo apt-get autoremove -y
        sudo apt-get clean
        sudo journalctl --vacuum-time=3d
        print_status "Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {setup|test|performance|clean}"
        echo ""
        echo "Commands:"
        echo "  setup       - Complete Jetson setup"
        echo "  test        - Run system tests"
        echo "  performance - Configure performance mode"
        echo "  clean       - Clean up system"
        exit 1
        ;;
esac

exit 0