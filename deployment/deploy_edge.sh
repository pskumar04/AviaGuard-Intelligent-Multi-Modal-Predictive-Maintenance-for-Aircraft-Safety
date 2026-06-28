#!/bin/bash

# Multi-Modal Aircraft Predictive Maintenance - Edge Deployment Script
# Deploys optimized models to edge devices (NVIDIA Jetson, Raspberry Pi, etc.)

set -e

echo "=================================================="
echo "Aircraft Predictive Maintenance - Edge Deployment"
echo "=================================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_PATH="models/optimized"
DEPLOYMENT_TARGET="${1:-jetson_tx2}"
INSTALL_PATH="/opt/aircraft-pm"
SERVICE_USER="aircraftpm"
LOG_PATH="/var/log/aircraft-pm"

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

# Detect platform
detect_platform() {
    if [ -f /etc/nv_tegra_release ]; then
        PLATFORM="JETSON"
        JETSON_TYPE=$(cat /etc/nv_tegra_release | cut -d ' ' -f 3)
        case $JETSON_TYPE in
            "32.7.3")  DEVICE="JETSON_NANO" ;;
            "32.7.4")  DEVICE="JETSON_TX2" ;;
            "35.3.1")  DEVICE="JETSON_XAVIER" ;;
            *)         DEVICE="JETSON_UNKNOWN" ;;
        esac
    elif [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model)
        if [[ $MODEL == *"Raspberry Pi"* ]]; then
            PLATFORM="RASPBERRY_PI"
            DEVICE="RASPBERRY_PI"
        elif [[ $MODEL == *"Jetson"* ]]; then
            PLATFORM="JETSON"
            DEVICE="JETSON"
        else
            PLATFORM="OTHER"
            DEVICE="OTHER"
        fi
    else
        PLATFORM="UNKNOWN"
        DEVICE="UNKNOWN"
    fi
    
    echo "Detected: $PLATFORM ($DEVICE)"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python3 not found"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_warning "pip3 not found, installing..."
        sudo apt-get install -y python3-pip
    fi
    
    # Check model files
    if [ -d "$MODEL_PATH" ]; then
        MODEL_COUNT=$(ls -1 $MODEL_PATH/*.tflite $MODEL_PATH/*.trt $MODEL_PATH/*.onnx 2>/dev/null | wc -l)
        if [ $MODEL_COUNT -gt 0 ]; then
            print_status "Found $MODEL_COUNT optimized model(s)"
        else
            print_warning "No optimized models found in $MODEL_PATH"
            print_warning "Please run optimization first"
            exit 1
        fi
    else
        print_error "Model directory $MODEL_PATH not found"
        exit 1
    fi
    
    # Check disk space
    FREE_SPACE=$(df / --output=avail | tail -1)
    if [ $FREE_SPACE -lt 1048576 ]; then  # Less than 1GB
        print_warning "Low disk space: $(($FREE_SPACE/1024))MB available"
    else
        print_status "Disk space: $(($FREE_SPACE/1024/1024))GB available"
    fi
    
    # Check memory
    TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
    if [ $TOTAL_MEM -lt 1024 ]; then  # Less than 1GB
        print_warning "Low memory: ${TOTAL_MEM}MB available"
    else
        print_status "Memory: ${TOTAL_MEM}MB available"
    fi
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    sudo apt-get update -qq
    
    case $PLATFORM in
        "JETSON")
            print_info "Installing Jetson-specific dependencies..."
            sudo apt-get install -y -qq \
                python3.9 \
                python3-pip \
                python3-dev \
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
            ;;
        
        "RASPBERRY_PI")
            print_info "Installing Raspberry Pi-specific dependencies..."
            sudo apt-get install -y -qq \
                python3.9 \
                python3-pip \
                python3-dev \
                libatlas-base-dev \
                libjasper-dev \
                libqtgui4 \
                libqt4-test \
                libhdf5-dev \
                libhdf5-serial-dev \
                libopenblas-dev \
                libblas-dev \
                liblapack-dev \
                libatlas-base-dev \
                gfortran
            ;;
        
        *)
            print_info "Installing general dependencies..."
            sudo apt-get install -y -qq \
                python3.9 \
                python3-pip \
                python3-dev \
                libhdf5-dev \
                libatlas-base-dev \
                gfortran
            ;;
    esac
    
    print_status "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Create virtual environment
    if [ ! -d "$INSTALL_PATH/venv" ]; then
        python3 -m venv $INSTALL_PATH/venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source $INSTALL_PATH/venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install base dependencies
    pip install numpy pandas scikit-learn scipy matplotlib seaborn
    
    # Platform-specific dependencies
    case $PLATFORM in
        "JETSON")
            print_info "Installing Jetson-optimized libraries..."
            # Install TensorFlow for Jetson
            pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 \
                tensorflow==2.13.0
            
            # Try to install TensorRT if available
            if dpkg -l | grep -q tensorrt; then
                pip install tensorrt
            fi
            
            # Install Jetson utilities
            pip install jetson-stats
            ;;
        
        "RASPBERRY_PI")
            print_info "Installing Raspberry Pi-optimized libraries..."
            # Install TFLite Runtime for Pi
            pip install tflite-runtime
            
            # Install CPU-only TensorFlow
            pip install tensorflow==2.13.0
            ;;
        
        *)
            print_info "Installing standard libraries..."
            pip install tensorflow==2.13.0
            ;;
    esac
    
    # Install edge deployment dependencies
    pip install opencv-python pyserial pyzmq paho-mqtt
    
    # Install web framework
    pip install streamlit fastapi uvicorn
    
    # Install project-specific packages
    pip install plotly dash
    
    print_status "Python dependencies installed"
}

# Create service user
create_service_user() {
    print_info "Creating service user..."
    
    if ! id "$SERVICE_USER" &>/dev/null; then
        sudo useradd -r -s /bin/false -m -d $INSTALL_PATH $SERVICE_USER
        print_status "Service user created: $SERVICE_USER"
    else
        print_status "Service user already exists: $SERVICE_USER"
    fi
    
    # Set ownership
    sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_PATH
    sudo chmod 755 $INSTALL_PATH
}

# Copy application files
copy_application_files() {
    print_info "Copying application files..."
    
    # Create directory structure
    sudo mkdir -p $INSTALL_PATH/{models,config,logs,data}
    sudo mkdir -p $LOG_PATH
    
    # Copy models
    print_info "Copying optimized models..."
    sudo cp -r $MODEL_PATH/* $INSTALL_PATH/models/ 2>/dev/null || true
    
    # Copy application code
    print_info "Copying application code..."
    sudo cp -r edge/ $INSTALL_PATH/
    sudo cp -r web_app/ $INSTALL_PATH/
    sudo cp -r aircraft_configs/ $INSTALL_PATH/config/
    sudo cp requirements.txt $INSTALL_PATH/
    sudo cp README.md $INSTALL_PATH/
    
    # Create data directories
    sudo mkdir -p $INSTALL_PATH/data/{multi_sensor,vibration,thermal,acoustic}
    
    # Set permissions
    sudo chown -R $SERVICE_USER:$SERVICE_USER $INSTALL_PATH
    sudo chown -R $SERVICE_USER:$SERVICE_USER $LOG_PATH
    sudo chmod -R 755 $INSTALL_PATH
    
    print_status "Application files copied"
}

# Configure system service
configure_system_service() {
    print_info "Configuring system service..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/aircraft-pm.service > /dev/null << EOF
[Unit]
Description=Aircraft Predictive Maintenance System
After=network.target
Wants=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_PATH
Environment="PATH=$INSTALL_PATH/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=$INSTALL_PATH"
ExecStart=$INSTALL_PATH/venv/bin/python3 -m streamlit run $INSTALL_PATH/web_app/multi_modal_dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10
StandardOutput=append:$LOG_PATH/aircraft-pm.log
StandardError=append:$LOG_PATH/aircraft-pm-error.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$INSTALL_PATH $LOG_PATH

[Install]
WantedBy=multi-user.target
EOF
    
    # Create log rotation
    sudo tee /etc/logrotate.d/aircraft-pm > /dev/null << EOF
$LOG_PATH/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 640 $SERVICE_USER $SERVICE_USER
    sharedscripts
    postrotate
        systemctl reload aircraft-pm.service >/dev/null 2>&1 || true
    endscript
}
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable and start service
    sudo systemctl enable aircraft-pm.service
    sudo systemctl start aircraft-pm.service
    
    print_status "System service configured"
}

# Configure edge inference service
configure_edge_service() {
    print_info "Configuring edge inference service..."
    
    # Create edge service for low-latency inference
    sudo tee /etc/systemd/system/aircraft-pm-edge.service > /dev/null << EOF
[Unit]
Description=Aircraft Predictive Maintenance Edge Inference
After=network.target
Requires=aircraft-pm.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$INSTALL_PATH
Environment="PATH=$INSTALL_PATH/venv/bin"
Environment="EDGE_DEVICE=$DEVICE"
Environment="INFERENCE_MODE=REAL_TIME"
ExecStart=$INSTALL_PATH/venv/bin/python3 $INSTALL_PATH/edge/inference_engine.py
Restart=always
RestartSec=5
StandardOutput=append:$LOG_PATH/edge-inference.log
StandardError=append:$LOG_PATH/edge-inference-error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable edge service
    sudo systemctl enable aircraft-pm-edge.service
    
    print_status "Edge inference service configured"
}

# Configure network settings
configure_network() {
    print_info "Configuring network settings..."
    
    # Get IP address
    IP_ADDR=$(hostname -I | awk '{print $1}')
    
    # Create network configuration
    sudo tee $INSTALL_PATH/config/network_config.json > /dev/null << EOF
{
    "host_ip": "$IP_ADDR",
    "dashboard_port": 8501,
    "api_port": 8000,
    "mqtt_port": 1883,
    "websocket_port": 8765,
    "inference_port": 9000
}
EOF
    
    print_status "Network configured: Dashboard at http://$IP_ADDR:8501"
}

# Run health check
run_health_check() {
    print_info "Running health check..."
    
    # Check services
    if systemctl is-active --quiet aircraft-pm.service; then
        print_status "Main service is running"
    else
        print_error "Main service is not running"
        sudo systemctl status aircraft-pm.service
    fi
    
    # Check ports
    if netstat -tuln | grep -q ":8501"; then
        print_status "Dashboard port (8501) is listening"
    else
        print_warning "Dashboard port (8501) is not listening"
    fi
    
    # Check model files
    MODEL_FILES=$(ls -1 $INSTALL_PATH/models/*.tflite $INSTALL_PATH/models/*.trt 2>/dev/null | wc -l)
    if [ $MODEL_FILES -gt 0 ]; then
        print_status "Found $MODEL_FILES model file(s)"
    else
        print_error "No model files found in $INSTALL_PATH/models"
    fi
    
    # Check disk usage
    DISK_USAGE=$(df $INSTALL_PATH --output=pcent | tail -1 | tr -d ' %')
    if [ $DISK_USAGE -lt 90 ]; then
        print_status "Disk usage: ${DISK_USAGE}%"
    else
        print_warning "High disk usage: ${DISK_USAGE}%"
    fi
    
    # Test inference
    print_info "Testing inference..."
    if [ -f "$INSTALL_PATH/edge/inference_engine.py" ]; then
        timeout 10 $INSTALL_PATH/venv/bin/python3 -c "
import sys
sys.path.append('$INSTALL_PATH')
try:
    from edge.inference_engine import EdgeInferenceEngine
    engine = EdgeInferenceEngine(device='$DEVICE')
    print('✅ Inference engine test passed')
except Exception as e:
    print(f'❌ Inference engine test failed: {e}')
    sys.exit(1)
        " && print_status "Inference test passed" || print_warning "Inference test failed"
    fi
    
    print_status "Health check completed"
}

# Display deployment summary
display_summary() {
    echo ""
    echo "=================================================="
    echo "DEPLOYMENT COMPLETE!"
    echo "=================================================="
    echo ""
    echo "Application Information:"
    echo "  Installation Path: $INSTALL_PATH"
    echo "  Service User: $SERVICE_USER"
    echo "  Log Directory: $LOG_PATH"
    echo ""
    echo "Services Status:"
    echo "  Main Service: $(systemctl is-active aircraft-pm.service)"
    echo "  Edge Service: $(systemctl is-active aircraft-pm-edge.service 2>/dev/null || echo 'Not enabled')"
    echo ""
    echo "Access Information:"
    IP_ADDR=$(hostname -I | awk '{print $1}')
    echo "  Dashboard URL: http://$IP_ADDR:8501"
    echo "  API URL: http://$IP_ADDR:8000"
    echo ""
    echo "Management Commands:"
    echo "  View logs: sudo journalctl -u aircraft-pm.service -f"
    echo "  Restart service: sudo systemctl restart aircraft-pm.service"
    echo "  Stop service: sudo systemctl stop aircraft-pm.service"
    echo "  Check status: sudo systemctl status aircraft-pm.service"
    echo ""
    echo "Next Steps:"
    echo "  1. Open dashboard at http://$IP_ADDR:8501"
    echo "  2. Configure sensors in the web interface"
    echo "  3. Monitor system logs for any issues"
    echo ""
    echo "=================================================="
}

# Main deployment function
main_deployment() {
    print_info "Starting edge deployment..."
    
    # Step 1: Detect platform
    detect_platform
    
    # Step 2: Check prerequisites
    check_prerequisites
    
    # Step 3: Install system dependencies
    install_system_deps
    
    # Step 4: Create service user
    create_service_user
    
    # Step 5: Copy application files
    copy_application_files
    
    # Step 6: Install Python dependencies
    install_python_deps
    
    # Step 7: Configure system service
    configure_system_service
    
    # Step 8: Configure edge service (if applicable)
    if [ "$PLATFORM" = "JETSON" ] || [ "$PLATFORM" = "RASPBERRY_PI" ]; then
        configure_edge_service
    fi
    
    # Step 9: Configure network
    configure_network
    
    # Step 10: Run health check
    run_health_check
    
    # Step 11: Display summary
    display_summary
    
    print_status "Deployment completed successfully!"
}

# Handle command line arguments
case "$1" in
    "install")
        main_deployment
        ;;
    "uninstall")
        print_info "Uninstalling Aircraft Predictive Maintenance..."
        sudo systemctl stop aircraft-pm.service
        sudo systemctl disable aircraft-pm.service
        sudo systemctl stop aircraft-pm-edge.service 2>/dev/null || true
        sudo systemctl disable aircraft-pm-edge.service 2>/dev/null || true
        sudo rm -f /etc/systemd/system/aircraft-pm.service
        sudo rm -f /etc/systemd/system/aircraft-pm-edge.service
        sudo rm -rf $INSTALL_PATH
        sudo rm -rf $LOG_PATH
        sudo userdel $SERVICE_USER 2>/dev/null || true
        print_status "Uninstallation complete"
        ;;
    "status")
        print_info "Checking deployment status..."
        detect_platform
        check_prerequisites
        run_health_check
        ;;
    "update")
        print_info "Updating deployment..."
        # Stop services
        sudo systemctl stop aircraft-pm.service
        sudo systemctl stop aircraft-pm-edge.service 2>/dev/null || true
        
        # Update code
        copy_application_files
        
        # Update dependencies
        source $INSTALL_PATH/venv/bin/activate
        pip install -r $INSTALL_PATH/requirements.txt --upgrade
        
        # Restart services
        sudo systemctl start aircraft-pm.service
        sudo systemctl start aircraft-pm-edge.service 2>/dev/null || true
        
        print_status "Update complete"
        ;;
    "logs")
        print_info "Showing logs..."
        sudo journalctl -u aircraft-pm.service -f
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status|update|logs}"
        echo ""
        echo "Commands:"
        echo "  install   - Install and deploy the application"
        echo "  uninstall - Remove the application completely"
        echo "  status    - Check deployment status"
        echo "  update    - Update the application"
        echo "  logs      - View application logs"
        exit 1
        ;;
esac

exit 0