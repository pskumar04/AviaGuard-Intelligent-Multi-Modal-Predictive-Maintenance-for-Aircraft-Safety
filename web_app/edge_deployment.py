"""
Edge Deployment Interface
Web interface for deploying models to edge devices
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import subprocess
import time
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edge_optimizer import EdgeModelOptimizer

class EdgeDeploymentInterface:
    """Streamlit interface for edge deployment"""
    
    def __init__(self):
        self.setup_page_config()
        self.edge_optimizer = None
        
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Edge Deployment",
            page_icon="🖥️",
            layout="wide"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .deployment-card {
            background-color: #F8FAFC;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            margin-bottom: 1rem;
        }
        .success-card {
            background-color: #D1FAE5;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #10B981;
            margin-bottom: 1rem;
        }
        .warning-card {
            background-color: #FEF3C7;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #F59E0B;
            margin-bottom: 1rem;
        }
        .error-card {
            background-color: #FEE2E2;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #DC2626;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("🖥️ Edge Deployment")
            
            # Model selection
            st.subheader("Model Selection")
            model_files = self.get_model_files()
            selected_model = st.selectbox(
                "Select Model",
                model_files
            )
            
            # Target device
            st.subheader("Target Device")
            target_device = st.selectbox(
                "Select Target Device",
                ["JETSON_NANO", "JETSON_TX2", "JETSON_XAVIER", "RASPBERRY_PI", "CPU_ONLY"]
            )
            
            # Optimization settings
            st.subheader("Optimization Settings")
            precision = st.selectbox(
                "Precision",
                ["FP32", "FP16", "INT8"],
                help="Lower precision reduces model size and improves speed"
            )
            
            optimization_level = st.slider(
                "Optimization Level",
                min_value=1,
                max_value=5,
                value=3,
                help="Higher level = more optimization, but may take longer"
            )
            
            # Deployment options
            st.subheader("Deployment Options")
            generate_tflite = st.checkbox("Generate TFLite", value=True)
            generate_tensorrt = st.checkbox("Generate TensorRT", value=False)
            generate_onnx = st.checkbox("Generate ONNX", value=True)
            
            return {
                'model_path': selected_model,
                'target_device': target_device,
                'precision': precision,
                'optimization_level': optimization_level,
                'formats': {
                    'tflite': generate_tflite,
                    'tensorrt': generate_tensorrt,
                    'onnx': generate_onnx
                }
            }
    
    def get_model_files(self):
        """Get list of available model files"""
        model_dir = "results/models"
        model_files = []
        
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(('.h5', '.keras')):
                    model_files.append(os.path.join(model_dir, file))
        
        if not model_files:
            model_files = ["No models found. Please train a model first."]
        
        return model_files
    
    def render_deployment_status(self):
        """Render deployment status section"""
        st.markdown("## 📊 Deployment Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Available", len(self.get_model_files()) - 1)
        
        with col2:
            optimized_models = len([f for f in os.listdir("models") if "optimized" in f]) if os.path.exists("models") else 0
            st.metric("Optimized Models", optimized_models)
        
        with col3:
            deployment_status = "Ready" if len(self.get_model_files()) > 1 else "No Models"
            st.metric("Deployment Status", deployment_status)
    
    def optimize_model(self, config):
        """Optimize model for edge deployment"""
        st.markdown("## ⚙️ Model Optimization")
        
        with st.spinner("Initializing optimizer..."):
            try:
                self.edge_optimizer = EdgeModelOptimizer(
                    config['model_path'],
                    target_device=config['target_device']
                )
                
                if self.edge_optimizer.model is None:
                    st.error("Failed to load model")
                    return None
                
                st.success("✅ Model loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                return None
        
        # Create optimization report
        with st.expander("Model Information", expanded=True):
            if self.edge_optimizer.model:
                model = self.edge_optimizer.model.model
                st.write(f"**Model Name:** {model.name}")
                st.write(f"**Input Shapes:** {[layer.input_shape for layer in model.layers if hasattr(layer, 'input_shape')][:3]}")
                st.write(f"**Output Shapes:** {[layer.output_shape for layer in model.layers if hasattr(layer, 'output_shape')][:3]}")
                st.write(f"**Total Parameters:** {model.count_params():,}")
        
        # Benchmark original model
        st.markdown("### 📈 Original Model Performance")
        with st.spinner("Benchmarking original model..."):
            original_benchmark = self.edge_optimizer.benchmark_model(
                "original",
                num_iterations=100
            )
            
            if original_benchmark:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Inference Time", f"{original_benchmark['avg_inference_time_ms']:.2f} ms")
                with col2:
                    st.metric("Throughput", f"{original_benchmark['throughput_samples_per_second']:.1f} Hz")
                with col3:
                    meets_target = "✅" if original_benchmark['meets_10ms_target'] else "❌"
                    st.metric("10ms Target", meets_target)
        
        # Select optimization formats
        st.markdown("### 🔧 Select Optimization Formats")
        
        formats_to_generate = []
        
        if config['formats']['tflite']:
            formats_to_generate.append(f"tflite_{config['precision'].lower()}")
        
        if config['formats']['tensorrt']:
            formats_to_generate.append(f"tensorrt_{config['precision'].lower()}")
        
        if config['formats']['onnx']:
            formats_to_generate.append("onnx")
        
        # Run optimization
        if st.button("🚀 Start Optimization", type="primary"):
            with st.spinner("Optimizing model..."):
                try:
                    results = self.edge_optimizer.optimize_for_edge(
                        output_dir=f"models/optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        formats=formats_to_generate
                    )
                    
                    # Display results
                    self.display_optimization_results(results)
                    
                    return results
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {e}")
                    return None
    
    def display_optimization_results(self, results):
        """Display optimization results"""
        st.markdown("## 📊 Optimization Results")
        
        if not results:
            st.warning("No optimization results available")
            return
        
        # Performance comparison
        st.markdown("### ⚡ Performance Comparison")
        
        comparison_data = []
        for format_name, benchmark in results['benchmarks'].items():
            if benchmark:
                comparison_data.append({
                    'Format': format_name,
                    'Avg Time (ms)': benchmark['avg_inference_time_ms'],
                    'Throughput (Hz)': benchmark['throughput_samples_per_second'],
                    'Meets 10ms Target': '✅' if benchmark['meets_10ms_target'] else '❌'
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Create comparison chart
            import plotly.express as px
            
            fig = px.bar(
                df,
                x='Format',
                y='Avg Time (ms)',
                title='Inference Time Comparison',
                color='Meets 10ms Target',
                color_discrete_map={'✅': '#10B981', '❌': '#DC2626'}
            )
            fig.add_hline(y=10, line_dash="dash", line_color="red", 
                         annotation_text="Target: 10ms")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model sizes
        st.markdown("### 💾 Model Sizes")
        
        size_data = []
        for opt in results['optimizations']:
            if opt['success'] and os.path.exists(opt['path']):
                size_mb = os.path.getsize(opt['path']) / (1024 * 1024)
                size_data.append({
                    'Format': f"{opt['format']} ({opt['precision']})",
                    'Size (MB)': size_mb,
                    'Path': opt['path']
                })
        
        if size_data:
            df_sizes = pd.DataFrame(size_data)
            st.dataframe(df_sizes[['Format', 'Size (MB)']], use_container_width=True)
            
            # Size comparison chart
            fig = px.bar(
                df_sizes,
                x='Format',
                y='Size (MB)',
                title='Model Size Comparison',
                color='Format'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download links
        st.markdown("### 📥 Download Optimized Models")
        
        for opt in results['optimizations']:
            if opt['success']:
                with open(opt['path'], 'rb') as f:
                    st.download_button(
                        label=f"Download {opt['format'].upper()} ({opt['precision']})",
                        data=f,
                        file_name=os.path.basename(opt['path']),
                        mime="application/octet-stream"
                    )
    
    def render_deployment_scripts(self):
        """Render deployment scripts section"""
        st.markdown("## 🚀 Deployment Scripts")
        
        # Generate deployment script
        st.markdown("### 📜 Generate Deployment Script")
        
        script_type = st.selectbox(
            "Select Script Type",
            ["NVIDIA Jetson", "Raspberry Pi", "Docker", "Kubernetes"]
        )
        
        if st.button("Generate Deployment Script"):
            script = self.generate_deployment_script(script_type)
            
            st.code(script, language="bash")
            
            # Download button
            st.download_button(
                label="Download Script",
                data=script,
                file_name=f"deploy_{script_type.lower()}.sh",
                mime="text/x-shellscript"
            )
        
        # Pre-built scripts
        st.markdown("### 📋 Pre-built Scripts")
        
        scripts_dir = "deployment"
        if os.path.exists(scripts_dir):
            scripts = os.listdir(scripts_dir)
            
            for script_file in scripts:
                if script_file.endswith('.sh'):
                    with open(os.path.join(scripts_dir, script_file), 'r') as f:
                        script_content = f.read()
                    
                    with st.expander(f"📜 {script_file}"):
                        st.code(script_content, language="bash")
                        
                        # Run button
                        if st.button(f"Run {script_file}", key=script_file):
                            with st.spinner(f"Running {script_file}..."):
                                try:
                                    result = subprocess.run(
                                        ["bash", os.path.join(scripts_dir, script_file)],
                                        capture_output=True,
                                        text=True
                                    )
                                    
                                    if result.returncode == 0:
                                        st.success(f"✅ {script_file} executed successfully!")
                                        st.code(result.stdout)
                                    else:
                                        st.error(f"❌ {script_file} failed!")
                                        st.code(result.stderr)
                                        
                                except Exception as e:
                                    st.error(f"Error running script: {e}")
    
    def generate_deployment_script(self, script_type):
        """Generate deployment script"""
        
        if script_type == "NVIDIA Jetson":
            return """#!/bin/bash
# Deployment script for NVIDIA Jetson devices

echo "Installing dependencies for NVIDIA Jetson..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

echo "Installing TensorFlow for Jetson..."
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow

echo "Installing other dependencies..."
pip3 install numpy pandas scikit-learn
pip3 install opencv-python pyserial

echo "Copying model files..."
cp models/optimized_model.tflite /home/nvidia/
cp -r web_app/ /home/nvidia/

echo "Setting up system service..."
sudo cp deployment/aircraft-pm.service /etc/systemd/system/
sudo systemctl enable aircraft-pm.service

echo "Starting service..."
sudo systemctl start aircraft-pm.service

echo "Deployment complete!"
echo "Dashboard available at: http://<jetson-ip>:8501"
"""
        
        elif script_type == "Raspberry Pi":
            return """#!/bin/bash
# Deployment script for Raspberry Pi

echo "Installing dependencies for Raspberry Pi..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

echo "Installing TensorFlow Lite Runtime..."
pip3 install tflite-runtime

echo "Installing other dependencies..."
pip3 install numpy pandas scikit-learn
pip3 install opencv-python pyserial

echo "Copying model files..."
cp models/optimized_model.tflite /home/pi/
cp -r web_app/ /home/pi/

echo "Setting up auto-start..."
echo "@reboot python3 /home/pi/web_app/multi_modal_dashboard.py" | crontab -

echo "Deployment complete!"
echo "Reboot to start the application."
"""
        
        elif script_type == "Docker":
            return """#!/bin/bash
# Docker deployment script

echo "Building Docker image..."
docker build -t aircraft-pm:latest .

echo "Creating Docker network..."
docker network create aircraft-net

echo "Starting containers..."
docker-compose up -d

echo "Deployment complete!"
echo "Dashboard available at: http://localhost:8501"
"""
        
        else:  # Kubernetes
            return """#!/bin/bash
# Kubernetes deployment script

echo "Creating Kubernetes namespace..."
kubectl create namespace aircraft-pm

echo "Creating config maps..."
kubectl create configmap aircraft-models --from-file=aircraft_configs/ -n aircraft-pm
kubectl create configmap optimized-models --from-file=models/ -n aircraft-pm

echo "Deploying application..."
kubectl apply -f deployment/kubernetes.yaml -n aircraft-pm

echo "Creating service..."
kubectl apply -f deployment/service.yaml -n aircraft-pm

echo "Deployment complete!"
echo "Get service URL: kubectl get svc -n aircraft-pm"
"""
    
    def render_device_monitoring(self):
        """Render device monitoring section"""
        st.markdown("## 📱 Connected Devices")
        
        # Simulated device list
        devices = [
            {"name": "Jetson-TX2-01", "status": "online", "ip": "192.168.1.101", "model": "optimized_model.trt"},
            {"name": "Raspberry-Pi-01", "status": "online", "ip": "192.168.1.102", "model": "optimized_model.tflite"},
            {"name": "Edge-Server-01", "status": "offline", "ip": "192.168.1.103", "model": "original_model.h5"}
        ]
        
        for device in devices:
            status_color = "🟢" if device['status'] == 'online' else "🔴"
            
            with st.expander(f"{status_color} {device['name']} - {device['ip']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {device['status']}")
                    st.write(f"**Model:** {device['model']}")
                
                with col2:
                    if device['status'] == 'online':
                        if st.button(f"Deploy to {device['name']}", key=f"deploy_{device['name']}"):
                            with st.spinner(f"Deploying to {device['name']}..."):
                                time.sleep(2)  # Simulate deployment
                                st.success(f"✅ Deployed to {device['name']}!")
                    else:
                        st.warning("Device offline")
    
    def main(self):
        """Main deployment interface"""
        try:
            # Render sidebar
            config = self.render_sidebar()
            
            # Main content
            st.title("🖥️ Edge Deployment Interface")
            st.markdown("Deploy optimized models to edge devices for real-time inference.")
            
            # Render status
            self.render_deployment_status()
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "⚙️ Model Optimization",
                "🚀 Deployment",
                "📱 Device Management"
            ])
            
            with tab1:
                self.optimize_model(config)
            
            with tab2:
                self.render_deployment_scripts()
            
            with tab3:
                self.render_device_monitoring()
                
        except Exception as e:
            st.error(f"Deployment interface error: {e}")


# Run the deployment interface
if __name__ == "__main__":
    deploy_interface = EdgeDeploymentInterface()
    deploy_interface.main()