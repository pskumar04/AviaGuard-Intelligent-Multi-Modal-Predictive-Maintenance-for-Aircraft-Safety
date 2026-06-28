"""
Multi-Modal Real-Time Dashboard
Streamlit-based dashboard for multi-sensor monitoring and fault diagnosis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import sys
import os
from datetime import datetime, timedelta
import threading
import queue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_cnn_transformer import MultiModalHybridModel, MultiClassFaultClassifier
from data_processing.edge_data_processor import EdgeDataProcessor
from edge.inference_engine import EdgeInferenceEngine

class MultiModalDashboard:
    """Streamlit dashboard for multi-modal aircraft predictive maintenance"""
    
    def __init__(self):
        self.setup_page_config()
        self.load_configurations()
        self.initialize_components()
        
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Aircraft Predictive Maintenance",
            page_icon="✈️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #3B82F6;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #3B82F6;
            margin-bottom: 1rem;
        }
        .fault-card {
            background-color: #FEF2F2;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #DC2626;
            margin-bottom: 1rem;
        }
        .normal-card {
            background-color: #F0F9FF;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #0EA5E9;
            margin-bottom: 1rem;
        }
        .warning-card {
            background-color: #FEF3C7;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #F59E0B;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_configurations(self):
        """Load dashboard configurations"""
        try:
            with open('aircraft_configs/boeing_737_ng.json', 'r') as f:
                self.boeing_config = json.load(f)
            with open('aircraft_configs/cessna_172.json', 'r') as f:
                self.cessna_config = json.load(f)
            with open('aircraft_configs/airbus_h125.json', 'r') as f:
                self.airbus_config = json.load(f)
        except FileNotFoundError:
            st.error("Configuration files not found!")
            self.boeing_config = {}
            self.cessna_config = {}
            self.airbus_config = {}
    
    def initialize_components(self):
        """Initialize dashboard components"""
        # Initialize session state
        if 'dashboard_running' not in st.session_state:
            st.session_state.dashboard_running = False
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = {}
        if 'fault_history' not in st.session_state:
            st.session_state.fault_history = []
        if 'inference_times' not in st.session_state:
            st.session_state.inference_times = []
        
        # Initialize models
        self.model = None
        self.classifier = None
        self.edge_processor = None
        self.inference_engine = None
        
    def load_model(self):
        """Load trained model"""
        try:
            model_path = "results/models/final_model.h5"
            if os.path.exists(model_path):
                self.model = MultiModalHybridModel(
                    input_shape=[(1000, 1), (500, 1), (250, 1), (100, 1)],
                    num_classes=16
                )
                self.model.load(model_path)
                self.classifier = MultiClassFaultClassifier(num_fault_classes=16)
                st.success("✅ Model loaded successfully!")
                return True
            else:
                st.warning("⚠️ Model not found. Please train the model first.")
                return False
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("✈️ Aircraft PM Dashboard")
            
            # Aircraft selection
            st.subheader("Aircraft Selection")
            aircraft_type = st.selectbox(
                "Select Aircraft Model",
                ["Boeing 737 NG", "Cessna 172 Skyhawk", "Airbus H125 (AS350)"]
            )
            
            # Flight phase
            st.subheader("Flight Phase")
            flight_phase = st.select_slider(
                "Current Flight Phase",
                options=["Pre-flight", "Takeoff", "Climb", "Cruise", "Descent", "Landing", "Parked"]
            )
            
            # Sensor selection
            st.subheader("Sensor Configuration")
            col1, col2 = st.columns(2)
            with col1:
                show_vibration = st.checkbox("Vibration", value=True)
                show_thermal = st.checkbox("Thermal", value=True)
            with col2:
                show_acoustic = st.checkbox("Acoustic", value=True)
                show_pressure = st.checkbox("Pressure", value=True)
            
            # Model settings
            st.subheader("Model Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            
            # Dashboard control
            st.subheader("Dashboard Control")
            if st.button("🚀 Start Real-time Monitoring", type="primary"):
                st.session_state.dashboard_running = True
                st.rerun()
            
            if st.button("⏸️ Pause Monitoring"):
                st.session_state.dashboard_running = False
                st.rerun()
            
            if st.button("🔄 Reset Dashboard"):
                st.session_state.fault_history = []
                st.session_state.simulation_data = {}
                st.rerun()
            
            # System status
            st.subheader("System Status")
            st.metric("Model Loaded", "✅" if self.model else "❌")
            st.metric("Real-time", "✅" if st.session_state.dashboard_running else "❌")
            
            # Performance metrics
            if st.session_state.inference_times:
                avg_inference = np.mean(st.session_state.inference_times[-100:])
                st.metric("Avg Inference Time", f"{avg_inference:.2f} ms")
                st.metric("Throughput", f"{1000/avg_inference:.1f} Hz" if avg_inference > 0 else "0 Hz")
            
            return {
                'aircraft_type': aircraft_type,
                'flight_phase': flight_phase,
                'sensors': {
                    'vibration': show_vibration,
                    'thermal': show_thermal,
                    'acoustic': show_acoustic,
                    'pressure': show_pressure
                },
                'confidence_threshold': confidence_threshold
            }
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">✈️ Multi-Modal Aircraft Predictive Maintenance</h1>', 
                   unsafe_allow_html=True)
        
        # Status bar
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Aircraft", "3", "Boeing, Cessna, Airbus")
        with col2:
            st.metric("Total Sensors", "12", "4 per aircraft")
        with col3:
            faults_today = len([f for f in st.session_state.fault_history 
                              if f['timestamp'].date() == datetime.now().date()])
            st.metric("Faults Today", faults_today)
        with col4:
            uptime = "99.8%" if faults_today == 0 else f"{100 - faults_today}%"
            st.metric("System Uptime", uptime)
    
    def generate_sensor_data(self, aircraft_type):
        """Generate simulated sensor data"""
        np.random.seed(int(time.time()))
        
        # Base values based on aircraft type
        if aircraft_type == "Boeing 737 NG":
            base_values = {
                'vibration': 0.2,
                'thermal': 400,
                'acoustic': 75,
                'pressure': 3000
            }
            ranges = {
                'vibration': (0.1, 0.5),
                'thermal': (350, 650),
                'acoustic': (65, 85),
                'pressure': (2800, 3200)
            }
        elif aircraft_type == "Cessna 172 Skyhawk":
            base_values = {
                'vibration': 0.4,
                'thermal': 200,
                'acoustic': 80,
                'pressure': 60
            }
            ranges = {
                'vibration': (0.2, 0.8),
                'thermal': (150, 250),
                'acoustic': (70, 90),
                'pressure': (40, 80)
            }
        else:  # Airbus H125
            base_values = {
                'vibration': 0.3,
                'thermal': 600,
                'acoustic': 85,
                'pressure': 150
            }
            ranges = {
                'vibration': (0.1, 0.5),
                'thermal': (500, 700),
                'acoustic': (75, 95),
                'pressure': (100, 200)
            }
        
        # Generate data with some randomness
        sensor_data = {}
        for sensor, base in base_values.items():
            # Add some noise and trends
            noise = np.random.normal(0, 0.1 * (ranges[sensor][1] - ranges[sensor][0]))
            trend = np.sin(time.time() / 10) * 0.05 * (ranges[sensor][1] - ranges[sensor][0])
            
            value = base + noise + trend
            
            # Ensure within range
            value = max(ranges[sensor][0], min(ranges[sensor][1], value))
            
            sensor_data[sensor] = {
                'value': value,
                'unit': self._get_sensor_unit(sensor),
                'normal_range': ranges[sensor],
                'status': 'normal' if ranges[sensor][0] <= value <= ranges[sensor][1] else 'warning'
            }
        
        return sensor_data
    
    def _get_sensor_unit(self, sensor_type):
        """Get unit for sensor type"""
        units = {
            'vibration': 'g',
            'thermal': '°C',
            'acoustic': 'dB',
            'pressure': 'PSI'
        }
        return units.get(sensor_type, 'units')
    
    def render_sensor_monitoring(self, sensor_data, config):
        """Render sensor monitoring section"""
        st.markdown('<h2 class="sub-header">📊 Real-time Sensor Monitoring</h2>', 
                   unsafe_allow_html=True)
        
        # Create columns for sensors
        cols = st.columns(4)
        
        sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure']
        for idx, sensor in enumerate(sensor_types):
            if config['sensors'][sensor]:
                with cols[idx]:
                    data = sensor_data[sensor]
                    
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=data['value'],
                        title={'text': f"{sensor.title()} ({data['unit']})"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [data['normal_range'][0], data['normal_range'][1]]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [data['normal_range'][0], data['normal_range'][1]], 'color': "lightgreen"},
                                {'range': [data['normal_range'][1], data['normal_range'][1]*1.2], 'color': "yellow"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': data['normal_range'][1]
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Status indicator
                    if data['status'] == 'normal':
                        st.markdown('<div class="normal-card">✅ Normal</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-card">⚠️ Warning</div>', unsafe_allow_html=True)
    
    def render_time_series(self, sensor_data_history, config):
        """Render time series charts"""
        st.markdown('<h2 class="sub-header">📈 Sensor Time Series</h2>', 
                   unsafe_allow_html=True)
        
        # Create time series charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vibration', 'Thermal', 'Acoustic', 'Pressure'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure']
        colors = ['blue', 'red', 'green', 'purple']
        
        for idx, sensor in enumerate(sensor_types):
            if config['sensors'][sensor]:
                row = idx // 2 + 1
                col = idx % 2 + 1
                
                # Get historical data
                times = list(range(len(sensor_data_history)))
                values = [data[sensor]['value'] for data in sensor_data_history]
                
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=values,
                        mode='lines',
                        name=sensor.title(),
                        line=dict(color=colors[idx], width=2),
                        showlegend=True
                    ),
                    row=row, col=col
                )
                
                # Add normal range
                normal_range = sensor_data_history[0][sensor]['normal_range']
                fig.add_hrect(
                    y0=normal_range[0], y1=normal_range[1],
                    fillcolor="lightgreen", opacity=0.2,
                    line_width=0,
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def run_inference(self, sensor_data):
        """Run inference on sensor data"""
        if not self.model:
            return None
        
        try:
            # Prepare input data
            vibration_input = np.array([[sensor_data['vibration']['value']] * 1000]).reshape(1, 1000, 1)
            thermal_input = np.array([[sensor_data['thermal']['value']] * 500]).reshape(1, 500, 1)
            acoustic_input = np.array([[sensor_data['acoustic']['value']] * 250]).reshape(1, 250, 1)
            pressure_input = np.array([[sensor_data['pressure']['value']] * 100]).reshape(1, 100, 1)
            
            # Measure inference time
            start_time = time.time()
            predictions = self.model.predict(
                vibration_input, thermal_input, acoustic_input, pressure_input
            )
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Store inference time
            st.session_state.inference_times.append(inference_time)
            if len(st.session_state.inference_times) > 1000:
                st.session_state.inference_times.pop(0)
            
            # Get fault predictions
            classifier_results = self.classifier.predict_with_confidence(
                self.model.model,
                [vibration_input, thermal_input, acoustic_input, pressure_input],
                threshold=0.7
            )
            
            return {
                'predictions': predictions,
                'classifier_results': classifier_results,
                'inference_time': inference_time
            }
            
        except Exception as e:
            st.error(f"Inference error: {e}")
            return None
    
    def render_fault_detection(self, inference_results, config):
        """Render fault detection section"""
        st.markdown('<h2 class="sub-header">⚠️ Fault Detection & Diagnosis</h2>', 
                   unsafe_allow_html=True)
        
        if not inference_results:
            st.info("No inference results available. Start monitoring to see fault detection.")
            return
        
        results = inference_results['classifier_results']
        
        # Create columns for fault display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Fault summary
            st.metric("Total Faults Detected", results['total_faults'])
            st.metric("Critical Faults", results['critical_faults'])
            st.metric("Average Confidence", f"{results['average_confidence']:.3f}")
            st.metric("Inference Time", f"{inference_results['inference_time']:.2f} ms")
        
        with col2:
            # Performance gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=inference_results['inference_time'],
                title={'text': "Inference Time (ms)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 20], 'color': "yellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed fault information
        if results['predictions']:
            st.markdown('<h3 class="sub-header">📋 Fault Details</h3>', 
                       unsafe_allow_html=True)
            
            for fault in results['predictions'][:5]:  # Show top 5
                if fault['is_critical']:
                    card_class = "fault-card"
                    icon = "🔴"
                else:
                    card_class = "warning-card"
                    icon = "🟡"
                
                st.markdown(f'''
                <div class="{card_class}">
                    <h4>{icon} {fault['fault_type'].replace('_', ' ').title()}</h4>
                    <p><strong>Description:</strong> {fault['fault_description']}</p>
                    <p><strong>Confidence:</strong> {fault['confidence']:.3f}</p>
                    <p><strong>Severity:</strong> {fault['severity'].title()}</p>
                    <p><strong>Recommended Action:</strong> {fault['recommended_action']}</p>
                    <p><strong>Estimated Downtime:</strong> {fault['downtime_estimate']}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Add to history
                fault_record = {
                    'timestamp': datetime.now(),
                    'fault_type': fault['fault_type'],
                    'confidence': fault['confidence'],
                    'severity': fault['severity'],
                    'action': fault['recommended_action']
                }
                st.session_state.fault_history.append(fault_record)
    
    def render_fault_history(self):
        """Render fault history section"""
        if not st.session_state.fault_history:
            return
        
        st.markdown('<h2 class="sub-header">📜 Fault History</h2>', 
                   unsafe_allow_html=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.fault_history)
        
        # Display table
        st.dataframe(
            df.sort_values('timestamp', ascending=False).head(10),
            use_container_width=True
        )
        
        # Create fault timeline
        fig = px.timeline(
            df,
            x_start="timestamp",
            x_end=df["timestamp"] + pd.Timedelta(minutes=5),
            y="fault_type",
            color="severity",
            title="Fault Timeline"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance metrics section"""
        st.markdown('<h2 class="sub-header">⚡ Performance Metrics</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.inference_times:
            st.info("No performance data available yet.")
            return
        
        # Calculate metrics
        inference_times = st.session_state.inference_times
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        throughput = 1000 / avg_time if avg_time > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Inference Time", f"{avg_time:.2f} ms")
        with col2:
            st.metric("Min Inference Time", f"{min_time:.2f} ms")
        with col3:
            st.metric("Max Inference Time", f"{max_time:.2f} ms")
        with col4:
            st.metric("Throughput", f"{throughput:.1f} Hz")
        
        # Create performance chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=inference_times[-100:],  # Last 100 readings
            mode='lines',
            name='Inference Time',
            line=dict(color='blue', width=2)
        ))
        
        # Add target line
        fig.add_hline(y=10, line_dash="dash", line_color="red", 
                     annotation_text="Target: 10ms")
        
        fig.update_layout(
            title="Inference Time Over Time",
            xaxis_title="Sample",
            yaxis_title="Time (ms)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if target is met
        if avg_time < 10:
            st.success(f"✅ Average inference time ({avg_time:.2f} ms) meets <10ms target!")
        else:
            st.warning(f"⚠️ Average inference time ({avg_time:.2f} ms) exceeds 10ms target")
    
    def run_dashboard(self):
        """Main dashboard loop"""
        # Load model
        if not self.model:
            with st.spinner("Loading model..."):
                self.load_model()
        
        # Render sidebar
        config = self.render_sidebar()
        
        # Render header
        self.render_header()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Real-time Monitoring",
            "⚠️ Fault Detection",
            "📜 History & Analytics",
            "⚡ Performance"
        ])
        
        with tab1:
            # Real-time monitoring
            if st.session_state.dashboard_running:
                # Create placeholder for real-time updates
                sensor_placeholder = st.empty()
                time_series_placeholder = st.empty()
                
                # Initialize sensor data history
                if 'sensor_history' not in st.session_state:
                    st.session_state.sensor_history = []
                
                # Simulate real-time updates
                while st.session_state.dashboard_running:
                    # Generate sensor data
                    sensor_data = self.generate_sensor_data(config['aircraft_type'])
                    
                    # Add to history
                    st.session_state.sensor_history.append(sensor_data)
                    if len(st.session_state.sensor_history) > 100:
                        st.session_state.sensor_history.pop(0)
                    
                    # Update sensor display
                    with sensor_placeholder.container():
                        self.render_sensor_monitoring(sensor_data, config)
                    
                    # Update time series
                    with time_series_placeholder.container():
                        self.render_time_series(st.session_state.sensor_history, config)
                    
                    # Wait for next update
                    time.sleep(1)
            else:
                st.info("Click 'Start Real-time Monitoring' to begin")
        
        with tab2:
            # Fault detection
            if st.session_state.dashboard_running and self.model:
                # Create placeholder for inference results
                inference_placeholder = st.empty()
                
                while st.session_state.dashboard_running:
                    # Generate sensor data
                    sensor_data = self.generate_sensor_data(config['aircraft_type'])
                    
                    # Run inference
                    inference_results = self.run_inference(sensor_data)
                    
                    # Display results
                    with inference_placeholder.container():
                        self.render_fault_detection(inference_results, config)
                    
                    time.sleep(2)
            else:
                st.info("Start monitoring and ensure model is loaded for fault detection")
        
        with tab3:
            # History and analytics
            self.render_fault_history()
            
            # Additional analytics
            st.markdown('<h3 class="sub-header">📈 Sensor Analytics</h3>', 
                       unsafe_allow_html=True)
            
            if st.session_state.get('sensor_history'):
                # Calculate statistics
                sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure']
                stats = {}
                
                for sensor in sensor_types:
                    values = [data[sensor]['value'] for data in st.session_state.sensor_history]
                    if values:
                        stats[sensor] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                # Display statistics
                cols = st.columns(4)
                for idx, sensor in enumerate(sensor_types):
                    with cols[idx]:
                        if sensor in stats:
                            st.metric(f"{sensor.title()} Mean", f"{stats[sensor]['mean']:.2f}")
                            st.metric(f"{sensor.title()} Std", f"{stats[sensor]['std']:.2f}")
        
        with tab4:
            # Performance metrics
            self.render_performance_metrics()
            
            # System information
            st.markdown('<h3 class="sub-header">🖥️ System Information</h3>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Aircraft Model:** {config['aircraft_type']}")
                st.info(f"**Flight Phase:** {config['flight_phase']}")
                st.info(f"**Confidence Threshold:** {config['confidence_threshold']}")
            
            with col2:
                st.info("**Active Sensors:** " + ", ".join(
                    [s for s, active in config['sensors'].items() if active]
                ))
                st.info(f"**Model Status:** {'Loaded' if self.model else 'Not Loaded'}")
                st.info(f"**Dashboard Status:** {'Running' if st.session_state.dashboard_running else 'Paused'}")
    
    def main(self):
        """Main entry point"""
        try:
            self.run_dashboard()
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            st.stop()


# Run the dashboard
if __name__ == "__main__":
    dashboard = MultiModalDashboard()
    dashboard.main()