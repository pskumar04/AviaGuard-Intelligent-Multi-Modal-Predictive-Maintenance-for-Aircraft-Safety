"""
Edge-Optimized Data Processor
Real-time data processing for edge deployment
"""

import numpy as np
import pandas as pd
from scipy import signal
import threading
import queue
import time
from collections import deque
import json
from pathlib import Path

class EdgeDataProcessor:
    """Real-time data processor optimized for edge devices"""
    
    def __init__(self, config_path="edge_config.json", buffer_size=1000):
        self.buffer_size = buffer_size
        self.config = self.load_config(config_path)
        
        # Initialize sensor buffers
        self.sensor_buffers = {
            'vibration': deque(maxlen=buffer_size),
            'thermal': deque(maxlen=buffer_size),
            'acoustic': deque(maxlen=buffer_size),
            'pressure': deque(maxlen=buffer_size)
        }
        
        # Initialize processing queues
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue(maxsize=1000)
        
        # Thread management
        self.processing_thread = None
        self.running = False
        
        # Performance metrics
        self.metrics = {
            'processing_time': [],
            'samples_processed': 0,
            'throughput': 0.0
        }
    
    def load_config(self, config_path):
        """Load edge processing configuration"""
        default_config = {
            "downsample_rate": 4,
            "window_size": 256,
            "overlap": 0.5,
            "features_to_extract": ["mean", "std", "rms", "peak"],
            "real_time_threshold_ms": 10,
            "max_memory_mb": 512,
            "sensor_priorities": ["vibration", "pressure", "thermal", "acoustic"]
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
        
        return default_config
    
    def add_sensor_data(self, sensor_type, data):
        """Add sensor data to buffer"""
        if sensor_type in self.sensor_buffers:
            if isinstance(data, (list, np.ndarray)):
                self.sensor_buffers[sensor_type].extend(data)
            else:
                self.sensor_buffers[sensor_type].append(data)
            
            # Put in processing queue
            try:
                self.input_queue.put_nowait((sensor_type, data))
            except queue.Full:
                print(f"Input queue full. Dropping {sensor_type} data.")
    
    def extract_features_edge(self, sensor_data, sensor_type):
        """Extract features optimized for edge devices"""
        features = {}
        
        if len(sensor_data) < self.config['window_size']:
            # Use all available data
            window = np.array(sensor_data)
        else:
            # Use latest window
            window = np.array(list(sensor_data)[-self.config['window_size']:])
        
        # Downsample if needed
        if self.config['downsample_rate'] > 1:
            window = signal.decimate(window, self.config['downsample_rate'])
        
        # Extract basic features
        if "mean" in self.config['features_to_extract']:
            features[f'{sensor_type}_mean'] = np.mean(window)
        
        if "std" in self.config['features_to_extract']:
            features[f'{sensor_type}_std'] = np.std(window)
        
        if "rms" in self.config['features_to_extract']:
            features[f'{sensor_type}_rms'] = np.sqrt(np.mean(window**2))
        
        if "peak" in self.config['features_to_extract']:
            features[f'{sensor_type}_peak'] = np.max(np.abs(window))
        
        if "skewness" in self.config['features_to_extract'] and len(window) > 2:
            from scipy.stats import skew
            features[f'{sensor_type}_skew'] = skew(window)
        
        if "kurtosis" in self.config['features_to_extract'] and len(window) > 3:
            from scipy.stats import kurtosis
            features[f'{sensor_type}_kurtosis'] = kurtosis(window)
        
        # Sensor-specific features
        if sensor_type == 'vibration':
            # Simple frequency analysis
            if len(window) >= 64:
                fft_result = np.fft.rfft(window)
                freq_magnitude = np.abs(fft_result)
                if len(freq_magnitude) > 0:
                    features['vib_dominant_freq'] = np.argmax(freq_magnitude)
        
        elif sensor_type == 'acoustic':
            # Simple energy features
            features['acoustic_energy'] = np.sum(window**2)
        
        return features
    
    def real_time_fusion(self, features_dict):
        """Real-time sensor fusion"""
        fused_features = []
        feature_names = []
        
        # Priority-based fusion
        for sensor_type in self.config['sensor_priorities']:
            if sensor_type in features_dict:
                sensor_features = features_dict[sensor_type]
                for feature_name, feature_value in sensor_features.items():
                    fused_features.append(feature_value)
                    feature_names.append(feature_name)
        
        return np.array(fused_features), feature_names
    
    def process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get data from queue with timeout
                sensor_type, data = self.input_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Extract features
                features = self.extract_features_edge(data, sensor_type)
                
                # Update metrics
                processing_time = (time.time() - start_time) * 1000  # ms
                self.metrics['processing_time'].append(processing_time)
                self.metrics['samples_processed'] += 1
                
                # Calculate throughput
                if len(self.metrics['processing_time']) > 100:
                    avg_time = np.mean(self.metrics['processing_time'][-100:])
                    self.metrics['throughput'] = 1000 / avg_time if avg_time > 0 else 0
                
                # Put results in output queue
                result = {
                    'timestamp': time.time(),
                    'sensor_type': sensor_type,
                    'features': features,
                    'processing_time_ms': processing_time
                }
                
                self.output_queue.put_nowait(result)
                
                # Check real-time constraints
                if processing_time > self.config['real_time_threshold_ms']:
                    print(f"Warning: Processing time {processing_time:.2f}ms exceeds threshold")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def start_processing(self):
        """Start the processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Edge processing started")
    
    def stop_processing(self):
        """Stop the processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Edge processing stopped")
    
    def get_latest_features(self, timeout=0.1):
        """Get latest processed features"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_sensor_features(self):
        """Get features from all sensors"""
        all_features = {}
        
        for sensor_type, buffer in self.sensor_buffers.items():
            if len(buffer) > 0:
                features = self.extract_features_edge(buffer, sensor_type)
                all_features[sensor_type] = features
        
        # Fuse all features
        fused_features, feature_names = self.real_time_fusion(all_features)
        
        return {
            'individual_features': all_features,
            'fused_features': fused_features,
            'feature_names': feature_names,
            'timestamp': time.time()
        }
    
    def stream_simulation(self, duration=10, sample_rate=100):
        """Simulate sensor data streaming"""
        print(f"Simulating sensor stream for {duration} seconds...")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            # Generate synthetic sensor data
            t = time.time() - start_time
            
            # Vibration data (sine waves with noise)
            vibration = 0.5 * np.sin(2 * np.pi * 50 * t) + \
                       0.3 * np.sin(2 * np.pi * 120 * t) + \
                       0.1 * np.random.randn()
            
            # Thermal data (slow variation)
            thermal = 25 + 10 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.random.randn()
            
            # Acoustic data
            acoustic = 70 + 5 * np.sin(2 * np.pi * 1 * t) + 2 * np.random.randn()
            
            # Pressure data
            pressure = 100 + 20 * np.sin(2 * np.pi * 0.5 * t) + 3 * np.random.randn()
            
            # Add to buffers
            self.add_sensor_data('vibration', vibration)
            self.add_sensor_data('thermal', thermal)
            self.add_sensor_data('acoustic', acoustic)
            self.add_sensor_data('pressure', pressure)
            
            sample_count += 1
            time.sleep(1 / sample_rate)
        
        print(f"Stream simulation complete. Processed {sample_count} samples.")
        return sample_count
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        if len(self.metrics['processing_time']) > 0:
            avg_processing_time = np.mean(self.metrics['processing_time'])
            max_processing_time = np.max(self.metrics['processing_time'])
            min_processing_time = np.min(self.metrics['processing_time'])
            throughput = self.metrics['throughput']
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
            throughput = 0
        
        return {
            'avg_processing_time_ms': avg_processing_time,
            'max_processing_time_ms': max_processing_time,
            'min_processing_time_ms': min_processing_time,
            'throughput_samples_per_second': throughput,
            'samples_processed': self.metrics['samples_processed'],
            'buffer_sizes': {k: len(v) for k, v in self.sensor_buffers.items()}
        }


# Test the edge processor
if __name__ == "__main__":
    # Initialize edge processor
    processor = EdgeDataProcessor(buffer_size=500)
    
    # Start processing
    processor.start_processing()
    
    # Simulate sensor stream
    processor.stream_simulation(duration=5, sample_rate=50)
    
    # Wait for processing
    time.sleep(2)
    
    # Get latest features
    print("\nGetting latest features...")
    for _ in range(5):
        features = processor.get_latest_features(timeout=0.5)
        if features:
            print(f"Features from {features['sensor_type']}: {list(features['features'].keys())[:3]}")
    
    # Get all sensor features
    print("\nGetting all sensor features...")
    all_features = processor.get_all_sensor_features()
    print(f"Number of fused features: {len(all_features['fused_features'])}")
    print(f"Feature names: {all_features['feature_names'][:5]}...")
    
    # Get performance metrics
    metrics = processor.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"Avg processing time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"Throughput: {metrics['throughput_samples_per_second']:.1f} samples/sec")
    print(f"Target: <{processor.config['real_time_threshold_ms']}ms per sample")
    
    # Stop processing
    processor.stop_processing()
    
    # Check if real-time requirements are met
    if metrics['avg_processing_time_ms'] < processor.config['real_time_threshold_ms']:
        print("\n✅ Real-time requirements MET!")
    else:
        print(f"\n⚠️ Real-time requirements NOT met. Average time: {metrics['avg_processing_time_ms']:.2f}ms")