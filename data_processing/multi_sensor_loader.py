"""
Multi-Sensor Data Loader and Fusion Module
Handles vibration, thermal, acoustic, pressure sensors
"""

# from h11 import Data
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiSensorDataLoader:
    """Multi-modal sensor data loader with fusion capabilities"""
    
    def __init__(self, config_path="aircraft_configs/"):
        self.config_path = Path(config_path)
        self.sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure']
        self.fusion_methods = ['early', 'late', 'hybrid']
        
    def load_aircraft_config(self, aircraft_model):
        """Load aircraft configuration"""
        config_file = self.config_path / f"{aircraft_model.lower().replace(' ', '_')}.json"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def generate_multi_sensor_data(self, aircraft_model, num_samples=10000, 
                                  fault_percentage=0.15):
        """Generate synthetic multi-sensor data with faults"""
        
        config = self.load_aircraft_config(aircraft_model)
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq='S')
        
        data = {
            'timestamp': timestamps,
            'aircraft_id': np.random.randint(1, 51, num_samples),
            'flight_phase': np.random.choice(['takeoff', 'climb', 'cruise', 'descent', 'landing'], 
                                            num_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        }
        
        # Generate sensor data based on aircraft type
        if aircraft_model == "Boeing 737 NG":
            data.update(self._generate_boeing_sensor_data(config, num_samples, fault_percentage))
        elif aircraft_model == "Cessna 172 Skyhawk":
            data.update(self._generate_cessna_sensor_data(config, num_samples, fault_percentage))
        elif aircraft_model == "Airbus H125 (AS350)":
            data.update(self._generate_airbus_sensor_data(config, num_samples, fault_percentage))
        
        # Generate multi-class fault labels
        data.update(self._generate_fault_labels(config, num_samples, fault_percentage))
        
        return pd.DataFrame(data)
    
    def _generate_boeing_sensor_data(self, config, num_samples, fault_percentage):
        """Generate Boeing 737 multi-sensor data"""
        
        sensor_data = {}
        
        # Vibration sensors
        vibration_config = config['multi_sensor_configuration']['vibration_sensors']
        for sensor_name, spec in vibration_config.items():
            base_freq = np.random.uniform(10, 1000, num_samples)
            amplitude = np.random.normal(0.2, 0.05, num_samples)
            
            # Add faults
            fault_mask = np.random.random(num_samples) < fault_percentage
            amplitude[fault_mask] += np.random.uniform(0.3, 1.0, fault_mask.sum())
            
            # Generate time series
            time = np.arange(num_samples) / spec.get('sampling_rate', 10000)
            signal = amplitude * np.sin(2 * np.pi * base_freq * time[:, None]).mean(axis=1)
            
            sensor_data[f'vib_{sensor_name}'] = signal
        
        # Thermal sensors
        thermal_config = config['multi_sensor_configuration']['thermal_sensors']
        for sensor_name, spec in thermal_config.items():
            base_temp = np.random.normal(
                np.mean(spec['normal_range']),
                (spec['normal_range'][1] - spec['normal_range'][0]) / 6,
                num_samples
            )
            
            # Add faults
            fault_mask = np.random.random(num_samples) < fault_percentage
            base_temp[fault_mask] += np.random.uniform(50, 200, fault_mask.sum())
            
            sensor_data[f'thermal_{sensor_name}'] = base_temp
        
        # Acoustic sensors
        acoustic_config = config['multi_sensor_configuration']['acoustic_sensors']
        for sensor_name, spec in acoustic_config.items():
            base_db = np.random.normal(
                np.mean(spec['normal_range']),
                (spec['normal_range'][1] - spec['normal_range'][0]) / 6,
                num_samples
            )
            
            # Add harmonic components for rotating machinery
            harmonics = sum([
                0.1 * np.sin(2 * np.pi * (i+1) * np.arange(num_samples) / 100)
                for i in range(5)
            ])
            
            base_db += harmonics
            
            # Add faults
            fault_mask = np.random.random(num_samples) < fault_percentage
            base_db[fault_mask] += np.random.uniform(10, 30, fault_mask.sum())
            
            sensor_data[f'acoustic_{sensor_name}'] = base_db
        
        # Pressure sensors
        pressure_config = config['multi_sensor_configuration']['pressure_sensors']
        for sensor_name, spec in pressure_config.items():
            base_pressure = np.random.normal(
                np.mean(spec['normal_range']),
                (spec['normal_range'][1] - spec['normal_range'][0]) / 10,
                num_samples
            )
            
            # Add faults
            fault_mask = np.random.random(num_samples) < fault_percentage
            pressure_change = np.random.choice([-1, 1], fault_mask.sum()) * \
                             np.random.uniform(10, 30, fault_mask.sum())
            base_pressure[fault_mask] += pressure_change
            
            sensor_data[f'pressure_{sensor_name}'] = base_pressure
        
        return sensor_data
    
    def _generate_fault_labels(self, config, num_samples, fault_percentage):
        """Generate multi-class fault labels"""
        
        fault_categories = config['multi_class_fault_categories']
        
        # Initialize labels
        labels = {
            'no_fault': np.ones(num_samples, dtype=int),
            'engine_fault': np.zeros(num_samples, dtype=int),
            'structural_fault': np.zeros(num_samples, dtype=int),
            'systems_fault': np.zeros(num_samples, dtype=int),
            'control_fault': np.zeros(num_samples, dtype=int),
            'fault_severity': np.zeros(num_samples, dtype=int)  # 0: none, 1: low, 2: medium, 3: high
        }
        
        # Generate faults
        n_faults = int(num_samples * fault_percentage)
        fault_indices = np.random.choice(num_samples, n_faults, replace=False)
        
        for idx in fault_indices:
            # Randomly select fault category
            fault_category = np.random.choice(list(fault_categories.keys()))
            
            # Set binary labels
            labels['no_fault'][idx] = 0
            labels[f'{fault_category}_fault'][idx] = 1
            
            # Set severity (based on fault type)
            fault_type = np.random.choice(list(fault_categories[fault_category].keys()))
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 3}
            severity = fault_categories[fault_category][fault_type]['severity']
            labels['fault_severity'][idx] = severity_map.get(severity, 2)
            
            # Add specific fault type
            labels[f'fault_{fault_type}'] = labels.get(f'fault_{fault_type}', np.zeros(num_samples, dtype=int))
            labels[f'fault_{fault_type}'][idx] = 1
        
        return labels
    
    def fuse_sensors(self, sensor_data, method='hybrid'):
        """Fuse multi-sensor data"""
        
        if method == 'early':
            # Early fusion: Concatenate features
            fused_features = np.column_stack([
                sensor_data.filter(like='vib_').values,
                sensor_data.filter(like='thermal_').values,
                sensor_data.filter(like='acoustic_').values,
                sensor_data.filter(like='pressure_').values
            ])
        elif method == 'late':
            # Late fusion: Process each sensor type separately
            vibration_features = self._extract_vibration_features(
                sensor_data.filter(like='vib_').values
            )
            thermal_features = self._extract_thermal_features(
                sensor_data.filter(like='thermal_').values
            )
            acoustic_features = self._extract_acoustic_features(
                sensor_data.filter(like='acoustic_').values
            )
            pressure_features = self._extract_pressure_features(
                sensor_data.filter(like='pressure_').values
            )
            
            fused_features = {
                'vibration': vibration_features,
                'thermal': thermal_features,
                'acoustic': acoustic_features,
                'pressure': pressure_features
            }
            
        elif method == 'hybrid':
            # Hybrid fusion: Combine early and late fusion
            # Time-domain features
            time_features = np.column_stack([
                sensor_data.filter(like='vib_').values.mean(axis=1),
                sensor_data.filter(like='thermal_').values.mean(axis=1),
                sensor_data.filter(like='acoustic_').values.mean(axis=1),
                sensor_data.filter(like='pressure_').values.mean(axis=1)
            ])
            
            # Frequency-domain features for vibration
            vib_data = sensor_data.filter(like='vib_').values
            freq_features = []
            for i in range(vib_data.shape[1]):
                fft_features = np.abs(np.fft.fft(vib_data[:, i]))[:500]  # First 500 frequencies
                freq_features.append(fft_features)
            
            freq_features = np.column_stack(freq_features)[:len(time_features)]
            
            # Statistical features
            stat_features = np.column_stack([
                sensor_data.filter(like='vib_').values.std(axis=1),
                sensor_data.filter(like='thermal_').values.std(axis=1),
                sensor_data.filter(like='acoustic_').values.max(axis=1) - 
                sensor_data.filter(like='acoustic_').values.min(axis=1),
                sensor_data.filter(like='pressure_').values.var(axis=1)
            ])
            
            fused_features = np.column_stack([time_features, freq_features, stat_features])
        
        return fused_features
    
    def _extract_vibration_features(self, vibration_data):
        """Extract vibration signal features"""
        features = []
        
        # Time-domain features
        features.append(vibration_data.mean(axis=0))  # Mean
        features.append(vibration_data.std(axis=0))   # Standard deviation
        features.append(np.sqrt(np.mean(vibration_data**2, axis=0)))  # RMS
        features.append(np.max(np.abs(vibration_data), axis=0))  # Peak value
        
        # Frequency-domain features
        for i in range(vibration_data.shape[1]):
            fft_result = np.fft.fft(vibration_data[:, i])
            freq_magnitude = np.abs(fft_result)[:len(fft_result)//2]
            
            # Dominant frequency
            dominant_freq = np.argmax(freq_magnitude)
            features.append(dominant_freq)
            
            # Spectral centroid
            if np.sum(freq_magnitude) > 0:
                spectral_centroid = np.sum(
                    np.arange(len(freq_magnitude)) * freq_magnitude
                ) / np.sum(freq_magnitude)
                features.append(spectral_centroid)
        
        return np.concatenate(features)
    
    def _extract_thermal_features(self, thermal_data):
        """Extract thermal sensor features"""
        features = []
        
        # Rate of change
        rate_of_change = np.diff(thermal_data, axis=0)
        features.append(np.mean(rate_of_change, axis=0))
        features.append(np.std(rate_of_change, axis=0))
        
        # Temperature gradients
        gradients = np.gradient(thermal_data, axis=0)
        features.append(np.mean(gradients, axis=0))
        
        # Statistical moments
        features.append(np.mean(thermal_data, axis=0))
        features.append(np.std(thermal_data, axis=0))
        features.append(np.max(thermal_data, axis=0))
        features.append(np.min(thermal_data, axis=0))
        
        return np.concatenate(features)
    
    def _extract_acoustic_features(self, acoustic_data):
        """Extract acoustic sensor features"""
        features = []
        
        # Decibel features
        db_data = acoustic_data
        features.append(np.mean(db_data, axis=0))
        features.append(np.std(db_data, axis=0))
        
        # Spectral features
        for i in range(db_data.shape[1]):
            fft_result = np.fft.fft(db_data[:, i])
            freq_magnitude = np.abs(fft_result)[:len(fft_result)//2]
            
            # Spectral rolloff (85th percentile)
            total_energy = np.sum(freq_magnitude)
            cumulative_sum = np.cumsum(freq_magnitude)
            rolloff_index = np.where(cumulative_sum >= 0.85 * total_energy)[0]
            if len(rolloff_index) > 0:
                features.append(rolloff_index[0])
            
            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(freq_magnitude + 1e-10)))
            arithmetic_mean = np.mean(freq_magnitude)
            flatness = geometric_mean / arithmetic_mean
            features.append(flatness)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(db_data), axis=0) != 0, axis=0)
        features.append(zero_crossings / len(db_data))
        
        return np.concatenate(features)
    
    def _extract_pressure_features(self, pressure_data):
        """Extract pressure sensor features"""
        features = []
        
        # Pressure features
        features.append(np.mean(pressure_data, axis=0))
        features.append(np.std(pressure_data, axis=0))
        
        # Pressure differentials
        for i in range(pressure_data.shape[1] - 1):
            differential = pressure_data[:, i] - pressure_data[:, i + 1]
            features.append(np.mean(differential))
            features.append(np.std(differential))
        
        # Rate of pressure change
        pressure_rate = np.diff(pressure_data, axis=0)
        features.append(np.mean(pressure_rate, axis=0))
        features.append(np.max(pressure_rate, axis=0))
        
        # Pressure stability (variance over time windows)
        window_size = 100
        stability = []
        for i in range(0, len(pressure_data), window_size):
            window = pressure_data[i:i+window_size]
            if len(window) > 0:
                stability.append(np.var(window, axis=0))
        
        features.append(np.mean(stability, axis=0))
        
        return np.concatenate(features)
    
    def create_multi_sensor_dataset(self, aircraft_models=None, samples_per_model=5000):
        """Create complete multi-sensor dataset for all aircraft models"""
        
        if aircraft_models is None:
            aircraft_models = ["Boeing 737 NG", "Cessna 172 Skyhawk", "Airbus H125 (AS350)"]
        
        all_data = []
        all_labels = []
        aircraft_ids = []
        
        for aircraft_model in aircraft_models:
            print(f"Generating data for {aircraft_model}...")
            
            # Generate sensor data
            sensor_data = self.generate_multi_sensor_data(
                aircraft_model=aircraft_model,
                num_samples=samples_per_model,
                fault_percentage=0.15
            )
            
            # Extract features and labels
            features = self.fuse_sensors(sensor_data, method='hybrid')
            labels = sensor_data.filter(like='fault_').values
            
            all_data.append(features)
            all_labels.append(labels)
            
            # Add aircraft identifier (one-hot encoding)
            aircraft_id = np.zeros((samples_per_model, len(aircraft_models)))
            aircraft_id[:, aircraft_models.index(aircraft_model)] = 1
            aircraft_ids.append(aircraft_id)
        
        # Combine all data
        X = np.vstack(all_data)
        y = np.vstack(all_labels)
        aircraft_encoded = np.vstack(aircraft_ids)
        
        # Add aircraft type as additional feature
        X = np.column_stack([X, aircraft_encoded])
        
        return X, y, aircraft_models
    
    def save_dataset(self, X, y, filename="multi_sensor_dataset.h5"):
        """Save dataset to HDF5 file"""
        import h5py
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['samples'] = X.shape[0]
            f.attrs['features'] = X.shape[1]
            f.attrs['classes'] = y.shape[1]
        
        print(f"Dataset saved to {filename}")
        print(f"Shape: {X.shape}, Classes: {y.shape[1]}")
    
    def load_dataset(self, filename="multi_sensor_dataset.h5"):
        """Load dataset from HDF5 file"""
        import h5py
        
        with h5py.File(filename, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
            metadata = dict(f.attrs)
        
        print(f"Dataset loaded from {filename}")
        print(f"Shape: {X.shape}, Classes: {y.shape[1]}")
        print(f"Created: {metadata.get('created', 'Unknown')}")
        
        return X, y
    
    def generate_realistic_sensor_data(self, aircraft_model, condition='normal', num_samples=1):
        """Generate realistic sensor data based on aircraft model and condition"""
        
        config = self.load_aircraft_config(aircraft_model)
        sensor_ranges = config.get('real_sensor_ranges', {})
        
        result = {}
        
        # Define condition multipliers
        condition_multipliers = {
            'normal': 1.0,
            'warning': 1.2,
            'critical': 1.5
        }
        
        multiplier = condition_multipliers.get(condition, 1.0)
        
        # Generate vibration data
        if 'vibration' in sensor_ranges:
            for sensor, ranges in sensor_ranges['vibration'].items():
                min_val = ranges['min']
                max_val = ranges['max']
                critical_min = ranges.get('critical_min', max_val * 1.5)
                
                if condition == 'normal':
                    value = np.random.uniform(min_val, max_val * 0.9)
                elif condition == 'warning':
                    value = np.random.uniform(max_val * 0.95, ranges.get('warning', '').split('-')[1] if '-' in ranges.get('warning', '') else max_val * 1.1)
                else:  # critical
                    value = np.random.uniform(critical_min * 0.95, critical_min * 1.1)
                
                result[f'vib_{sensor}'] = round(value, 2)
        
        # Generate thermal data
        if 'thermal' in sensor_ranges:
            for sensor, ranges in sensor_ranges['thermal'].items():
                min_val = ranges['min']
                max_val = ranges['max']
                
                if condition == 'normal':
                    value = np.random.uniform(min_val * 1.1, max_val * 0.9)
                elif condition == 'warning':
                    value = np.random.uniform(max_val * 0.95, max_val * 1.1)
                else:  # critical
                    value = np.random.uniform(max_val * 1.1, max_val * 1.3)
                
                result[f'thermal_{sensor}'] = round(value, 1)
        
        # Generate acoustic data
        if 'acoustic' in sensor_ranges:
            for sensor, ranges in sensor_ranges['acoustic'].items():
                min_val = ranges['min']
                max_val = ranges['max']
                
                if condition == 'normal':
                    value = np.random.uniform(min_val * 1.1, max_val * 0.95)
                elif condition == 'warning':
                    value = np.random.uniform(max_val * 0.98, max_val * 1.05)
                else:  # critical
                    value = np.random.uniform(max_val * 1.05, max_val * 1.2)
                
                result[f'acoustic_{sensor}'] = round(value, 1)
        
        # Generate pressure data
        if 'pressure' in sensor_ranges:
            for sensor, ranges in sensor_ranges['pressure'].items():
                min_val = ranges['min']
                max_val = ranges['max']
                
                if condition == 'normal':
                    value = np.random.uniform(min_val * 1.1, max_val * 0.95)
                elif condition == 'warning':
                    value = np.random.uniform(min_val * 0.9, min_val) or np.random.uniform(max_val, max_val * 1.1)
                else:  # critical
                    value = np.random.uniform(min_val * 0.8, min_val * 0.9) or np.random.uniform(max_val * 1.1, max_val * 1.2)
                
                result[f'pressure_{sensor}'] = round(value, 1)
        
        # Add condition label
        result['condition'] = condition
        result['aircraft_model'] = aircraft_model
        
        return result


# Usage example
if __name__ == "__main__":
    # Initialize the multi-sensor loader
    loader = MultiSensorDataLoader()
    
    # Generate dataset for all aircraft models
    X, y, aircraft_models = loader.create_multi_sensor_dataset(
        aircraft_models=["Boeing 737 NG", "Cessna 172 Skyhawk", "Airbus H125 (AS350)"],
        samples_per_model=2000
    )
    
    # Save the dataset
    loader.save_dataset(X, y, "aircraft_multi_sensor_dataset.h5")
    
    print(f"Dataset created successfully!")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of fault classes: {y.shape[1]}")
    print(f"Aircraft models: {aircraft_models}")