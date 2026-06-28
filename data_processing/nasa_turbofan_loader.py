"""
NASA Turbofan Dataset Loader
Loads and processes NASA Turbofan degradation dataset
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

class NASATurbofanLoader:
    """Load and process NASA Turbofan dataset"""
    
    def __init__(self, data_path="data/NASA_Turbofan/"):
        self.data_path = Path(data_path)
        self.sensor_columns = [
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
            'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
            'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
        ]
        
    def load_training_data(self, dataset_number=1):
        """Load training data for a specific FD dataset"""
        filename = f"train_FD00{dataset_number}.txt"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        # Load data with column names
        column_names = ['unit_id', 'cycle'] + self.sensor_columns + ['op_setting_1', 'op_setting_2', 'op_setting_3']
        df = pd.read_csv(filepath, sep=' ', header=None, names=column_names)
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        print(f"Loaded training data from {filename}: {df.shape}")
        return df
    
    def load_test_data(self, dataset_number=1):
        """Load test data for a specific FD dataset"""
        filename = f"test_FD00{dataset_number}.txt"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        column_names = ['unit_id', 'cycle'] + self.sensor_columns + ['op_setting_1', 'op_setting_2', 'op_setting_3']
        df = pd.read_csv(filepath, sep=' ', header=None, names=column_names)
        df = df.dropna()
        
        print(f"Loaded test data from {filename}: {df.shape}")
        return df
    
    def load_rul_data(self, dataset_number=1):
        """Load remaining useful life (RUL) data"""
        filename = f"RUL_FD00{dataset_number}.txt"
        filepath = self.data_path / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        rul = pd.read_csv(filepath, sep=' ', header=None, names=['rul'])
        rul = rul.dropna()
        
        print(f"Loaded RUL data from {filename}: {rul.shape}")
        return rul
    
    def create_multi_sensor_features(self, df):
        """Extract features from sensor data for training"""
        features = []
        labels = []
        
        # Group by unit_id
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            
            # Extract sensor readings
            sensor_data = unit_data[self.sensor_columns].values
            
            # Create feature window (use last 50 cycles)
            window_size = 50
            if len(sensor_data) > window_size:
                sensor_window = sensor_data[-window_size:]
            else:
                sensor_window = sensor_data
            
            # Flatten features
            features.append(sensor_window.flatten())
            
            # Calculate RUL (remaining useful life)
            max_cycle = unit_data['cycle'].max()
            rul = max_cycle - unit_data['cycle'].values
            labels.append(rul[-1])  # Use final RUL value
        
        # Pad features to same length
        max_len = max([len(f) for f in features])
        features_padded = []
        for f in features:
            if len(f) < max_len:
                f = np.pad(f, (0, max_len - len(f)), 'constant')
            features_padded.append(f)
        
        return np.array(features_padded), np.array(labels)
    
    def prepare_for_aviaGuard(self, dataset_number=1):
        """Prepare NASA Turbofan data for AviaGuard training"""
        
        # Load all data
        train_df = self.load_training_data(dataset_number)
        test_df = self.load_test_data(dataset_number)
        rul_df = self.load_rul_data(dataset_number)
        
        if train_df is None:
            print("Could not load training data")
            return None, None
        
        # Extract features
        X_train, y_train = self.create_multi_sensor_features(train_df)
        
        print(f"Prepared training data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
        
        return X_train, y_train
    
    def get_sensor_statistics(self, dataset_number=1):
        """Get statistical information about sensors"""
        df = self.load_training_data(dataset_number)
        
        if df is None:
            return {}
        
        stats = {}
        for sensor in self.sensor_columns:
            stats[sensor] = {
                'mean': df[sensor].mean(),
                'std': df[sensor].std(),
                'min': df[sensor].min(),
                'max': df[sensor].max(),
                'normal_range': [df[sensor].quantile(0.1), df[sensor].quantile(0.9)]
            }
        
        return stats


# Test the loader
if __name__ == "__main__":
    loader = NASATurbofanLoader()
    
    # Load and prepare data for dataset 1
    X, y = loader.prepare_for_aviaGuard(1)
    
    # Get sensor statistics
    stats = loader.get_sensor_statistics(1)
    print("\nSensor Statistics:")
    for sensor, values in stats.items():
        print(f"{sensor}: {values['normal_range']}")