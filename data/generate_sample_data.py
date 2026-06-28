"""
Sample Data Generator
Generates synthetic multi-sensor data for testing and demonstration
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

def generate_multi_sensor_sample_data(
    num_samples: int = 10000,
    fault_percentage: float = 0.15,
    output_dir: str = "data/multi_sensor"
) -> Dict:
    """Generate sample multi-sensor data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples of multi-sensor data...")
    
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_samples)]
    
    # Generate sensor data
    data = {
        'timestamp': timestamps,
        'aircraft_id': np.random.randint(1, 11, num_samples),
        'flight_phase': np.random.choice(
            ['preflight', 'takeoff', 'climb', 'cruise', 'descent', 'landing'],
            num_samples,
            p=[0.05, 0.1, 0.15, 0.4, 0.2, 0.1]
        ),
        'vibration_rms': np.random.normal(0.2, 0.05, num_samples),
        'thermal_celsius': np.random.normal(400, 50, num_samples),
        'acoustic_db': np.random.normal(75, 5, num_samples),
        'pressure_psi': np.random.normal(3000, 200, num_samples)
    }
    
    # Add faults
    fault_mask = np.random.random(num_samples) < fault_percentage
    data['fault_detected'] = fault_mask.astype(int)
    
    # Add fault types
    fault_types = [
        'bearing_wear', 'blade_damage', 'oil_leak', 'compressor_stall',
        'ignition_failure', 'crack_detection', 'corrosion', 'fatigue'
    ]
    
    data['fault_type'] = ['none'] * num_samples
    for i in np.where(fault_mask)[0]:
        data['fault_type'][i] = np.random.choice(fault_types)
    
    # Add severity
    data['severity'] = np.random.uniform(0, 1, num_samples)
    data['severity'][fault_mask] = np.random.uniform(0.5, 1.0, fault_mask.sum())
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "sample_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        'generated': datetime.now().isoformat(),
        'num_samples': num_samples,
        'fault_percentage': fault_percentage,
        'sensors': list(data.keys()),
        'fault_types': fault_types,
        'data_path': csv_path
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Sample data generated:")
    print(f"   CSV file: {csv_path}")
    print(f"   Metadata: {metadata_path}")
    print(f"   Samples: {num_samples}")
    print(f"   Faults: {fault_mask.sum()} ({fault_percentage*100:.1f}%)")
    
    return {
        'dataframe': df,
        'metadata': metadata,
        'csv_path': csv_path,
        'metadata_path': metadata_path
    }

def generate_time_series_data(
    output_dir: str = "data/vibration",
    num_sequences: int = 1000,
    sequence_length: int = 1000
) -> Dict:
    """Generate vibration time series data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_sequences} vibration sequences...")
    
    np.random.seed(42)
    
    sequences = []
    labels = []
    
    for i in range(num_sequences):
        # Generate base signal
        t = np.linspace(0, 1, sequence_length)
        
        # Normal vibration (multiple harmonics)
        if i < num_sequences * 0.85:  # 85% normal
            signal = (
                0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz fundamental
                0.3 * np.sin(2 * np.pi * 100 * t) + # 100 Hz harmonic
                0.2 * np.sin(2 * np.pi * 150 * t) + # 150 Hz harmonic
                np.random.normal(0, 0.1, sequence_length)  # Noise
            )
            label = 0  # Normal
            
        else:  # 15% faulty
            # Add fault characteristics
            fault_freq = np.random.choice([25, 75, 125])  # Different fault frequencies
            signal = (
                0.5 * np.sin(2 * np.pi * 50 * t) +
                0.3 * np.sin(2 * np.pi * 100 * t) +
                0.4 * np.sin(2 * np.pi * fault_freq * t) +  # Fault frequency
                np.random.normal(0, 0.15, sequence_length)  # More noise
            )
            label = 1  # Faulty
        
        sequences.append(signal)
        labels.append(label)
    
    # Save as numpy arrays
    sequences_np = np.array(sequences)
    labels_np = np.array(labels)
    
    sequences_path = os.path.join(output_dir, "vibration_sequences.npy")
    labels_path = os.path.join(output_dir, "vibration_labels.npy")
    
    np.save(sequences_path, sequences_np)
    np.save(labels_path, labels_np)
    
    metadata = {
        'generated': datetime.now().isoformat(),
        'num_sequences': num_sequences,
        'sequence_length': sequence_length,
        'sampling_rate': 1000,  # Hz
        'normal_ratio': 0.85,
        'fault_ratio': 0.15,
        'sequences_path': sequences_path,
        'labels_path': labels_path
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Vibration data generated:")
    print(f"   Sequences: {sequences_np.shape}")
    print(f"   Labels: {labels_np.shape}")
    print(f"   Normal: {(labels_np == 0).sum()}")
    print(f"   Faulty: {(labels_np == 1).sum()}")
    
    return {
        'sequences': sequences_np,
        'labels': labels_np,
        'metadata': metadata
    }

if __name__ == "__main__":
    # Generate all sample data
    print("Generating sample datasets for Aircraft Predictive Maintenance...")
    print("="*60)
    
    # Generate multi-sensor data
    multi_sensor_data = generate_multi_sensor_sample_data(
        num_samples=5000,
        fault_percentage=0.15,
        output_dir="data/multi_sensor"
    )
    
    print("\n" + "="*60)
    
    # Generate vibration time series data
    vibration_data = generate_time_series_data(
        output_dir="data/vibration",
        num_sequences=500,
        sequence_length=1000
    )
    
    print("\n" + "="*60)
    print("✅ All sample data generated successfully!")
    print("\nData directories created:")
    print("  - data/multi_sensor/")
    print("  - data/vibration/")
    print("  - data/thermal/ (empty - add your thermal data)")
    print("  - data/acoustic/ (empty - add your acoustic data)")