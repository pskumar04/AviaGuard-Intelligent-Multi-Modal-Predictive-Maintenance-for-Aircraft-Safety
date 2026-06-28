"""
Advanced Sensor Fusion Techniques
Early, Late, and Hybrid Fusion for Multi-Modal Data
"""

import numpy as np
# import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import kurtosis, skew
# import pywt
import warnings
warnings.filterwarnings('ignore')

class AdvancedSensorFusion:
    """Advanced multi-sensor fusion techniques"""
    
    def __init__(self, fusion_method='attention_based'):
        self.fusion_method = fusion_method
        self.sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure']
        
    def wavelet_fusion(self, sensor_signals, wavelet='db4', level=3):
        """Wavelet-based sensor fusion"""
        fused_signals = []
        
        for signal_data in sensor_signals:
            # Apply wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # Apply fusion rules (max for approximation, mean for details)
            fused_coeffs = []
            for i, coeff in enumerate(coeffs):
                if i == 0:  # Approximation coefficients
                    fused_coeffs.append(coeff)  # Keep as is
                else:  # Detail coefficients
                    # Use weighted average based on energy
                    energy = np.sum(coeff**2, axis=0)
                    weights = energy / np.sum(energy)
                    fused = np.average(coeff, axis=0, weights=weights)
                    fused_coeffs.append(fused)
            
            # Reconstruct signal
            fused_signal = pywt.waverec(fused_coeffs, wavelet)
            fused_signals.append(fused_signal[:len(signal_data)])
        
        return np.array(fused_signals)
    
    # def deep_fusion_network(self, sensor_features):
    #     """Deep learning based fusion network"""
        
    #     class FusionNetwork(nn.Module):
    #         def __init__(self, input_dims, hidden_dim=128):
    #             super().__init__()
                
    #             # Separate encoders for each sensor type
    #             self.vibration_encoder = nn.Sequential(
    #                 nn.Linear(input_dims[0], hidden_dim),
    #                 nn.BatchNorm1d(hidden_dim),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3)
    #             )
                
    #             self.thermal_encoder = nn.Sequential(
    #                 nn.Linear(input_dims[1], hidden_dim),
    #                 nn.BatchNorm1d(hidden_dim),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3)
    #             )
                
    #             self.acoustic_encoder = nn.Sequential(
    #                 nn.Linear(input_dims[2], hidden_dim),
    #                 nn.BatchNorm1d(hidden_dim),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3)
    #             )
                
    #             self.pressure_encoder = nn.Sequential(
    #                 nn.Linear(input_dims[3], hidden_dim),
    #                 nn.BatchNorm1d(hidden_dim),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3)
    #             )
                
    #             # Attention-based fusion
    #             self.attention = nn.MultiheadAttention(
    #                 embed_dim=hidden_dim,
    #                 num_heads=4,
    #                 dropout=0.1,
    #                 batch_first=True
    #             )
                
    #             # Fusion layer
    #             self.fusion_layer = nn.Sequential(
    #                 nn.Linear(hidden_dim * 4, hidden_dim * 2),
    #                 nn.ReLU(),
    #                 nn.Dropout(0.3),
    #                 nn.Linear(hidden_dim * 2, hidden_dim),
    #                 nn.ReLU()
    #             )
                
    #             # Output layer
    #             self.output_layer = nn.Linear(hidden_dim, input_dims[0])
                
    #         def forward(self, vibration, thermal, acoustic, pressure):
    #             # Encode each sensor modality
    #             vib_encoded = self.vibration_encoder(vibration)
    #             th_encoded = self.thermal_encoder(thermal)
    #             ac_encoded = self.acoustic_encoder(acoustic)
    #             pr_encoded = self.pressure_encoder(pressure)
                
    #             # Stack for attention
    #             sensor_stack = torch.stack([
    #                 vib_encoded, th_encoded, ac_encoded, pr_encoded
    #             ], dim=1)  # [batch, 4, hidden_dim]
                
    #             # Apply attention
    #             attn_output, attn_weights = self.attention(
    #                 sensor_stack, sensor_stack, sensor_stack
    #             )
                
    #             # Flatten and fuse
    #             attn_flattened = attn_output.flatten(start_dim=1)
    #             fused = self.fusion_layer(attn_flattened)
                
    #             # Output
    #             output = self.output_layer(fused)
                
    #             return output, attn_weights
        
    #     # Convert to PyTorch tensors
    #     vib_tensor = torch.FloatTensor(sensor_features[0])
    #     th_tensor = torch.FloatTensor(sensor_features[1])
    #     ac_tensor = torch.FloatTensor(sensor_features[2])
    #     pr_tensor = torch.FloatTensor(sensor_features[3])
        
    #     # Initialize model
    #     input_dims = [
    #         sensor_features[0].shape[1],
    #         sensor_features[1].shape[1],
    #         sensor_features[2].shape[1],
    #         sensor_features[3].shape[1]
    #     ]
        
    #     model = FusionNetwork(input_dims)
        
    #     # Forward pass
    #     with torch.no_grad():
    #         fused_output, attention_weights = model(
    #             vib_tensor, th_tensor, ac_tensor, pr_tensor
    #         )
        
    #     return fused_output.numpy(), attention_weights.numpy()
    
    def kalman_fusion(self, sensor_readings, process_noise=0.1, measurement_noise=0.1):
        """Kalman filter based sensor fusion"""
        
        n_sensors, n_samples = sensor_readings.shape
        
        # Initialize Kalman filter
        x = np.mean(sensor_readings[:, 0])  # Initial state
        P = 1.0  # Initial covariance
        Q = process_noise  # Process noise covariance
        R = measurement_noise  # Measurement noise covariance
        
        fused = np.zeros(n_samples)
        kalman_gains = np.zeros(n_samples)
        
        for t in range(n_samples):
            # Time update (prediction)
            x_pred = x
            P_pred = P + Q
            
            # Measurement update (correction)
            # Average of all sensor readings
            z = np.mean(sensor_readings[:, t])
            
            # Kalman gain
            K = P_pred / (P_pred + R)
            kalman_gains[t] = K
            
            # State update
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            
            fused[t] = x
        
        return fused, kalman_gains
    
    def d_s_evidence_fusion(self, sensor_evidence):
        """Dempster-Shafer evidence theory fusion"""
        
        n_sensors, n_classes = sensor_evidence.shape
        
        # Normalize evidence
        evidence_norm = sensor_evidence / np.sum(sensor_evidence, axis=1, keepdims=True)
        
        # Initialize combined belief
        combined_belief = np.ones(n_classes)
        
        # Apply Dempster's rule of combination
        for i in range(n_sensors):
            sensor_belief = evidence_norm[i]
            
            # Calculate conflict
            conflict = 1 - np.sum(combined_belief * sensor_belief)
            
            # Combine if not completely conflicting
            if conflict < 0.999:
                combined_belief = (combined_belief * sensor_belief) / (1 - conflict)
            else:
                # If high conflict, use weighted average
                combined_belief = 0.5 * combined_belief + 0.5 * sensor_belief
        
        return combined_belief / np.sum(combined_belief)
    
    def adaptive_weighted_fusion(self, sensor_data, reliability_scores):
        """Adaptive weighted fusion based on sensor reliability"""
        
        n_sensors, n_features = sensor_data.shape
        
        # Calculate adaptive weights
        weights = reliability_scores / np.sum(reliability_scores)
        
        # Apply weights
        fused = np.zeros(n_features)
        for i in range(n_sensors):
            fused += weights[i] * sensor_data[i]
        
        # Calculate fusion confidence
        confidence = 1.0 - np.std(reliability_scores) / np.mean(reliability_scores)
        
        return fused, weights, confidence
    
    def time_sync_fusion(self, sensor_streams, sampling_rates):
        """Time synchronization and fusion for heterogeneous sensors"""
        
        # Find common time base (highest sampling rate)
        target_rate = max(sampling_rates)
        n_samples = len(sensor_streams[0])
        
        # Resample all streams to common rate
        synced_streams = []
        for stream, rate in zip(sensor_streams, sampling_rates):
            if rate != target_rate:
                # Resample using linear interpolation
                original_time = np.arange(len(stream)) / rate
                target_time = np.arange(n_samples) / target_rate
                resampled = np.interp(target_time, original_time, stream)
                synced_streams.append(resampled)
            else:
                synced_streams.append(stream)
        
        # Apply fusion
        synced_streams = np.array(synced_streams)
        
        # Use weighted fusion based on sensor quality
        # For now, use simple average
        fused = np.mean(synced_streams, axis=0)
        
        return fused, synced_streams
    
    def extract_time_frequency_features(self, sensor_data, sampling_rate):
        """Extract time-frequency features for fusion"""
        
        features = {}
        
        # Time-domain features
        features['mean'] = np.mean(sensor_data, axis=1)
        features['std'] = np.std(sensor_data, axis=1)
        features['rms'] = np.sqrt(np.mean(sensor_data**2, axis=1))
        features['peak'] = np.max(np.abs(sensor_data), axis=1)
        features['skewness'] = skew(sensor_data, axis=1)
        features['kurtosis'] = kurtosis(sensor_data, axis=1)
        
        # Frequency-domain features
        for i in range(sensor_data.shape[0]):
            fft_result = np.fft.fft(sensor_data[i])
            freq_magnitude = np.abs(fft_result)[:len(fft_result)//2]
            frequencies = np.fft.fftfreq(len(sensor_data[i]), 1/sampling_rate)[:len(freq_magnitude)]
            
            # Dominant frequency
            dominant_idx = np.argmax(freq_magnitude)
            features[f'dominant_freq_{i}'] = frequencies[dominant_idx]
            
            # Spectral centroid
            if np.sum(freq_magnitude) > 0:
                spectral_centroid = np.sum(frequencies * freq_magnitude) / np.sum(freq_magnitude)
                features[f'spectral_centroid_{i}'] = spectral_centroid
            
            # Band energy
            low_band = np.sum(freq_magnitude[frequencies < 100])
            mid_band = np.sum(freq_magnitude[(frequencies >= 100) & (frequencies < 1000)])
            high_band = np.sum(freq_magnitude[frequencies >= 1000])
            
            features[f'low_band_energy_{i}'] = low_band
            features[f'mid_band_energy_{i}'] = mid_band
            features[f'high_band_energy_{i}'] = high_band
        
        return features
    
    def fuse_all_modalities(self, vibration_data, thermal_data, 
                           acoustic_data, pressure_data, method='adaptive'):
        """Main fusion function for all sensor modalities"""
        
        if method == 'adaptive':
            # Estimate reliability scores (simplified)
            vib_reliability = 1.0 - np.std(vibration_data, axis=1) / np.mean(vibration_data, axis=1)
            th_reliability = 1.0 - np.std(thermal_data, axis=1) / np.mean(thermal_data, axis=1)
            ac_reliability = 1.0 - np.std(acoustic_data, axis=1) / np.mean(acoustic_data, axis=1)
            pr_reliability = 1.0 - np.std(pressure_data, axis=1) / np.mean(pressure_data, axis=1)
            
            reliability_scores = np.array([
                np.mean(vib_reliability),
                np.mean(th_reliability),
                np.mean(ac_reliability),
                np.mean(pr_reliability)
            ])
            
            # Prepare data for fusion
            sensor_means = np.array([
                np.mean(vibration_data, axis=1),
                np.mean(thermal_data, axis=1),
                np.mean(acoustic_data, axis=1),
                np.mean(pressure_data, axis=1)
            ])
            
            fused_result, weights, confidence = self.adaptive_weighted_fusion(
                sensor_means, reliability_scores
            )
            
            return {
                'fused_output': fused_result,
                'weights': weights,
                'confidence': confidence,
                'reliability_scores': reliability_scores
            }
        
        elif method == 'deep':
            # Deep learning based fusion
            fused_output, attention_weights = self.deep_fusion_network([
                vibration_data, thermal_data, acoustic_data, pressure_data
            ])
            
            return {
                'fused_output': fused_output,
                'attention_weights': attention_weights,
                'method': 'deep_fusion'
            }
        
        elif method == 'kalman':
            # Kalman filter fusion
            sensor_readings = np.vstack([
                np.mean(vibration_data, axis=1),
                np.mean(thermal_data, axis=1),
                np.mean(acoustic_data, axis=1),
                np.mean(pressure_data, axis=1)
            ])
            
            fused_result, kalman_gains = self.kalman_fusion(sensor_readings)
            
            return {
                'fused_output': fused_result,
                'kalman_gains': kalman_gains,
                'method': 'kalman'
            }
        
        else:
            # Default: simple weighted average
            weights = np.array([0.3, 0.2, 0.3, 0.2])  # vibration, thermal, acoustic, pressure
            fused_result = (
                weights[0] * np.mean(vibration_data, axis=1) +
                weights[1] * np.mean(thermal_data, axis=1) +
                weights[2] * np.mean(acoustic_data, axis=1) +
                weights[3] * np.mean(pressure_data, axis=1)
            )
            
            return {
                'fused_output': fused_result,
                'weights': weights,
                'method': 'weighted_average'
            }


# Test the fusion module
if __name__ == "__main__":
    # Create synthetic sensor data
    np.random.seed(42)
    
    n_samples = 1000
    vibration_data = np.random.normal(0, 1, (4, n_samples))
    thermal_data = np.random.normal(25, 5, (3, n_samples))
    acoustic_data = np.random.normal(70, 10, (2, n_samples))
    pressure_data = np.random.normal(100, 20, (2, n_samples))
    
    # Initialize fusion module
    fusion = AdvancedSensorFusion()
    
    # Test different fusion methods
    print("Testing sensor fusion methods...")
    
    # Adaptive fusion
    result_adaptive = fusion.fuse_all_modalities(
        vibration_data, thermal_data, acoustic_data, pressure_data,
        method='adaptive'
    )
    print(f"\nAdaptive Fusion:")
    print(f"Output shape: {result_adaptive['fused_output'].shape}")
    print(f"Weights: {result_adaptive['weights']}")
    print(f"Confidence: {result_adaptive['confidence']:.3f}")
    
    # Deep fusion
    try:
        result_deep = fusion.fuse_all_modalities(
            vibration_data[:, :100], thermal_data[:, :100],
            acoustic_data[:, :100], pressure_data[:, :100],
            method='deep'
        )
        print(f"\nDeep Fusion:")
        print(f"Output shape: {result_deep['fused_output'].shape}")
    except Exception as e:
        print(f"\nDeep Fusion requires PyTorch installation: {e}")
    
    # Kalman fusion
    result_kalman = fusion.fuse_all_modalities(
        vibration_data, thermal_data, acoustic_data, pressure_data,
        method='kalman'
    )
    print(f"\nKalman Fusion:")
    print(f"Output shape: {result_kalman['fused_output'].shape}")
    
    # Extract time-frequency features
    features = fusion.extract_time_frequency_features(vibration_data, sampling_rate=1000)
    print(f"\nTime-Frequency Features extracted: {len(features)} features")
    
    print("\nSensor fusion testing completed!")