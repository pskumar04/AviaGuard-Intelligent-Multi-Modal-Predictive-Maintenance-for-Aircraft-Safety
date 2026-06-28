"""
Hybrid CNN+Transformer Model for Multi-Modal Aircraft Predictive Maintenance
Multi-class fault diagnosis with attention mechanisms
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Tuple, Dict, Optional

class MultiModalHybridModel:
    """Hybrid CNN+Transformer model for multi-modal sensor data"""
    
    def __init__(self, 
                 input_shape: Tuple,
                 num_classes: int,
                 transformer_layers: int = 2,
                 num_heads: int = 8,
                 ff_dim: int = 256,
                 dropout_rate: float = 0.1):
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.model = self._build_model()
        
    def _cnn_encoder(self, inputs: tf.Tensor, modality_name: str) -> tf.Tensor:
        """CNN encoder for each sensor modality"""
        
        # 1D CNN for time-series sensor data
        x = layers.Conv1D(
            filters=64, 
            kernel_size=3, 
            strides=1, 
            padding='same',
            name=f'{modality_name}_conv1'
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(
            filters=128, 
            kernel_size=3, 
            strides=1, 
            padding='same',
            name=f'{modality_name}_conv2'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(
            filters=256, 
            kernel_size=3, 
            strides=1, 
            padding='same',
            name=f'{modality_name}_conv3'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _transformer_encoder(self, 
                           inputs: tf.Tensor, 
                           sequence_length: int, 
                           hidden_dim: int) -> tf.Tensor:
        """Transformer encoder for temporal dependencies"""
        
        # Positional encoding
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # Add batch dimension
        
        position_embedding = layers.Embedding(
            input_dim=sequence_length, 
            output_dim=hidden_dim
        )(positions)
        
        # Add positional encoding to inputs
        x = inputs + position_embedding
        
        # Transformer blocks
        for i in range(self.transformer_layers):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=hidden_dim // self.num_heads,
                dropout=self.dropout_rate,
                name=f'transformer_block_{i}_attention'
            )(x, x)
            
            # Skip connection and normalization
            x = layers.LayerNormalization(epsilon=1e-6)(
                attention_output + x
            )
            
            # Feed-forward network
            ffn_output = layers.Dense(self.ff_dim, activation='relu')(x)
            ffn_output = layers.Dense(hidden_dim)(ffn_output)
            ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
            
            # Skip connection and normalization
            x = layers.LayerNormalization(epsilon=1e-6)(
                ffn_output + x
            )
        
        return x
    
    def _attention_fusion(self, 
                         modality_features: List[tf.Tensor], 
                         modality_names: List[str]) -> tf.Tensor:
        """Attention-based fusion of multiple modalities"""
        
        # Concatenate all modality features
        concatenated = layers.Concatenate()(modality_features)
        
        # Compute attention weights for each modality
        attention_weights = []
        for modality_feature in modality_features:
            # Learn attention weights for each modality
            attention = layers.Dense(1, activation='tanh')(modality_feature)
            attention_weights.append(attention)
        
        # Softmax over modalities
        attention_weights = layers.Concatenate()(attention_weights)
        attention_weights = layers.Softmax()(attention_weights)
        
        # Split attention weights
        split_weights = tf.split(attention_weights, len(modality_features), axis=1)
        
        # Apply attention weights
        weighted_features = []
        for modality_feature, weight in zip(modality_features, split_weights):
            weighted = modality_feature * weight
            weighted_features.append(weighted)
        
        # Sum weighted features
        fused_features = layers.Add()(weighted_features)
        
        return fused_features, attention_weights
    
    def _build_model(self) -> Model:
        """Build the complete hybrid model"""
        
        # Input layers for each sensor modality
        vibration_input = layers.Input(
            shape=(self.input_shape[0], 1), 
            name='vibration_input'
        )
        thermal_input = layers.Input(
            shape=(self.input_shape[1], 1), 
            name='thermal_input'
        )
        acoustic_input = layers.Input(
            shape=(self.input_shape[2], 1), 
            name='acoustic_input'
        )
        pressure_input = layers.Input(
            shape=(self.input_shape[3], 1), 
            name='pressure_input'
        )
        
        # CNN encoders for each modality
        vibration_features = self._cnn_encoder(vibration_input, 'vibration')
        thermal_features = self._cnn_encoder(thermal_input, 'thermal')
        acoustic_features = self._cnn_encoder(acoustic_input, 'acoustic')
        pressure_features = self._cnn_encoder(pressure_input, 'pressure')
        
        # Reshape for transformer (add sequence dimension)
        vibration_seq = layers.Reshape((1, -1))(vibration_features)
        thermal_seq = layers.Reshape((1, -1))(thermal_features)
        acoustic_seq = layers.Reshape((1, -1))(acoustic_features)
        pressure_seq = layers.Reshape((1, -1))(pressure_features)
        
        # Concatenate all modality sequences
        combined_seq = layers.Concatenate(axis=1)([
            vibration_seq, thermal_seq, acoustic_seq, pressure_seq
        ])
        
        # Transformer encoder for temporal dependencies
        transformer_output = self._transformer_encoder(
            combined_seq, 
            sequence_length=4,  # 4 modalities
            hidden_dim=256
        )
        
        # Global pooling
        pooled_output = layers.GlobalAveragePooling1D()(transformer_output)
        
        # Fully connected layers
        x = layers.Dense(512, activation='relu')(pooled_output)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Multi-task output layer
        # Main fault classification
        fault_output = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='fault_classification'
        )(x)
        
        # Severity regression
        severity_output = layers.Dense(
            1, 
            activation='linear', 
            name='severity_prediction'
        )(x)
        
        # Aircraft type classification (3 aircraft models)
        aircraft_output = layers.Dense(
            3, 
            activation='softmax', 
            name='aircraft_classification'
        )(x)
        
        # Create model
        model = Model(
            inputs=[vibration_input, thermal_input, acoustic_input, pressure_input],
            outputs=[fault_output, severity_output, aircraft_output],
            name='Hybrid_CNN_Transformer'
        )
        
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     loss_weights: Dict = None) -> None:
        """Compile the model with multi-task learning"""
        
        if loss_weights is None:
            loss_weights = {
                'fault_classification': 1.0,
                'severity_prediction': 0.5,
                'aircraft_classification': 0.3
            }
        
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        
        losses = {
            'fault_classification': 'categorical_crossentropy',
            'severity_prediction': 'mse',
            'aircraft_classification': 'categorical_crossentropy'
        }
        
        metrics = {
            'fault_classification': ['accuracy', tf.keras.metrics.Precision(), 
                                   tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
            'severity_prediction': ['mae', 'mse'],
            'aircraft_classification': ['accuracy']
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
    
    def summary(self) -> None:
        """Print model summary"""
        self.model.summary()
    
    def train(self, 
             train_data: Tuple,
             val_data: Tuple,
             epochs: int = 100,
             batch_size: int = 32,
             callbacks: List = None) -> keras.callbacks.History:
        """Train the model"""
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        history = self.model.fit(
            x=train_data[0],
            y=train_data[1],
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _get_default_callbacks(self) -> List:
        """Get default training callbacks"""
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_fault_classification_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath='results/models/best_model.h5',
                monitor='val_fault_classification_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_fault_classification_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='results/logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def evaluate(self, test_data: Tuple) -> Dict:
        """Evaluate the model on test data"""
        
        results = self.model.evaluate(
            x=test_data[0],
            y=test_data[1],
            verbose=1,
            return_dict=True
        )
        
        return results
    
    def predict(self, 
               vibration_data: np.ndarray,
               thermal_data: np.ndarray,
               acoustic_data: np.ndarray,
               pressure_data: np.ndarray,
               batch_size: int = 32) -> Dict:
        """Make predictions on new data"""
        
        predictions = self.model.predict(
            x=[vibration_data, thermal_data, acoustic_data, pressure_data],
            batch_size=batch_size,
            verbose=0
        )
        
        # Format predictions
        result = {
            'fault_probabilities': predictions[0],
            'fault_predictions': np.argmax(predictions[0], axis=1),
            'severity_scores': predictions[1].flatten(),
            'aircraft_probabilities': predictions[2],
            'aircraft_predictions': np.argmax(predictions[2], axis=1)
        }
        
        return result
    
    def save(self, filepath: str) -> None:
        """Save the model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class MultiClassFaultClassifier:
    """Multi-class fault classifier with interpretability"""
    
    def __init__(self, num_fault_classes: int):
        self.num_fault_classes = num_fault_classes
        self.fault_names = [
            'bearing_wear', 'blade_damage', 'oil_leak', 'compressor_stall',
            'ignition_failure', 'crack_detection', 'corrosion', 'fatigue',
            'loose_fasteners', 'hydraulic_leak', 'electrical_short',
            'avionics_failure', 'fuel_system_issue', 'actuator_jam',
            'sensor_failure', 'runaway_trim'
        ]
        
        # Severity mapping
        self.severity_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
    
    def get_fault_description(self, fault_index: int) -> Dict:
        """Get detailed description of a fault"""
        
        fault_descriptions = {
            0: {
                'name': 'bearing_wear',
                'description': 'Excessive wear in engine bearings',
                'symptoms': ['Increased vibration', 'Higher temperature', 'Unusual noise'],
                'severity': 'medium',
                'maintenance_action': 'Replace bearing assembly',
                'estimated_downtime': '24-48 hours'
            },
            1: {
                'name': 'blade_damage',
                'description': 'Damage to compressor or turbine blades',
                'symptoms': ['Reduced efficiency', 'Vibration spikes', 'EGT fluctuations'],
                'severity': 'high',
                'maintenance_action': 'Blade inspection and replacement',
                'estimated_downtime': '3-5 days'
            },
            2: {
                'name': 'oil_leak',
                'description': 'Oil leakage in engine system',
                'symptoms': ['Oil pressure drop', 'Oil temperature increase', 'Visible leaks'],
                'severity': 'medium',
                'maintenance_action': 'Seal replacement and leak check',
                'estimated_downtime': '12-24 hours'
            },
            3: {
                'name': 'compressor_stall',
                'description': 'Compressor stall condition',
                'symptoms': ['Loud banging noise', 'Engine surge', 'Power loss'],
                'severity': 'critical',
                'maintenance_action': 'Immediate engine shutdown and inspection',
                'estimated_downtime': 'Immediate'
            },
            4: {
                'name': 'ignition_failure',
                'description': 'Failure in ignition system',
                'symptoms': ['Engine misfire', 'Hard starting', 'Rough idle'],
                'severity': 'high',
                'maintenance_action': 'Ignition system overhaul',
                'estimated_downtime': '6-12 hours'
            }
        }
        
        return fault_descriptions.get(fault_index, {
            'name': 'unknown_fault',
            'description': 'Unknown fault condition',
            'severity': 'unknown',
            'maintenance_action': 'Further investigation required'
        })
    
    def predict_with_confidence(self, 
                              model: Model, 
                              sensor_data: List[np.ndarray],
                              threshold: float = 0.7) -> Dict:
        """Make predictions with confidence scores"""
        
        predictions = model.predict(sensor_data)
        fault_probs = predictions[0]
        
        results = []
        for i in range(len(fault_probs)):
            fault_idx = np.argmax(fault_probs[i])
            confidence = fault_probs[i][fault_idx]
            
            if confidence >= threshold:
                fault_info = self.get_fault_description(fault_idx)
                result = {
                    'sample_index': i,
                    'fault_type': fault_info['name'],
                    'fault_description': fault_info['description'],
                    'confidence': float(confidence),
                    'severity': fault_info['severity'],
                    'recommended_action': fault_info['maintenance_action'],
                    'downtime_estimate': fault_info.get('estimated_downtime', 'Unknown'),
                    'is_critical': confidence > 0.9
                }
                results.append(result)
        
        return {
            'predictions': results,
            'total_faults': len(results),
            'critical_faults': sum(1 for r in results if r['is_critical']),
            'average_confidence': np.mean([r['confidence'] for r in results]) if results else 0
        }


# Test the hybrid model
if __name__ == "__main__":
    print("Testing Hybrid CNN+Transformer Model...")
    
    # Create synthetic data
    np.random.seed(42)
    
    n_samples = 100
    n_timesteps = 1000
    
    # Generate multi-modal sensor data
    vibration_data = np.random.randn(n_samples, n_timesteps, 1)
    thermal_data = np.random.randn(n_samples, n_timesteps // 2, 1)
    acoustic_data = np.random.randn(n_samples, n_timesteps // 4, 1)
    pressure_data = np.random.randn(n_samples, n_timesteps // 10, 1)
    
    # Generate labels
    fault_labels = np.eye(16)[np.random.randint(0, 16, n_samples)]  # 16 fault classes
    severity_labels = np.random.rand(n_samples, 1)
    aircraft_labels = np.eye(3)[np.random.randint(0, 3, n_samples)]  # 3 aircraft types
    
    print(f"Data shapes:")
    print(f"Vibration: {vibration_data.shape}")
    print(f"Thermal: {thermal_data.shape}")
    print(f"Acoustic: {acoustic_data.shape}")
    print(f"Pressure: {pressure_data.shape}")
    print(f"Fault labels: {fault_labels.shape}")
    print(f"Severity labels: {severity_labels.shape}")
    print(f"Aircraft labels: {aircraft_labels.shape}")
    
    # Initialize model
    input_shapes = [
        vibration_data.shape[1:],
        thermal_data.shape[1:],
        acoustic_data.shape[1:],
        pressure_data.shape[1:]
    ]
    
    model = MultiModalHybridModel(
        input_shape=input_shapes,
        num_classes=16,
        transformer_layers=2,
        num_heads=8,
        ff_dim=256
    )
    
    # Compile model
    model.compile_model(learning_rate=0.001)
    
    # Print model summary
    model.summary()
    
    # Train-test split
    split_idx = int(0.8 * n_samples)
    
    train_data = (
        [
            vibration_data[:split_idx],
            thermal_data[:split_idx],
            acoustic_data[:split_idx],
            pressure_data[:split_idx]
        ],
        {
            'fault_classification': fault_labels[:split_idx],
            'severity_prediction': severity_labels[:split_idx],
            'aircraft_classification': aircraft_labels[:split_idx]
        }
    )
    
    test_data = (
        [
            vibration_data[split_idx:],
            thermal_data[split_idx:],
            acoustic_data[split_idx:],
            pressure_data[split_idx:]
        ],
        {
            'fault_classification': fault_labels[split_idx:],
            'severity_prediction': severity_labels[split_idx:],
            'aircraft_classification': aircraft_labels[split_idx:]
        }
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        train_data=train_data,
        val_data=test_data,
        epochs=5,  # Short training for testing
        batch_size=16
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(test_data)
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(
        vibration_data[:5],
        thermal_data[:5],
        acoustic_data[:5],
        pressure_data[:5]
    )
    
    print(f"Fault predictions: {predictions['fault_predictions'][:5]}")
    print(f"Severity scores: {predictions['severity_scores'][:5]:.3f}")
    print(f"Aircraft predictions: {predictions['aircraft_predictions'][:5]}")
    
    # Test fault classifier
    classifier = MultiClassFaultClassifier(num_fault_classes=16)
    
    print("\nTesting fault classifier...")
    classifier_results = classifier.predict_with_confidence(
        model.model,
        [vibration_data[:5], thermal_data[:5], acoustic_data[:5], pressure_data[:5]],
        threshold=0.5
    )
    
    print(f"Total faults detected: {classifier_results['total_faults']}")
    print(f"Critical faults: {classifier_results['critical_faults']}")
    print(f"Average confidence: {classifier_results['average_confidence']:.3f}")
    
    if classifier_results['predictions']:
        print("\nSample predictions:")
        for pred in classifier_results['predictions'][:3]:
            print(f"  Fault: {pred['fault_type']}")
            print(f"  Confidence: {pred['confidence']:.3f}")
            print(f"  Severity: {pred['severity']}")
            print(f"  Action: {pred['recommended_action']}")
            print()
    
    print("\n✅ Hybrid CNN+Transformer model test completed!")