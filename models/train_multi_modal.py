"""
Multi-Modal Model Trainer
Trains hybrid CNN+Transformer model on multi-sensor data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from data_processing.multi_sensor_loader import MultiSensorDataLoader
from models.hybrid_cnn_transformer import MultiModalHybridModel, MultiClassFaultClassifier

class MultiModalTrainer:
    """Trainer for multi-modal aircraft predictive maintenance model"""
    
    def __init__(self, config_path: str = "training_config.json"):
        self.config = self.load_config(config_path)
        self.data_loader = MultiSensorDataLoader()
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs("results/models", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        os.makedirs("results/logs", exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        
        default_config = {
            "aircraft_models": ["Boeing 737 NG", "Cessna 172 Skyhawk", "Airbus H125 (AS350)"],
            "samples_per_model": 10000,
            "test_split": 0.2,
            "val_split": 0.1,
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "early_stopping_patience": 20,
            "model_checkpoint": True,
            "reduce_lr_patience": 10,
            "tensorboard_logging": True,
            "data_augmentation": True,
            "class_weights": "balanced",
            "multi_task_weights": {
                "fault_classification": 1.0,
                "severity_prediction": 0.5,
                "aircraft_classification": 0.3
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
        
        return default_config
    
    def load_or_generate_data(self) -> Tuple:
        """Load existing dataset or generate new one"""
        
        dataset_path = "data/multi_sensor_dataset.h5"
        
        if os.path.exists(dataset_path):
            print("Loading existing dataset...")
            X, y = self.data_loader.load_dataset(dataset_path)
        else:
            print("Generating new dataset...")
            X, y, aircraft_models = self.data_loader.create_multi_sensor_dataset(
                aircraft_models=self.config["aircraft_models"],
                samples_per_model=self.config["samples_per_model"]
            )
            self.data_loader.save_dataset(X, y, dataset_path)
            print(f"Dataset saved to {dataset_path}")
        
        return X, y
    
    def prepare_multi_modal_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Prepare data for multi-modal model"""
        
        n_samples = X.shape[0]
        
        # Extract different sensor modalities (simplified)
        # In real implementation, you would separate based on feature indices
        vibration_features = X[:, :1000]  # First 1000 features for vibration
        thermal_features = X[:, 1000:1500]  # Next 500 for thermal
        acoustic_features = X[:, 1500:1750]  # Next 250 for acoustic
        pressure_features = X[:, 1750:1850]  # Next 100 for pressure
        aircraft_features = X[:, 1850:]  # Remaining for aircraft type
        
        # Reshape for CNN (add channel dimension)
        vibration_data = vibration_features.reshape(-1, 1000, 1)
        thermal_data = thermal_features.reshape(-1, 500, 1)
        acoustic_data = acoustic_features.reshape(-1, 250, 1)
        pressure_data = pressure_features.reshape(-1, 100, 1)
        
        # Prepare labels
        # y contains multiple fault classes (multi-label)
        fault_labels = y[:, :16]  # First 16 columns for fault types
        severity_labels = np.sum(y[:, :16], axis=1, keepdims=True)  # Severity as sum of faults
        
        # Aircraft labels from one-hot encoded features
        aircraft_labels = aircraft_features[:, :3]  # First 3 columns for aircraft type
        
        # Split data
        split_idx = int(n_samples * (1 - self.config["test_split"]))
        val_split_idx = int(split_idx * (1 - self.config["val_split"]))
        
        # Training data
        X_train = [
            vibration_data[:val_split_idx],
            thermal_data[:val_split_idx],
            acoustic_data[:val_split_idx],
            pressure_data[:val_split_idx]
        ]
        
        y_train = {
            'fault_classification': fault_labels[:val_split_idx],
            'severity_prediction': severity_labels[:val_split_idx],
            'aircraft_classification': aircraft_labels[:val_split_idx]
        }
        
        # Validation data
        X_val = [
            vibration_data[val_split_idx:split_idx],
            thermal_data[val_split_idx:split_idx],
            acoustic_data[val_split_idx:split_idx],
            pressure_data[val_split_idx:split_idx]
        ]
        
        y_val = {
            'fault_classification': fault_labels[val_split_idx:split_idx],
            'severity_prediction': severity_labels[val_split_idx:split_idx],
            'aircraft_classification': aircraft_labels[val_split_idx:split_idx]
        }
        
        # Test data
        X_test = [
            vibration_data[split_idx:],
            thermal_data[split_idx:],
            acoustic_data[split_idx:],
            pressure_data[split_idx:]
        ]
        
        y_test = {
            'fault_classification': fault_labels[split_idx:],
            'severity_prediction': severity_labels[split_idx:],
            'aircraft_classification': aircraft_labels[split_idx:]
        }
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'data_shapes': {
                'vibration': vibration_data.shape,
                'thermal': thermal_data.shape,
                'acoustic': acoustic_data.shape,
                'pressure': pressure_data.shape
            }
        }
    
    def create_data_augmentation(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Create augmented data for training"""
        
        if not self.config["data_augmentation"]:
            return data
        
        augmented_data = []
        for modal_data in data:
            n_samples, n_timesteps, n_features = modal_data.shape
            
            # Apply different augmentations
            augmented_modal = []
            
            for i in range(n_samples):
                sample = modal_data[i]
                
                # Add noise
                noisy = sample + np.random.normal(0, 0.01, sample.shape)
                
                # Time shift
                shift = np.random.randint(-10, 10)
                shifted = np.roll(sample, shift, axis=0)
                
                # Scaling
                scale = np.random.uniform(0.9, 1.1)
                scaled = sample * scale
                
                augmented_modal.extend([sample, noisy, shifted, scaled])
            
            augmented_modal = np.array(augmented_modal)
            augmented_data.append(augmented_modal)
        
        return augmented_data
    
    def calculate_class_weights(self, y: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced data"""
        
        if self.config["class_weights"] == "balanced":
            # Calculate class frequencies
            class_counts = np.sum(y, axis=0)
            total_samples = len(y)
            
            # Calculate weights
            weights = {}
            for i in range(len(class_counts)):
                if class_counts[i] > 0:
                    weights[i] = total_samples / (len(class_counts) * class_counts[i])
                else:
                    weights[i] = 1.0
            
            return weights
        else:
            return {i: 1.0 for i in range(y.shape[1])}
    
    def build_model(self, input_shapes: List[Tuple]) -> MultiModalHybridModel:
        """Build the hybrid CNN+Transformer model"""
        
        print("Building hybrid CNN+Transformer model...")
        
        model = MultiModalHybridModel(
            input_shape=input_shapes,
            num_classes=16,  # 16 fault types
            transformer_layers=2,
            num_heads=8,
            ff_dim=256,
            dropout_rate=0.1
        )
        
        # Compile model
        model.compile_model(
            learning_rate=self.config["learning_rate"],
            loss_weights=self.config["multi_task_weights"]
        )
        
        self.model = model
        return model
    
    def train_model(self, train_data: Tuple, val_data: Tuple) -> keras.callbacks.History:
        """Train the model"""
        
        print("\nStarting model training...")
        print(f"Training samples: {len(train_data[0][0])}")
        print(f"Validation samples: {len(val_data[0][0])}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Epochs: {self.config['epochs']}")
        
        # Calculate class weights
        fault_labels = train_data[1]['fault_classification']
        class_weights = self.calculate_class_weights(fault_labels)
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Train the model
        self.history = self.model.train(
            train_data=train_data,
            val_data=val_data,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks
        )
        
        return self.history
    
    def import_nasa_turbofan_dataset(self, filepath="data/NASA_turbofan.csv"):
        """Import actual NASA Turbofan dataset"""
        import pandas as pd
        
        try:
            # Load NASA dataset
            df = pd.read_csv(filepath)
            print(f"✅ NASA Turbofan dataset loaded: {df.shape}")
            
            # Extract features (sensor readings)
            sensor_columns = [col for col in df.columns if 'sensor' in col]
            X = df[sensor_columns].values
            
            # Extract labels (RUL - Remaining Useful Life)
            y = df['RUL'].values
            
            # Convert RUL to fault classes
            y_class = np.zeros_like(y)
            y_class[y < 50] = 1  # Critical fault
            y_class[(y >= 50) & (y < 100)] = 2  # Warning
            y_class[y >= 100] = 0  # Normal
            
            return X, y_class
            
        except Exception as e:
            print(f"❌ Error loading NASA dataset: {e}")
            return None, None
    
    def _create_callbacks(self) -> List:
        """Create training callbacks"""
        
        callbacks = []
        
        # Early stopping
        if self.config.get("early_stopping_patience", 0) > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_fault_classification_accuracy',
                    patience=self.config["early_stopping_patience"],
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Model checkpoint
        if self.config.get("model_checkpoint", False):
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath='results/models/best_model_{epoch:02d}.h5',
                    monitor='val_fault_classification_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
        
        # Reduce learning rate on plateau
        if self.config.get("reduce_lr_patience", 0) > 0:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_fault_classification_loss',
                    factor=0.5,
                    patience=self.config["reduce_lr_patience"],
                    min_lr=1e-6,
                    verbose=1
                )
            )
        
        # TensorBoard logging
        if self.config.get("tensorboard_logging", False):
            log_dir = f"results/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        # CSV logger
        callbacks.append(
            keras.callbacks.CSVLogger(
                'results/logs/training_log.csv',
                separator=',',
                append=False
            )
        )
        
        return callbacks
    
    def evaluate_model(self, test_data: Tuple) -> Dict:
        """Evaluate the trained model"""
        
        print("\nEvaluating model on test data...")
        
        results = self.model.evaluate(test_data)
        
        print("\nTest Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        # Make predictions
        predictions = self.model.predict(
            test_data[0][0],  # vibration
            test_data[0][1],  # thermal
            test_data[0][2],  # acoustic
            test_data[0][3],  # pressure
            batch_size=self.config["batch_size"]
        )
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_true = np.argmax(test_data[1]['fault_classification'], axis=1)
        y_pred = predictions['fault_predictions']
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=[
            f"Fault_{i}" for i in range(16)
        ]))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'metrics': results,
            'predictions': predictions,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot fault classification metrics
        axes[0].plot(history['fault_classification_accuracy'], label='Train')
        axes[0].plot(history['val_fault_classification_accuracy'], label='Validation')
        axes[0].set_title('Fault Classification Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['fault_classification_loss'], label='Train')
        axes[1].plot(history['val_fault_classification_loss'], label='Validation')
        axes[1].set_title('Fault Classification Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot severity prediction metrics
        axes[2].plot(history['severity_prediction_mae'], label='Train')
        axes[2].plot(history['val_severity_prediction_mae'], label='Validation')
        axes[2].set_title('Severity Prediction MAE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAE')
        axes[2].legend()
        axes[2].grid(True)
        
        axes[3].plot(history['severity_prediction_mse'], label='Train')
        axes[3].plot(history['val_severity_prediction_mse'], label='Validation')
        axes[3].set_title('Severity Prediction MSE')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('MSE')
        axes[3].legend()
        axes[3].grid(True)
        
        # Plot aircraft classification metrics
        axes[4].plot(history['aircraft_classification_accuracy'], label='Train')
        axes[4].plot(history['val_aircraft_classification_accuracy'], label='Validation')
        axes[4].set_title('Aircraft Classification Accuracy')
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('Accuracy')
        axes[4].legend()
        axes[4].grid(True)
        
        axes[5].plot(history['aircraft_classification_loss'], label='Train')
        axes[5].plot(history['val_aircraft_classification_loss'], label='Validation')
        axes[5].set_title('Aircraft Classification Loss')
        axes[5].set_xlabel('Epoch')
        axes[5].set_ylabel('Loss')
        axes[5].legend()
        axes[5].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_report(self, evaluation_results: Dict):
        """Save comprehensive training report"""
        
        report = {
            "training_config": self.config,
            "training_history": {
                "final_epoch": len(self.history.history['loss']),
                "final_train_loss": self.history.history['loss'][-1],
                "final_val_loss": self.history.history['val_loss'][-1]
            },
            "evaluation_results": evaluation_results,
            "model_summary": str(self.model.model.summary()),
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info()
        }
        
        report_path = "results/training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Training report saved to {report_path}")
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        
        import platform
        import psutil
        
        return {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "system": platform.system(),
            "processor": platform.processor(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
        }
    
    def run_training_pipeline(self) -> Dict:
        """Run complete training pipeline"""
        
        print("="*60)
        print("MULTI-MODAL AIRCRAFT PREDICTIVE MAINTENANCE TRAINING")
        print("="*60)
        
        # Step 1: Load or generate data
        print("\n1. Loading/Generating dataset...")
        X, y = self.load_or_generate_data()
        print(f"   Dataset shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Step 2: Prepare multi-modal data
        print("\n2. Preparing multi-modal data...")
        data_dict = self.prepare_multi_modal_data(X, y)
        print(f"   Training samples: {len(data_dict['train'][0][0])}")
        print(f"   Validation samples: {len(data_dict['val'][0][0])}")
        print(f"   Test samples: {len(data_dict['test'][0][0])}")
        
        # Step 3: Build model
        print("\n3. Building model...")
        input_shapes = [
            data_dict['data_shapes']['vibration'][1:],
            data_dict['data_shapes']['thermal'][1:],
            data_dict['data_shapes']['acoustic'][1:],
            data_dict['data_shapes']['pressure'][1:]
        ]
        self.build_model(input_shapes)
        
        # Step 4: Train model
        print("\n4. Training model...")
        self.train_model(data_dict['train'], data_dict['val'])
        
        # Step 5: Evaluate model
        print("\n5. Evaluating model...")
        evaluation_results = self.evaluate_model(data_dict['test'])
        
        # Step 6: Save model
        print("\n6. Saving model...")
        model_path = "results/models/final_model.h5"
        self.model.save(model_path)
        print(f"   Model saved to {model_path}")
        
        # Step 7: Generate plots
        print("\n7. Generating training plots...")
        self.plot_training_history()
        
        # Step 8: Save report
        print("\n8. Saving training report...")
        self.save_training_report(evaluation_results)
        
        # Step 9: Check if accuracy target is met
        final_accuracy = evaluation_results['metrics']['fault_classification_accuracy']
        target_accuracy = 0.95  # 95% target
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nFinal Fault Classification Accuracy: {final_accuracy:.4f}")
        print(f"Target Accuracy: {target_accuracy:.4f}")
        
        if final_accuracy >= target_accuracy:
            print("\n✅ SUCCESS: Model meets 95% accuracy target!")
        else:
            print(f"\n⚠️  WARNING: Model accuracy ({final_accuracy:.2%}) below target ({target_accuracy:.2%})")
        
        print("="*60)
        
        return {
            "model": self.model,
            "history": self.history,
            "evaluation": evaluation_results,
            "model_path": model_path,
            "meets_accuracy_target": final_accuracy >= target_accuracy
        }


# Main training script
if __name__ == "__main__":
    # Initialize trainer
    trainer = MultiModalTrainer()
    
    # Run training pipeline
    results = trainer.run_training_pipeline()
    
    # Test fault classifier
    print("\nTesting fault classifier...")
    classifier = MultiClassFaultClassifier(num_fault_classes=16)
    
    # Load test data
    X, y = trainer.load_or_generate_data()
    data_dict = trainer.prepare_multi_modal_data(X, y)
    test_data = data_dict['test']
    
    # Make predictions with confidence
    classifier_results = classifier.predict_with_confidence(
        trainer.model.model,
        test_data[0],
        threshold=0.7
    )
    
    print(f"\nFault Detection Summary:")
    print(f"  Total samples: {len(test_data[0][0])}")
    print(f"  Faults detected: {classifier_results['total_faults']}")
    print(f"  Critical faults: {classifier_results['critical_faults']}")
    print(f"  Average confidence: {classifier_results['average_confidence']:.3f}")
    
    # Save classifier results
    if classifier_results['predictions']:
        predictions_df = pd.DataFrame(classifier_results['predictions'])
        predictions_df.to_csv('results/fault_predictions.csv', index=False)
        print(f"  Predictions saved to results/fault_predictions.csv")
    
    print("\n✅ Multi-modal training pipeline completed successfully!")