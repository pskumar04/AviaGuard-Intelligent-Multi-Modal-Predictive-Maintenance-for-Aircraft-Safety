"""
SHAP (SHapley Additive exPlanations) Explainer
Provides feature importance and model interpretability
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """SHAP-based explainer for multi-modal predictive maintenance models"""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or self._generate_feature_names()
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for multi-modal data"""
        
        feature_names = []
        
        # Vibration features
        for i in range(1000):
            feature_names.append(f'vib_freq_{i}')
        
        # Thermal features
        for i in range(500):
            feature_names.append(f'thermal_{i}')
        
        # Acoustic features
        for i in range(250):
            feature_names.append(f'acoustic_{i}')
        
        # Pressure features
        for i in range(100):
            feature_names.append(f'pressure_{i}')
        
        # Aircraft features
        feature_names.extend(['aircraft_boeing', 'aircraft_cessna', 'airbus_h125'])
        
        return feature_names
    
    def fit_explainer(self, X: np.ndarray, sample_size: int = 1000):
        """Fit SHAP explainer to data"""
        
        print("Fitting SHAP explainer...")
        
        # Sample data for faster computation
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            # Scikit-learn model
            self.explainer = shap.Explainer(self.model.predict_proba, X_sample)
        elif hasattr(self.model, 'predict'):
            # TensorFlow/Keras model
            self.explainer = shap.Explainer(self.model.predict, X_sample)
        else:
            # Generic model with predict function
            self.explainer = shap.Explainer(self.model, X_sample)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        self.shap_values = self.explainer(X_sample)
        self.expected_value = self.shap_values.base_values
        
        print(f"✅ SHAP explainer fitted with {len(X_sample)} samples")
        
        return self
    
    def get_feature_importance(self, class_index: int = 0) -> pd.DataFrame:
        """Get feature importance for a specific class"""
        
        if self.shap_values is None:
            raise ValueError("SHAP explainer not fitted. Call fit_explainer first.")
        
        # Calculate mean absolute SHAP values
        if len(self.shap_values.shape) == 3:  # Multi-class
            shap_abs = np.abs(self.shap_values.values[:, :, class_index])
        else:  # Binary
            shap_abs = np.abs(self.shap_values.values)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names[:shap_abs.shape[1]],
            'importance': np.mean(shap_abs, axis=0)
        })
        
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)
        
        return feature_importance
    
    def get_sample_explanation(self, 
                              sample: np.ndarray, 
                              sample_index: int = 0) -> Dict:
        """Get explanation for a specific sample"""
        
        if self.explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit_explainer first.")
        
        # Calculate SHAP values for sample
        sample_shap = self.explainer(sample.reshape(1, -1))
        
        # Get top contributing features
        if len(sample_shap.shape) == 3:  # Multi-class
            shap_values_sample = sample_shap.values[0, :, 0]
        else:  # Binary
            shap_values_sample = sample_shap.values[0, :]
        
        # Get feature contributions
        contributions = []
        for i, value in enumerate(shap_values_sample):
            if abs(value) > 0.001:  # Only significant contributions
                feature_type = self._get_feature_type(i)
                contributions.append({
                    'feature_index': i,
                    'feature_name': self.feature_names[i],
                    'feature_type': feature_type,
                    'contribution': float(value),
                    'absolute_contribution': abs(float(value)),
                    'feature_value': float(sample[i])
                })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x['absolute_contribution'], reverse=True)
        
        # Group by feature type
        contributions_by_type = {}
        for contrib in contributions:
            feature_type = contrib['feature_type']
            if feature_type not in contributions_by_type:
                contributions_by_type[feature_type] = []
            contributions_by_type[feature_type].append(contrib)
        
        # Calculate summary
        total_positive = sum(c['contribution'] for c in contributions if c['contribution'] > 0)
        total_negative = sum(c['contribution'] for c in contributions if c['contribution'] < 0)
        total_absolute = sum(c['absolute_contribution'] for c in contributions)
        
        explanation = {
            'sample_index': sample_index,
            'prediction': float(self.model.predict(sample.reshape(1, -1))[0]),
            'expected_value': float(sample_shap.base_values[0]),
            'total_contribution': float(total_positive + total_negative),
            'total_absolute_contribution': float(total_absolute),
            'positive_contributions': float(total_positive),
            'negative_contributions': float(total_negative),
            'top_contributors': contributions[:10],
            'contributions_by_type': contributions_by_type,
            'feature_contributions': {
                'vibration': sum(c['contribution'] for c in contributions 
                               if c['feature_type'] == 'vibration'),
                'thermal': sum(c['contribution'] for c in contributions 
                             if c['feature_type'] == 'thermal'),
                'acoustic': sum(c['contribution'] for c in contributions 
                              if c['feature_type'] == 'acoustic'),
                'pressure': sum(c['contribution'] for c in contributions 
                              if c['feature_type'] == 'pressure'),
                'aircraft': sum(c['contribution'] for c in contributions 
                              if c['feature_type'] == 'aircraft')
            }
        }
        
        return explanation
    
    def _get_feature_type(self, feature_index: int) -> str:
        """Get the type of feature based on index"""
        
        if feature_index < 1000:
            return 'vibration'
        elif feature_index < 1500:
            return 'thermal'
        elif feature_index < 1750:
            return 'acoustic'
        elif feature_index < 1850:
            return 'pressure'
        else:
            return 'aircraft'
    
    def generate_global_explanations(self, X: np.ndarray) -> Dict:
        """Generate global explanations for the model"""
        
        if self.shap_values is None:
            self.fit_explainer(X)
        
        explanations = {
            'global_feature_importance': {},
            'feature_interactions': {},
            'model_summary': {}
        }
        
        # Global feature importance
        feature_importance = self.get_feature_importance()
        explanations['global_feature_importance'] = feature_importance.to_dict('records')
        
        # Feature interactions (simplified)
        if len(X.shape) == 2 and X.shape[1] > 1:
            try:
                # Calculate interaction for top features
                top_features = feature_importance.head(10)['feature'].tolist()
                top_indices = [self.feature_names.index(f) for f in top_features]
                
                interactions = {}
                for i, idx1 in enumerate(top_indices):
                    for idx2 in top_indices[i+1:]:
                        # Simplified interaction calculation
                        corr = np.corrcoef(X[:, idx1], X[:, idx2])[0, 1]
                        if abs(corr) > 0.3:  # Only significant correlations
                            interactions[f"{top_features[i]}-{top_features[idx2]}"] = {
                                'correlation': float(corr),
                                'feature1': top_features[i],
                                'feature2': self.feature_names[idx2]
                            }
                
                explanations['feature_interactions'] = interactions
            except:
                pass
        
        # Model summary
        explanations['model_summary'] = {
            'total_features': len(self.feature_names),
            'samples_used': len(self.shap_values) if self.shap_values else 0,
            'feature_types': {
                'vibration': 1000,
                'thermal': 500,
                'acoustic': 250,
                'pressure': 100,
                'aircraft': 3
            }
        }
        
        return explanations
    
    def generate_fault_explanations(self, 
                                   X_fault: np.ndarray, 
                                   X_normal: np.ndarray,
                                   fault_labels: List[str]) -> Dict:
        """Generate explanations specifically for fault cases"""
        
        explanations = {}
        
        for i, (sample, fault_label) in enumerate(zip(X_fault, fault_labels)):
            # Get sample explanation
            sample_explanation = self.get_sample_explanation(sample, i)
            
            # Compare with normal samples
            normal_contributions = []
            for normal_sample in X_normal[:10]:  # Compare with 10 normal samples
                normal_exp = self.get_sample_explanation(normal_sample)
                normal_contributions.append(normal_exp)
            
            # Calculate differences
            avg_normal_contrib = {
                'vibration': np.mean([e['feature_contributions']['vibration'] 
                                     for e in normal_contributions]),
                'thermal': np.mean([e['feature_contributions']['thermal'] 
                                   for e in normal_contributions]),
                'acoustic': np.mean([e['feature_contributions']['acoustic'] 
                                    for e in normal_contributions]),
                'pressure': np.mean([e['feature_contributions']['pressure'] 
                                    for e in normal_contributions])
            }
            
            fault_contrib = sample_explanation['feature_contributions']
            
            differences = {
                'vibration': fault_contrib['vibration'] - avg_normal_contrib['vibration'],
                'thermal': fault_contrib['thermal'] - avg_normal_contrib['thermal'],
                'acoustic': fault_contrib['acoustic'] - avg_normal_contrib['acoustic'],
                'pressure': fault_contrib['pressure'] - avg_normal_contrib['pressure']
            }
            
            # Identify primary fault indicator
            primary_indicator = max(differences.items(), key=lambda x: abs(x[1]))
            
            explanations[fault_label] = {
                'sample_explanation': sample_explanation,
                'differences_from_normal': differences,
                'primary_fault_indicator': {
                    'sensor_type': primary_indicator[0],
                    'contribution_difference': primary_indicator[1]
                },
                'top_contributing_features': sample_explanation['top_contributors'][:5]
            }
        
        return explanations
    
    def save_explanations(self, explanations: Dict, filepath: str):
        """Save explanations to file"""
        
        with open(filepath, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        print(f"✅ Explanations saved to {filepath}")
    
    def load_explanations(self, filepath: str) -> Dict:
        """Load explanations from file"""
        
        with open(filepath, 'r') as f:
            explanations = json.load(f)
        
        print(f"✅ Explanations loaded from {filepath}")
        return explanations


# Test function
def test_shap_explainer():
    """Test SHAP explainer"""
    
    print("Testing SHAP Explainer...")
    
    # Create dummy model and data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=1853,  # Matching our feature count
        n_informative=100,
        n_redundant=500,
        random_state=42
    )
    
    # Train a simple model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create SHAP explainer
    explainer = SHAPExplainer(model)
    
    # Fit explainer
    explainer.fit_explainer(X, sample_size=100)
    
    # Test feature importance
    print("\nTesting feature importance...")
    feature_importance = explainer.get_feature_importance()
    print(f"Top 5 features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Test sample explanation
    print("\nTesting sample explanation...")
    sample_idx = 0
    sample_explanation = explainer.get_sample_explanation(X[sample_idx], sample_idx)
    
    print(f"Sample {sample_idx} Explanation:")
    print(f"  Prediction: {sample_explanation['prediction']:.4f}")
    print(f"  Total Contribution: {sample_explanation['total_contribution']:.4f}")
    print(f"  Feature Contributions:")
    for sensor, contrib in sample_explanation['feature_contributions'].items():
        print(f"    {sensor}: {contrib:.4f}")
    
    print("\nTop Contributing Features:")
    for contrib in sample_explanation['top_contributors'][:3]:
        print(f"  {contrib['feature_name']}: {contrib['contribution']:.4f}")
    
    # Test global explanations
    print("\nTesting global explanations...")
    global_explanations = explainer.generate_global_explanations(X)
    
    print(f"Global Feature Importance: {len(global_explanations['global_feature_importance'])} features")
    print(f"Feature Interactions: {len(global_explanations['feature_interactions'])} interactions")
    
    # Test fault explanations
    print("\nTesting fault explanations...")
    
    # Separate fault and normal samples
    fault_indices = np.where(y == 1)[0][:5]
    normal_indices = np.where(y == 0)[0][:10]
    
    X_fault = X[fault_indices]
    X_normal = X[normal_indices]
    fault_labels = [f"fault_{i}" for i in range(len(fault_indices))]
    
    fault_explanations = explainer.generate_fault_explanations(
        X_fault, X_normal, fault_labels
    )
    
    print(f"Generated explanations for {len(fault_explanations)} fault cases")
    
    for fault_label, explanation in fault_explanations.items():
        primary = explanation['primary_fault_indicator']
        print(f"  {fault_label}: Primary indicator = {primary['sensor_type']} "
              f"(difference: {primary['contribution_difference']:.4f})")
    
    # Save explanations
    explainer.save_explanations(
        {'test_explanations': fault_explanations},
        'shap_test_explanations.json'
    )
    
    print("\n✅ SHAP explainer test completed!")
    return explainer


if __name__ == "__main__":
    test_shap_explainer()