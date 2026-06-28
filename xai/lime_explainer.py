"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer
Provides local explanations for individual predictions
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

class LIMEExplainer:
    """LIME-based explainer for local model interpretability"""
    
    def __init__(self, 
                 model,
                 feature_names: List[str] = None,
                 class_names: List[str] = None,
                 mode: str = 'classification'):
        
        self.model = model
        self.feature_names = feature_names or self._generate_feature_names()
        self.class_names = class_names or self._generate_class_names()
        self.mode = mode
        self.explainer = None
        self.categorical_features = None
        
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for multi-modal data"""
        
        feature_names = []
        
        # Group features by sensor type
        # Vibration features
        vib_features = []
        for i in range(1000):
            if i < 100:
                vib_features.append(f'vib_low_{i}')
            elif i < 500:
                vib_features.append(f'vib_mid_{i}')
            else:
                vib_features.append(f'vib_high_{i}')
        feature_names.extend(vib_features)
        
        # Thermal features
        thermal_features = []
        for i in range(500):
            if i < 100:
                thermal_features.append(f'thermal_low_{i}')
            elif i < 300:
                thermal_features.append(f'thermal_mid_{i}')
            else:
                thermal_features.append(f'thermal_high_{i}')
        feature_names.extend(thermal_features)
        
        # Acoustic features
        acoustic_features = []
        for i in range(250):
            if i < 50:
                acoustic_features.append(f'acoustic_low_{i}')
            elif i < 150:
                acoustic_features.append(f'acoustic_mid_{i}')
            else:
                acoustic_features.append(f'acoustic_high_{i}')
        feature_names.extend(acoustic_features)
        
        # Pressure features
        pressure_features = []
        for i in range(100):
            pressure_features.append(f'pressure_{i}')
        feature_names.extend(pressure_features)
        
        # Aircraft features (categorical)
        feature_names.extend(['aircraft_type_boeing', 
                             'aircraft_type_cessna', 
                             'aircraft_type_airbus'])
        
        return feature_names
    
    def _generate_class_names(self) -> List[str]:
        """Generate class names for fault classification"""
        
        return [
            'no_fault',
            'bearing_wear',
            'blade_damage', 
            'oil_leak',
            'compressor_stall',
            'ignition_failure',
            'crack_detection',
            'corrosion',
            'fatigue',
            'loose_fasteners',
            'hydraulic_leak',
            'electrical_short',
            'avionics_failure',
            'fuel_system_issue',
            'actuator_jam',
            'sensor_failure',
            'runaway_trim'
        ]
    
    def _get_categorical_features(self) -> List[int]:
        """Identify categorical features"""
        
        categorical = []
        for i, name in enumerate(self.feature_names):
            if 'aircraft_type' in name:
                categorical.append(i)
        
        return categorical
    
    def fit_explainer(self, X: np.ndarray):
        """Fit LIME explainer to data"""
        
        print("Fitting LIME explainer...")
        
        # Identify categorical features
        self.categorical_features = self._get_categorical_features()
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X,
            feature_names=self.feature_names,
            class_names=self.class_names,
            categorical_features=self.categorical_features,
            mode=self.mode,
            discretize_continuous=True,
            random_state=42
        )
        
        print(f"✅ LIME explainer fitted with {len(X)} samples")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Categorical features: {len(self.categorical_features)}")
        
        return self
    
    def explain_instance(self, 
                        instance: np.ndarray, 
                        num_features: int = 10,
                        num_samples: int = 5000) -> Dict:
        """Explain a single instance prediction"""
        
        if self.explainer is None:
            raise ValueError("LIME explainer not fitted. Call fit_explainer first.")
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        # Generate explanation
        if self.mode == 'classification':
            exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=3  # Explain top 3 predicted classes
            )
            
            # Get predicted class
            predicted_probs = predict_fn(instance.reshape(1, -1))[0]
            predicted_class = np.argmax(predicted_probs)
            predicted_class_name = self.class_names[predicted_class]
            
            # Get explanation for predicted class
            explanation_data = []
            
            # For predicted class
            local_exp = exp.local_exp[predicted_class]
            for feature_idx, weight in local_exp:
                feature_name = self.feature_names[feature_idx]
                feature_type = self._get_feature_type(feature_idx)
                feature_value = instance[feature_idx]
                
                explanation_data.append({
                    'feature_index': int(feature_idx),
                    'feature_name': feature_name,
                    'feature_type': feature_type,
                    'weight': float(weight),
                    'absolute_weight': abs(float(weight)),
                    'feature_value': float(feature_value),
                    'contribution': 'positive' if weight > 0 else 'negative'
                })
            
            # Sort by absolute weight
            explanation_data.sort(key=lambda x: x['absolute_weight'], reverse=True)
            
            # Group by feature type
            contributions_by_type = {}
            for item in explanation_data:
                feature_type = item['feature_type']
                if feature_type not in contributions_by_type:
                    contributions_by_type[feature_type] = []
                contributions_by_type[feature_type].append(item)
            
            # Calculate summary
            total_positive = sum(item['weight'] for item in explanation_data 
                               if item['weight'] > 0)
            total_negative = sum(item['weight'] for item in explanation_data 
                               if item['weight'] < 0)
            
            explanation = {
                'instance_index': None,
                'predicted_class': predicted_class,
                'predicted_class_name': predicted_class_name,
                'prediction_probability': float(predicted_probs[predicted_class]),
                'top_features': explanation_data[:num_features],
                'feature_contributions': {
                    feature_type: sum(item['weight'] for item in items)
                    for feature_type, items in contributions_by_type.items()
                },
                'contribution_summary': {
                    'total_positive': float(total_positive),
                    'total_negative': float(total_negative),
                    'net_contribution': float(total_positive + total_negative)
                },
                'alternative_classes': []
            }
            
            # Add alternative class explanations
            top_labels = exp.available_labels()[:3]
            for label in top_labels:
                if label != predicted_class:
                    label_name = self.class_names[label]
                    label_prob = predicted_probs[label]
                    
                    # Get explanation for this class
                    label_exp = exp.local_exp[label]
                    label_features = []
                    
                    for feature_idx, weight in label_exp[:5]:  # Top 5 features
                        feature_name = self.feature_names[feature_idx]
                        label_features.append({
                            'feature_name': feature_name,
                            'weight': float(weight)
                        })
                    
                    explanation['alternative_classes'].append({
                        'class': label,
                        'class_name': label_name,
                        'probability': float(label_prob),
                        'top_features': label_features
                    })
            
            return explanation
            
        else:  # Regression mode
            exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Similar processing for regression
            # (Implementation omitted for brevity)
            return {}
    
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
    
    def explain_multiple_instances(self, 
                                  instances: np.ndarray,
                                  instance_indices: List[int] = None,
                                  num_features: int = 10) -> Dict:
        """Explain multiple instances"""
        
        if instance_indices is None:
            instance_indices = list(range(len(instances)))
        
        explanations = {}
        
        for idx, instance in zip(instance_indices, instances):
            explanation = self.explain_instance(instance, num_features)
            explanation['instance_index'] = idx
            explanations[f'instance_{idx}'] = explanation
        
        return explanations
    
    def generate_comparative_explanations(self,
                                         instances_a: np.ndarray,
                                         instances_b: np.ndarray,
                                         label_a: str = 'fault',
                                         label_b: str = 'normal',
                                         num_features: int = 10) -> Dict:
        """Generate comparative explanations between two sets of instances"""
        
        # Explain instances from group A
        explanations_a = self.explain_multiple_instances(
            instances_a[:5],  # Use first 5 instances
            num_features=num_features
        )
        
        # Explain instances from group B
        explanations_b = self.explain_multiple_instances(
            instances_b[:5],  # Use first 5 instances
            num_features=num_features
        )
        
        # Calculate average contributions by feature type
        def get_average_contributions(explanations_dict):
            contributions = {
                'vibration': [],
                'thermal': [],
                'acoustic': [],
                'pressure': [],
                'aircraft': []
            }
            
            for exp in explanations_dict.values():
                for feature_type, contrib in exp['feature_contributions'].items():
                    contributions[feature_type].append(contrib)
            
            return {
                feature_type: np.mean(values) if values else 0
                for feature_type, values in contributions.items()
            }
        
        avg_a = get_average_contributions(explanations_a)
        avg_b = get_average_contributions(explanations_b)
        
        # Calculate differences
        differences = {}
        for feature_type in avg_a.keys():
            differences[feature_type] = avg_a[feature_type] - avg_b[feature_type]
        
        # Identify key differentiators
        sorted_differences = sorted(
            differences.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        comparative_explanation = {
            'group_a': {
                'label': label_a,
                'count': len(instances_a),
                'average_contributions': avg_a,
                'sample_explanations': explanations_a
            },
            'group_b': {
                'label': label_b,
                'count': len(instances_b),
                'average_contributions': avg_b,
                'sample_explanations': explanations_b
            },
            'differences': differences,
            'key_differentiators': [
                {
                    'feature_type': feature_type,
                    'difference': diff,
                    'interpretation': self._interpret_difference(feature_type, diff)
                }
                for feature_type, diff in sorted_differences[:3]  # Top 3
            ],
            'summary': {
                'primary_differentiator': sorted_differences[0][0] if sorted_differences else None,
                'largest_difference': sorted_differences[0][1] if sorted_differences else 0
            }
        }
        
        return comparative_explanation
    
    def _interpret_difference(self, feature_type: str, difference: float) -> str:
        """Interpret the difference in contributions"""
        
        interpretations = {
            'vibration': {
                'positive': "Higher vibration contributes more to fault classification",
                'negative': "Lower vibration contributes more to fault classification",
                'neutral': "Vibration contribution is similar"
            },
            'thermal': {
                'positive': "Higher temperature contributes more to fault classification",
                'negative': "Lower temperature contributes more to fault classification",
                'neutral': "Temperature contribution is similar"
            },
            'acoustic': {
                'positive': "Higher acoustic levels contribute more to fault classification",
                'negative': "Lower acoustic levels contribute more to fault classification",
                'neutral': "Acoustic contribution is similar"
            },
            'pressure': {
                'positive': "Higher pressure contributes more to fault classification",
                'negative': "Lower pressure contributes more to fault classification",
                'neutral': "Pressure contribution is similar"
            },
            'aircraft': {
                'positive': "Aircraft type contributes more to fault classification",
                'negative': "Aircraft type contributes less to fault classification",
                'neutral': "Aircraft type contribution is similar"
            }
        }
        
        if feature_type not in interpretations:
            return "Unknown feature type"
        
        if difference > 0.1:
            return interpretations[feature_type]['positive']
        elif difference < -0.1:
            return interpretations[feature_type]['negative']
        else:
            return interpretations[feature_type]['neutral']
    
    def generate_fault_pattern_explanations(self,
                                          fault_instances: np.ndarray,
                                          fault_labels: List[str],
                                          normal_instances: np.ndarray) -> Dict:
        """Generate explanations highlighting fault patterns"""
        
        explanations = {}
        
        for i, (instance, fault_label) in enumerate(zip(fault_instances, fault_labels)):
            # Explain fault instance
            fault_exp = self.explain_instance(instance, num_features=15)
            
            # Find similar normal instances
            similar_normals = self._find_similar_instances(
                instance, normal_instances, n=3
            )
            
            # Explain similar normal instances
            normal_exps = self.explain_multiple_instances(similar_normals)
            
            # Compare with normals
            comparisons = []
            for normal_idx, normal_exp in normal_exps.items():
                comparison = {
                    'normal_instance': normal_idx,
                    'feature_differences': []
                }
                
                # Compare top features
                fault_features = {item['feature_name']: item['weight'] 
                                for item in fault_exp['top_features'][:5]}
                
                # Find corresponding features in normal explanation
                for normal_item in normal_exp['top_features'][:5]:
                    feature_name = normal_item['feature_name']
                    if feature_name in fault_features:
                        diff = fault_features[feature_name] - normal_item['weight']
                        comparison['feature_differences'].append({
                            'feature': feature_name,
                            'fault_weight': fault_features[feature_name],
                            'normal_weight': normal_item['weight'],
                            'difference': diff
                        })
                
                comparisons.append(comparison)
            
            # Identify unique fault indicators
            fault_indicators = []
            for item in fault_exp['top_features']:
                if abs(item['weight']) > 0.2:  # Significant contribution
                    fault_indicators.append({
                        'feature': item['feature_name'],
                        'weight': item['weight'],
                        'feature_type': item['feature_type'],
                        'interpretation': self._interpret_feature_contribution(
                            item['feature_type'], item['weight']
                        )
                    })
            
            explanations[fault_label] = {
                'fault_explanation': fault_exp,
                'similar_normal_explanations': normal_exps,
                'comparisons_with_normals': comparisons,
                'fault_indicators': fault_indicators,
                'primary_fault_sensor': self._identify_primary_sensor(fault_exp),
                'recommended_investigation': self._generate_investigation_recommendation(
                    fault_exp, fault_indicators
                )
            }
        
        return explanations
    
    def _find_similar_instances(self, 
                               query: np.ndarray, 
                               instances: np.ndarray, 
                               n: int = 3) -> np.ndarray:
        """Find n most similar instances using Euclidean distance"""
        
        distances = np.linalg.norm(instances - query, axis=1)
        closest_indices = np.argsort(distances)[:n]
        
        return instances[closest_indices]
    
    def _interpret_feature_contribution(self, feature_type: str, weight: float) -> str:
        """Interpret feature contribution"""
        
        interpretations = {
            'vibration': {
                'positive': "High vibration level indicates potential mechanical fault",
                'negative': "Unusually low vibration may indicate sensor fault or damping issue",
                'neutral': "Vibration within normal range"
            },
            'thermal': {
                'positive': "Elevated temperature indicates potential overheating issue",
                'negative': "Unusually low temperature may indicate cooling system fault",
                'neutral': "Temperature within normal range"
            },
            'acoustic': {
                'positive': "Abnormal acoustic signature indicates potential component wear",
                'negative': "Unusually quiet operation may indicate sensor fault",
                'neutral': "Acoustic levels within normal range"
            },
            'pressure': {
                'positive': "High pressure indicates potential system overpressure",
                'negative': "Low pressure indicates potential leak or pump failure",
                'neutral': "Pressure within normal range"
            }
        }
        
        if feature_type not in interpretations:
            return "Contribution from non-sensor feature"
        
        if weight > 0.1:
            return interpretations[feature_type]['positive']
        elif weight < -0.1:
            return interpretations[feature_type]['negative']
        else:
            return interpretations[feature_type]['neutral']
    
    def _identify_primary_sensor(self, explanation: Dict) -> Dict:
        """Identify primary sensor contributing to fault"""
        
        sensor_contributions = explanation['feature_contributions']
        
        # Find sensor with highest absolute contribution
        primary_sensor = max(sensor_contributions.items(), 
                           key=lambda x: abs(x[1]))
        
        return {
            'sensor_type': primary_sensor[0],
            'contribution': primary_sensor[1],
            'confidence': 'high' if abs(primary_sensor[1]) > 0.3 else 'medium'
        }
    
    def _generate_investigation_recommendation(self, 
                                             explanation: Dict,
                                             fault_indicators: List) -> Dict:
        """Generate investigation recommendations based on explanation"""
        
        primary_sensor = self._identify_primary_sensor(explanation)
        
        recommendations = {
            'immediate_actions': [],
            'further_investigation': [],
            'maintenance_suggestions': []
        }
        
        # Based on primary sensor
        if primary_sensor['sensor_type'] == 'vibration':
            recommendations['immediate_actions'].append(
                "Check mechanical components for wear or imbalance"
            )
            recommendations['further_investigation'].append(
                "Perform frequency analysis to identify specific fault frequencies"
            )
            recommendations['maintenance_suggestions'].append(
                "Schedule vibration analysis and bearing inspection"
            )
        
        elif primary_sensor['sensor_type'] == 'thermal':
            recommendations['immediate_actions'].append(
                "Check cooling systems and thermal management"
            )
            recommendations['further_investigation'].append(
                "Perform thermal imaging to identify hot spots"
            )
            recommendations['maintenance_suggestions'].append(
                "Clean heat exchangers and check fluid levels"
            )
        
        elif primary_sensor['sensor_type'] == 'acoustic':
            recommendations['immediate_actions'].append(
                "Listen for unusual noises in the identified frequency ranges"
            )
            recommendations['further_investigation'].append(
                "Perform acoustic emission testing"
            )
            recommendations['maintenance_suggestions'].append(
                "Inspect for loose components or structural issues"
            )
        
        elif primary_sensor['sensor_type'] == 'pressure':
            recommendations['immediate_actions'].append(
                "Check for pressure leaks in the identified system"
            )
            recommendations['further_investigation'].append(
                "Perform pressure decay test"
            )
            recommendations['maintenance_suggestions'].append(
                "Inspect seals, valves, and pressure regulators"
            )
        
        # Add general recommendations
        recommendations['immediate_actions'].append(
            "Review recent maintenance records for related components"
        )
        recommendations['further_investigation'].append(
            "Correlate with other sensor data for comprehensive diagnosis"
        )
        
        return recommendations
    
    def save_explanations(self, explanations: Dict, filepath: str):
        """Save explanations to file"""
        
        with open(filepath, 'w') as f:
            json.dump(explanations, f, indent=2)
        
        print(f"✅ LIME explanations saved to {filepath}")


# Test function
def test_lime_explainer():
    """Test LIME explainer"""
    
    print("Testing LIME Explainer...")
    
    # Create dummy model and data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=1853,
        n_informative=50,
        n_redundant=300,
        n_classes=5,
        random_state=42
    )
    
    # Train a simple model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create LIME explainer
    explainer = LIMEExplainer(model, mode='classification')
    
    # Fit explainer
    explainer.fit_explainer(X)
    
    # Test single instance explanation
    print("\nTesting single instance explanation...")
    sample_idx = 0
    sample = X[sample_idx]
    
    explanation = explainer.explain_instance(sample, num_features=8)
    
    print(f"Predicted Class: {explanation['predicted_class_name']}")
    print(f"Probability: {explanation['prediction_probability']:.4f}")
    print(f"Net Contribution: {explanation['contribution_summary']['net_contribution']:.4f}")
    
    print("\nTop Contributing Features:")
    for i, feature in enumerate(explanation['top_features'][:5]):
        print(f"  {feature['feature_name']}: {feature['weight']:.4f} "
              f"({feature['feature_type']})")
    
    print("\nFeature Contributions by Sensor Type:")
    for sensor, contrib in explanation['feature_contributions'].items():
        print(f"  {sensor}: {contrib:.4f}")
    
    # Test comparative explanations
    print("\nTesting comparative explanations...")
    
    # Split into fault and normal (simulated)
    fault_indices = np.where(y != 0)[0][:10]
    normal_indices = np.where(y == 0)[0][:10]
    
    X_fault = X[fault_indices]
    X_normal = X[normal_indices]
    
    comparative = explainer.generate_comparative_explanations(
        X_fault, X_normal, 'fault', 'normal'
    )
    
    print(f"\nComparative Analysis:")
    print(f"Primary Differentiator: {comparative['summary']['primary_differentiator']}")
    print(f"Largest Difference: {comparative['summary']['largest_difference']:.4f}")
    
    print("\nKey Differentiators:")
    for diff in comparative['key_differentiators']:
        print(f"  {diff['feature_type']}: {diff['difference']:.4f}")
        print(f"    {diff['interpretation']}")
    
    # Test fault pattern explanations
    print("\nTesting fault pattern explanations...")
    
    fault_labels = [f"test_fault_{i}" for i in range(3)]
    fault_patterns = explainer.generate_fault_pattern_explanations(
        X_fault[:3], fault_labels, X_normal
    )
    
    print(f"Generated explanations for {len(fault_patterns)} fault patterns")
    
    for fault_label, pattern in fault_patterns.items():
        primary = pattern['primary_fault_sensor']
        print(f"\n{fault_label}:")
        print(f"  Primary Sensor: {primary['sensor_type']} "
              f"(contribution: {primary['contribution']:.4f})")
        print(f"  Recommended Investigation:")
        for action in pattern['recommended_investigation']['immediate_actions'][:2]:
            print(f"    - {action}")
    
    # Save explanations
    explainer.save_explanations(
        {'test_explanations': explanation},
        'lime_test_explanations.json'
    )
    
    print("\n✅ LIME explainer test completed!")
    return explainer


if __name__ == "__main__":
    test_lime_explainer()