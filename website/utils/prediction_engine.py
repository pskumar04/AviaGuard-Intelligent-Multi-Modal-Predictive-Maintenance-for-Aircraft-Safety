"""
Prediction Engine for Aircraft Predictive Maintenance
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
import sys
import os

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

class PredictionEngine:
    """Main prediction engine for flight safety assessment"""
    
    def __init__(self):
        self.thresholds = self.load_thresholds()
        self.rules = self.load_rules()
        
    def load_thresholds(self) -> Dict:
        """Load threshold values for different parameters"""
        return {
            'boeing_737': {
                'engine': {
                    'n1_rpm': {'min': 95, 'max': 101, 'critical': 10},
                    'n2_rpm': {'min': 98, 'max': 102, 'critical': 10},
                    'egt': {'min': 400, 'max': 650, 'critical': 100},
                    'oil_temperature': {'min': 60, 'max': 110, 'critical': 30},
                    'oil_pressure': {'min': 40, 'max': 80, 'critical': 20},
                    'vibration_level': {'min': 0, 'max': 0.5, 'critical': 0.5}
                },
                'structural': {
                    'wing_vibration': {'min': 0, 'max': 0.3, 'critical': 0.3},
                    'fuselage_strain': {'min': 0, 'max': 500, 'critical': 300}
                }
            },
            'cessna_172': {
                'engine': {
                    'rpm': {'min': 2000, 'max': 2700, 'critical': 500},
                    'oil_temperature': {'min': 100, 'max': 245, 'critical': 50},
                    'oil_pressure': {'min': 20, 'max': 60, 'critical': 15}
                }
            },
            'airbus_h125': {
                'rotor_system': {
                    'rotor_rpm': {'min': 300, 'max': 400, 'critical': 50},
                    'vibration_level': {'min': 0, 'max': 0.5, 'critical': 0.3}
                }
            }
        }
    
    def load_rules(self) -> List[Dict]:
        """Load prediction rules"""
        return [
            {
                'condition': 'critical_failure',
                'rules': [
                    lambda data: any(v > t['max'] + t['critical'] or v < t['min'] - t['critical'] 
                                     for k, v in data.items() 
                                     if k in self.thresholds.get('boeing_737', {}).get('engine', {})),
                    lambda data: data.get('oil_pressure', 0) < 20,
                    lambda data: data.get('vibration_level', 0) > 1.0
                ]
            },
            {
                'condition': 'warning',
                'rules': [
                    lambda data: any(v > t['max'] or v < t['min'] 
                                     for k, v in data.items() 
                                     if k in self.thresholds.get('boeing_737', {}).get('engine', {})),
                    lambda data: 0.5 < data.get('vibration_level', 0) <= 1.0
                ]
            }
        ]
    
    def predict_flight_safety(self, survey_data: Dict, model_type: str) -> Tuple[Dict, Dict]:
        """Predict flight safety based on survey data"""
        try:
            # Extract relevant data based on model type
            processed_data = self.process_survey_data(survey_data, model_type)
            
            # Apply rules-based prediction
            rule_based_result = self.apply_rules(processed_data, model_type)
            
            # Calculate confidence score
            confidence = self.calculate_confidence(processed_data, model_type)
            
            # Determine flight condition
            condition = self.determine_condition(rule_based_result, confidence)
            
            # Generate fault detections
            fault_detections = self.detect_faults(processed_data, model_type)
            
            # Prepare prediction result
            prediction = {
                'model_type': model_type,
                'timestamp': datetime.utcnow().isoformat(),
                'flight_condition': condition,
                'confidence_score': confidence,
                'rule_based_result': rule_based_result,
                'recommendation': self.generate_recommendation(condition, fault_detections)
            }
            
            logger.info(f"Prediction generated: {condition} with confidence {confidence:.2f}")
            return prediction, fault_detections
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self.get_fallback_prediction(model_type)
    
    def process_survey_data(self, survey_data: Dict, model_type: str) -> Dict:
        """Process survey data for prediction"""
        processed = {}
        
        if model_type == 'boeing_737':
            # Process Boeing data
            engine_data = survey_data.get('engine', {})
            structural_data = survey_data.get('structural', {})
            
            processed.update({
                'n1_rpm': engine_data.get('n1_rpm', 0),
                'n2_rpm': engine_data.get('n2_rpm', 0),
                'egt': engine_data.get('egt', 0),
                'oil_temperature': engine_data.get('oil_temperature', 0),
                'oil_pressure': engine_data.get('oil_pressure', 0),
                'vibration_level': engine_data.get('vibration_level', 0),
                'wing_vibration': structural_data.get('wing_vibration', 0),
                'fuselage_strain': structural_data.get('fuselage_strain', 0)
            })
            
        elif model_type == 'cessna_172':
            # Process Cessna data
            engine_data = survey_data.get('engine', {})
            
            processed.update({
                'rpm': engine_data.get('rpm', 0),
                'manifold_pressure': engine_data.get('manifold_pressure', 0),
                'oil_temperature': engine_data.get('oil_temperature', 0),
                'oil_pressure': engine_data.get('oil_pressure', 0),
                'fuel_pressure': engine_data.get('fuel_pressure', 0),
                'cylinder_head_temp': engine_data.get('cylinder_head_temp', 0)
            })
            
        elif model_type == 'airbus_h125':
            # Process Airbus data
            rotor_data = survey_data.get('rotor_system', {})
            engine_data = survey_data.get('engine', {})
            
            processed.update({
                'rotor_rpm': rotor_data.get('rotor_rpm', 0),
                'vibration_level': rotor_data.get('vibration_level', 0),
                'track_and_balance': rotor_data.get('track_and_balance', 0),
                'torque': engine_data.get('torque', 0),
                'ng': engine_data.get('ng', 0),
                't4': engine_data.get('t4', 0)
            })
        
        return processed
    
    def apply_rules(self, data: Dict, model_type: str) -> Dict:
        """Apply prediction rules to data"""
        result = {
            'critical_failures': 0,
            'warnings': 0,
            'normal_parameters': 0,
            'failed_rules': []
        }
        
        model_thresholds = self.thresholds.get(model_type, {})
        
        for category, params in model_thresholds.items():
            for param, thresholds in params.items():
                value = data.get(param, 0)
                
                if param in data:
                    if value < thresholds['min'] - thresholds['critical'] or value > thresholds['max'] + thresholds['critical']:
                        result['critical_failures'] += 1
                        result['failed_rules'].append({
                            'parameter': param,
                            'value': value,
                            'threshold': thresholds,
                            'severity': 'critical'
                        })
                    elif value < thresholds['min'] or value > thresholds['max']:
                        result['warnings'] += 1
                        result['failed_rules'].append({
                            'parameter': param,
                            'value': value,
                            'threshold': thresholds,
                            'severity': 'warning'
                        })
                    else:
                        result['normal_parameters'] += 1
        
        return result
    
    def calculate_confidence(self, data: Dict, model_type: str) -> float:
        """Calculate prediction confidence score"""
        model_thresholds = self.thresholds.get(model_type, {})
        total_params = 0
        normal_params = 0
        
        for category, params in model_thresholds.items():
            for param, thresholds in params.items():
                if param in data:
                    total_params += 1
                    value = data[param]
                    
                    # Check if within normal range
                    if thresholds['min'] <= value <= thresholds['max']:
                        normal_params += 1
                    # Check if within acceptable range (including buffer)
                    elif thresholds['min'] - 0.1 * (thresholds['max'] - thresholds['min']) <= value <= thresholds['max'] + 0.1 * (thresholds['max'] - thresholds['min']):
                        normal_params += 0.5
        
        if total_params == 0:
            return 0.5  # Default confidence
        
        base_confidence = normal_params / total_params
        
        # Adjust confidence based on critical parameters
        critical_params = ['oil_pressure', 'vibration_level', 'rotor_rpm']
        critical_score = 1.0
        
        for param in critical_params:
            if param in data:
                thresholds = next((p.get(param) for p in model_thresholds.values() if param in p), None)
                if thresholds:
                    value = data[param]
                    if thresholds['min'] <= value <= thresholds['max']:
                        critical_score *= 1.0
                    else:
                        critical_score *= 0.7
        
        final_confidence = base_confidence * critical_score
        return min(max(final_confidence, 0.0), 1.0)
    
    def determine_condition(self, rule_result: Dict, confidence: float) -> str:
        """Determine flight condition"""
        critical_failures = rule_result['critical_failures']
        warnings = rule_result['warnings']
        
        if critical_failures > 0 or confidence < 0.6:
            return 'critical'
        elif warnings > 0 or confidence < 0.8:
            return 'warning'
        else:
            return 'normal'
    
    def detect_faults(self, data: Dict, model_type: str) -> Dict:
        """Detect specific faults in the data"""
        faults = []
        model_thresholds = self.thresholds.get(model_type, {})
        
        # Common fault patterns
        fault_patterns = [
            {
                'name': 'high_vibration',
                'condition': lambda d: d.get('vibration_level', 0) > 0.7,
                'severity': 'high',
                'description': 'Excessive vibration detected',
                'action': 'Inspect vibration dampers and balance components'
            },
            {
                'name': 'low_oil_pressure',
                'condition': lambda d: d.get('oil_pressure', 0) < 30,
                'severity': 'critical',
                'description': 'Oil pressure below minimum threshold',
                'action': 'Check oil pump, filter, and lines immediately'
            },
            {
                'name': 'high_temperature',
                'condition': lambda d: d.get('oil_temperature', 0) > 120 or d.get('egt', 0) > 700,
                'severity': 'high',
                'description': 'Operating temperature too high',
                'action': 'Check cooling system and reduce load'
            }
        ]
        
        # Check each fault pattern
        for pattern in fault_patterns:
            if pattern['condition'](data):
                faults.append({
                    'fault_type': pattern['name'],
                    'description': pattern['description'],
                    'severity': pattern['severity'],
                    'recommended_action': pattern['action'],
                    'confidence': 0.85,
                    'is_critical': pattern['severity'] == 'critical'
                })
        
        # Check threshold violations
        for category, params in model_thresholds.items():
            for param, thresholds in params.items():
                if param in data:
                    value = data[param]
                    
                    if value < thresholds['min'] or value > thresholds['max']:
                        severity = 'critical' if (value < thresholds['min'] - thresholds['critical'] or 
                                                 value > thresholds['max'] + thresholds['critical']) else 'warning'
                        
                        faults.append({
                            'fault_type': f'threshold_violation_{param}',
                            'description': f'{param} outside normal range ({thresholds["min"]}-{thresholds["max"]})',
                            'severity': severity,
                            'recommended_action': f'Adjust {param} to normal range',
                            'confidence': 0.9,
                            'is_critical': severity == 'critical'
                        })
        
        return {
            'predictions': faults,
            'total_faults': len(faults),
            'critical_faults': len([f for f in faults if f['is_critical']]),
            'warning_faults': len([f for f in faults if not f['is_critical']]),
            'average_confidence': np.mean([f['confidence'] for f in faults]) if faults else 0
        }
    
    def generate_recommendation(self, condition: str, fault_detections: Dict) -> str:
        """Generate recommendation based on condition and faults"""
        if condition == 'critical':
            return "GROUND AIRCRAFT IMMEDIATELY. Do not attempt to fly until all critical issues are resolved."
        elif condition == 'warning':
            critical_faults = fault_detections.get('critical_faults', 0)
            if critical_faults > 0:
                return f"Schedule emergency maintenance. {critical_faults} critical faults detected."
            else:
                return "Schedule maintenance before next flight. Monitor affected systems closely."
        else:
            return "Continue with regular maintenance schedule. All systems operating normally."
    
    def get_fallback_prediction(self, model_type: str) -> Tuple[Dict, Dict]:
        """Get fallback prediction in case of error"""
        return {
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat(),
            'flight_condition': 'warning',
            'confidence_score': 0.5,
            'rule_based_result': {'critical_failures': 0, 'warnings': 1, 'normal_parameters': 0},
            'recommendation': 'Unable to generate detailed prediction. Manual inspection recommended.'
        }, {
            'predictions': [{
                'fault_type': 'prediction_error',
                'description': 'Prediction engine encountered an error',
                'severity': 'warning',
                'recommended_action': 'Manual inspection required',
                'confidence': 0.5,
                'is_critical': False
            }],
            'total_faults': 1,
            'critical_faults': 0,
            'warning_faults': 1,
            'average_confidence': 0.5
        }
    
    def batch_predict(self, survey_data_list: List[Dict], model_type: str) -> List[Tuple[Dict, Dict]]:
        """Perform batch predictions"""
        results = []
        for survey_data in survey_data_list:
            prediction, faults = self.predict_flight_safety(survey_data, model_type)
            results.append((prediction, faults))
        return results

# Global prediction engine instance
prediction_engine = PredictionEngine()

def predict_flight_safety(survey_data: Dict, model_type: str) -> Tuple[Dict, Dict]:
    """Predict flight safety (compatibility function)"""
    return prediction_engine.predict_flight_safety(survey_data, model_type)