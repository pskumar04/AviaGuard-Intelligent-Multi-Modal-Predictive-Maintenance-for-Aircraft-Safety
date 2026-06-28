"""
Survey Data Processing Utility
"""

import json
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SurveyProcessor:
    """Process and validate survey data"""
    
    def __init__(self):
        self.validation_rules = self.load_validation_rules()
    
    def load_validation_rules(self) -> Dict:
        """Load validation rules for survey data"""
        return {
            'boeing_737': {
                'engine': {
                    'n1_rpm': {'type': 'float', 'min': 0, 'max': 120, 'required': True},
                    'n2_rpm': {'type': 'float', 'min': 0, 'max': 120, 'required': True},
                    'egt': {'type': 'float', 'min': 0, 'max': 1000, 'required': True},
                    'oil_temperature': {'type': 'float', 'min': 0, 'max': 200, 'required': True},
                    'oil_pressure': {'type': 'float', 'min': 0, 'max': 150, 'required': True},
                    'vibration_level': {'type': 'float', 'min': 0, 'max': 2, 'required': True}
                },
                'structural': {
                    'wing_vibration': {'type': 'float', 'min': 0, 'max': 1, 'required': True},
                    'fuselage_strain': {'type': 'float', 'min': 0, 'max': 1000, 'required': True}
                }
            },
            'cessna_172': {
                'engine': {
                    'rpm': {'type': 'float', 'min': 0, 'max': 3000, 'required': True},
                    'manifold_pressure': {'type': 'float', 'min': 0, 'max': 40, 'required': True},
                    'oil_temperature': {'type': 'float', 'min': 0, 'max': 300, 'required': True},
                    'oil_pressure': {'type': 'float', 'min': 0, 'max': 100, 'required': True},
                    'fuel_pressure': {'type': 'float', 'min': 0, 'max': 10, 'required': True},
                    'cylinder_head_temp': {'type': 'float', 'min': 0, 'max': 600, 'required': True}
                }
            },
            'airbus_h125': {
                'rotor_system': {
                    'rotor_rpm': {'type': 'float', 'min': 0, 'max': 500, 'required': True},
                    'vibration_level': {'type': 'float', 'min': 0, 'max': 1, 'required': True},
                    'track_and_balance': {'type': 'float', 'min': 0, 'max': 0.5, 'required': True}
                },
                'engine': {
                    'torque': {'type': 'float', 'min': 0, 'max': 100, 'required': True},
                    'ng': {'type': 'float', 'min': 0, 'max': 120, 'required': True},
                    't4': {'type': 'float', 'min': 0, 'max': 1000, 'required': True}
                }
            }
        }
    
    def process_survey_data(self, survey_data: Dict, model_type: str) -> Dict:
        """Process and validate survey data"""
        try:
            # Validate data structure
            if not self.validate_structure(survey_data, model_type):
                raise ValueError("Invalid survey data structure")
            
            # Validate values
            validation_result = self.validate_values(survey_data, model_type)
            if not validation_result['valid']:
                logger.warning(f"Validation warnings: {validation_result['warnings']}")
            
            # Normalize data
            normalized_data = self.normalize_data(survey_data, model_type)
            
            # Extract features for ML model
            features = self.extract_features(normalized_data, model_type)
            
            # Calculate data quality score
            quality_score = self.calculate_quality_score(survey_data, model_type)
            
            result = {
                'raw_data': survey_data,
                'normalized_data': normalized_data,
                'features': features,
                'validation_result': validation_result,
                'quality_score': quality_score,
                'processing_time': datetime.utcnow().isoformat(),
                'model_type': model_type
            }
            
            logger.info(f"Survey data processed for {model_type}. Quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Survey processing error: {str(e)}")
            return self.get_fallback_processing(survey_data, model_type)
    
    def validate_structure(self, data: Dict, model_type: str) -> bool:
        """Validate data structure"""
        rules = self.validation_rules.get(model_type, {})
        
        for category, fields in rules.items():
            if category not in data:
                logger.error(f"Missing category: {category}")
                return False
            
            for field, rule in fields.items():
                if rule['required'] and field not in data[category]:
                    logger.error(f"Missing required field: {field} in {category}")
                    return False
        
        return True
    
    def validate_values(self, data: Dict, model_type: str) -> Dict:
        """Validate field values"""
        rules = self.validation_rules.get(model_type, {})
        warnings = []
        errors = []
        
        for category, fields in rules.items():
            if category in data:
                for field, rule in fields.items():
                    if field in data[category]:
                        value = data[category][field]
                        
                        # Type validation
                        try:
                            if rule['type'] == 'float':
                                float_value = float(value)
                                data[category][field] = float_value
                                value = float_value
                            elif rule['type'] == 'int':
                                int_value = int(value)
                                data[category][field] = int_value
                                value = int_value
                        except (ValueError, TypeError):
                            errors.append(f"{field}: Invalid type, expected {rule['type']}")
                            continue
                        
                        # Range validation
                        if 'min' in rule and value < rule['min']:
                            warnings.append(f"{field}: Value {value} below minimum {rule['min']}")
                        
                        if 'max' in rule and value > rule['max']:
                            warnings.append(f"{field}: Value {value} above maximum {rule['max']}")
        
        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'has_warnings': len(warnings) > 0,
            'has_errors': len(errors) > 0
        }
    
    def normalize_data(self, data: Dict, model_type: str) -> Dict:
        """Normalize data to standard ranges"""
        normalized = {}
        rules = self.validation_rules.get(model_type, {})
        
        for category, fields in rules.items():
            if category in data:
                normalized[category] = {}
                for field, rule in fields.items():
                    if field in data[category]:
                        value = data[category][field]
                        
                        # Min-max normalization to [0, 1]
                        if 'min' in rule and 'max' in rule:
                            min_val = rule['min']
                            max_val = rule['max']
                            if max_val > min_val:
                                normalized_value = (value - min_val) / (max_val - min_val)
                                normalized_value = max(0, min(1, normalized_value))  # Clamp to [0, 1]
                                normalized[category][field] = normalized_value
                            else:
                                normalized[category][field] = 0.5  # Default if min == max
                        else:
                            normalized[category][field] = value
        
        return normalized
    
    def extract_features(self, normalized_data: Dict, model_type: str) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        if model_type == 'boeing_737':
            # Engine features
            engine = normalized_data.get('engine', {})
            features.extend([
                engine.get('n1_rpm', 0.5),
                engine.get('n2_rpm', 0.5),
                engine.get('egt', 0.5),
                engine.get('oil_temperature', 0.5),
                engine.get('oil_pressure', 0.5),
                engine.get('vibration_level', 0.5)
            ])
            
            # Structural features
            structural = normalized_data.get('structural', {})
            features.extend([
                structural.get('wing_vibration', 0.5),
                structural.get('fuselage_strain', 0.5)
            ])
            
            # Derived features
            avg_engine = np.mean([
                engine.get('n1_rpm', 0.5),
                engine.get('n2_rpm', 0.5),
                engine.get('oil_pressure', 0.5)
            ])
            features.append(avg_engine)
            
            vibration_score = engine.get('vibration_level', 0.5) + structural.get('wing_vibration', 0.5)
            features.append(vibration_score / 2)
        
        elif model_type == 'cessna_172':
            engine = normalized_data.get('engine', {})
            features.extend([
                engine.get('rpm', 0.5),
                engine.get('manifold_pressure', 0.5),
                engine.get('oil_temperature', 0.5),
                engine.get('oil_pressure', 0.5),
                engine.get('fuel_pressure', 0.5),
                engine.get('cylinder_head_temp', 0.5)
            ])
        
        elif model_type == 'airbus_h125':
            rotor = normalized_data.get('rotor_system', {})
            engine = normalized_data.get('engine', {})
            
            features.extend([
                rotor.get('rotor_rpm', 0.5),
                rotor.get('vibration_level', 0.5),
                rotor.get('track_and_balance', 0.5),
                engine.get('torque', 0.5),
                engine.get('ng', 0.5),
                engine.get('t4', 0.5)
            ])
        
        # Ensure minimum feature count
        while len(features) < 10:
            features.append(0.5)
        
        return np.array(features[:20])  # Limit to 20 features
    
    def calculate_quality_score(self, data: Dict, model_type: str) -> float:
        """Calculate data quality score"""
        rules = self.validation_rules.get(model_type, {})
        total_fields = 0
        valid_fields = 0
        
        for category, fields in rules.items():
            if category in data:
                for field, rule in fields.items():
                    total_fields += 1
                    if field in data[category]:
                        value = data[category][field]
                        
                        # Check if value is within range
                        if 'min' in rule and 'max' in rule:
                            if rule['min'] <= value <= rule['max']:
                                valid_fields += 1
                            else:
                                # Partial credit for being close
                                if rule['min'] * 0.9 <= value <= rule['max'] * 1.1:
                                    valid_fields += 0.5
                        else:
                            valid_fields += 1  # Count as valid if no range specified
        
        if total_fields == 0:
            return 0.5
        
        quality_score = valid_fields / total_fields
        return min(max(quality_score, 0.0), 1.0)
    
    def get_fallback_processing(self, data: Dict, model_type: str) -> Dict:
        """Get fallback processing result"""
        return {
            'raw_data': data,
            'normalized_data': {},
            'features': np.zeros(10),
            'validation_result': {
                'valid': False,
                'warnings': ['Processing failed, using fallback'],
                'errors': [],
                'has_warnings': True,
                'has_errors': False
            },
            'quality_score': 0.3,
            'processing_time': datetime.utcnow().isoformat(),
            'model_type': model_type
        }
    
    def batch_process(self, survey_data_list: List[Dict], model_type: str) -> List[Dict]:
        """Process multiple survey data entries"""
        results = []
        for survey_data in survey_data_list:
            result = self.process_survey_data(survey_data, model_type)
            results.append(result)
        return results
    
    def prepare_for_ml(self, processed_data: Dict) -> Dict:
        """Prepare data for ML model input"""
        features = processed_data['features']
        
        # Reshape for different model inputs
        ml_input = {
            'vibration_data': np.random.randn(1, 1000, 1),
            'thermal_data': np.random.randn(1, 500, 1),
            'acoustic_data': np.random.randn(1, 250, 1),
            'pressure_data': np.random.randn(1, 100, 1),
            'features': features.reshape(1, -1)
        }
        
        return ml_input

# Global survey processor instance
survey_processor = SurveyProcessor()

def process_survey_data(survey_data: Dict, model_type: str) -> Dict:
    """Process survey data (compatibility function)"""
    return survey_processor.process_survey_data(survey_data, model_type)