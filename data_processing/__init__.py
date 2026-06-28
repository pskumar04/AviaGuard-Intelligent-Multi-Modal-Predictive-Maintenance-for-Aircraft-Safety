"""
Multi-modal Aircraft Predictive Maintenance - Data Processing Module
"""

from .multi_sensor_loader import MultiSensorDataLoader
from .sensor_fusion import AdvancedSensorFusion
from .edge_data_processor import EdgeDataProcessor

__all__ = [
    'MultiSensorDataLoader',
    'AdvancedSensorFusion',
    'EdgeDataProcessor'
]

__version__ = '1.0.0'
__author__ = 'Aircraft Predictive Maintenance Team'