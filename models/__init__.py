"""
Multi-modal Aircraft Predictive Maintenance - Models Module
Hybrid CNN+Transformer models for multi-class fault diagnosis
"""

from .hybrid_cnn_transformer import MultiModalHybridModel, MultiClassFaultClassifier
from .edge_optimizer import EdgeModelOptimizer
from .train_multi_modal import MultiModalTrainer

__all__ = [
    'MultiModalHybridModel',
    'MultiClassFaultClassifier',
    'EdgeModelOptimizer',
    'MultiModalTrainer'
]

__version__ = '1.0.0'
__author__ = 'Aircraft Predictive Maintenance Team'