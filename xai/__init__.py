"""
Explainable AI (XAI) Module for Multi-Modal Aircraft Predictive Maintenance
Provides model interpretability and explanation generation
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .visualization import XAIVisualizer

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer', 
    'XAIVisualizer'
]

__version__ = '1.0.0'
__author__ = 'Aircraft Predictive Maintenance Team'