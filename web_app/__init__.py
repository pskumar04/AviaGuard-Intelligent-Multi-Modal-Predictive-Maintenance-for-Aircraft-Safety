"""
Multi-modal Aircraft Predictive Maintenance - Web Application Module
Real-time dashboard and edge deployment interface
"""

from .multi_modal_dashboard import MultiModalDashboard
from .edge_deployment import EdgeDeploymentInterface

__all__ = [
    'MultiModalDashboard',
    'EdgeDeploymentInterface'
]

__version__ = '1.0.0'
__author__ = 'Aircraft Predictive Maintenance Team'