"""
XAI Visualization Module
Creates interactive visualizations for model explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

class XAIVisualizer:
    """Visualization generator for XAI explanations"""
    
    def __init__(self, config_path: str = "visualization_config.json"):
        self.config = self._load_config(config_path)
        self.color_palette = self._get_color_palette()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load visualization configuration"""
        
        default_config = {
            "plot_style": "seaborn",
            "color_scheme": "viridis",
            "figure_size": (12, 8),
            "dpi": 300,
            "interactive_mode": True,
            "save_format": "png",
            "animation_enabled": False
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Visualization config {config_path} not found. Using defaults.")
        
        return default_config
    
    def _get_color_palette(self) -> Dict:
        """Get color palette for visualizations"""
        
        return {
            'vibration': '#FF6B6B',      # Red
            'thermal': '#FFA726',        # Orange
            'acoustic': '#66BB6A',       # Green
            'pressure': '#42A5F5',       # Blue
            'aircraft': '#AB47BC',       # Purple
            'normal': '#4CAF50',         # Green
            'warning': '#FFC107',        # Yellow
            'critical': '#F44336',       # Red
            'positive': '#2196F3',       # Blue
            'negative': '#FF5252',       # Red
            'neutral': '#9E9E9E'         # Gray
        }
    
    def plot_feature_importance(self, 
                               feature_importance: pd.DataFrame,
                               top_n: int = 20,
                               title: str = "Feature Importance") -> go.Figure:
        """Plot feature importance"""
        
        # Get top features
        top_features = feature_importance.head(top_n).copy()
        top_features = top_features.sort_values('importance', ascending=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            marker_color=self.color_palette['positive'],
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_shap_summary(self, 
                         shap_values: Any,
                         feature_names: List[str],
                         max_display: int = 20) -> go.Figure:
        """Plot SHAP summary plot"""
        
        # Convert SHAP values to array
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = shap_values
        
        # Calculate mean absolute SHAP values
        if len(shap_array.shape) == 3:  # Multi-class
            shap_abs = np.abs(shap_array).mean(axis=(0, 2))
        else:  # Binary
            shap_abs = np.abs(shap_array).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(shap_abs)[-max_display:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = shap_abs[top_indices]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=top_features,
            x=top_values,
            orientation='h',
            marker_color=self.color_palette['positive'],
            hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def plot_shap_waterfall(self, 
                           explanation: Dict,
                           title: str = "SHAP Waterfall Plot") -> go.Figure:
        """Plot SHAP waterfall plot for a single instance"""
        
        # Extract data from explanation
        expected_value = explanation.get('expected_value', 0)
        prediction = explanation.get('prediction', 0)
        features = explanation.get('top_contributors', [])
        
        # Prepare data for waterfall
        feature_names = [f['feature_name'] for f in features]
        values = [f['contribution'] for f in features]
        
        # Calculate cumulative values
        cumulative = expected_value
        cumulative_values = [cumulative]
        
        for value in values:
            cumulative += value
            cumulative_values.append(cumulative)
        
        # Create figure
        fig = go.Figure(go.Waterfall(
            name="Feature Contributions",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(feature_names) + ["total"],
            x=["Expected Value"] + feature_names + ["Prediction"],
            textposition="outside",
            text=[f"{v:.3f}" for v in [expected_value] + values + [prediction]],
            y=[expected_value] + values + [0],  # Last is dummy for total
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.color_palette['negative']}},
            increasing={"marker": {"color": self.color_palette['positive']}},
            totals={"marker": {"color": self.color_palette['critical']}}
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def plot_multi_sensor_contributions(self, 
                                       explanations: List[Dict],
                                       title: str = "Multi-Sensor Contributions") -> go.Figure:
        """Plot contributions from different sensor types"""
        
        # Extract sensor contributions
        sensor_types = ['vibration', 'thermal', 'acoustic', 'pressure', 'aircraft']
        contributions = {sensor: [] for sensor in sensor_types}
        
        for exp in explanations:
            sensor_contrib = exp.get('feature_contributions', {})
            for sensor in sensor_types:
                contributions[sensor].append(sensor_contrib.get(sensor, 0))
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{sensor.title()} Contributions" for sensor in sensor_types],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = [
            self.color_palette['vibration'],
            self.color_palette['thermal'],
            self.color_palette['acoustic'],
            self.color_palette['pressure'],
            self.color_palette['aircraft']
        ]
        
        # Add histograms for each sensor
        for i, (sensor, values) in enumerate(contributions.items()):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=sensor,
                    marker_color=colors[i],
                    nbinsx=20,
                    opacity=0.7
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Contribution", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def plot_fault_pattern_comparison(self, 
                                     fault_explanations: Dict,
                                     normal_explanations: Dict,
                                     title: str = "Fault vs Normal Patterns") -> go.Figure:
        """Compare fault and normal patterns"""
        
        # Calculate average contributions
        def calculate_averages(explanations):
            sensors = ['vibration', 'thermal', 'acoustic', 'pressure']
            averages = {sensor: [] for sensor in sensors}
            
            for exp in explanations.values():
                contribs = exp.get('feature_contributions', {})
                for sensor in sensors:
                    averages[sensor].append(contribs.get(sensor, 0))
            
            return {sensor: np.mean(values) for sensor, values in averages.items()}
        
        fault_avg = calculate_averages(fault_explanations)
        normal_avg = calculate_averages(normal_explanations)
        
        # Prepare data
        sensors = list(fault_avg.keys())
        fault_values = [fault_avg[s] for s in sensors]
        normal_values = [normal_avg[s] for s in sensors]
        differences = [fault_avg[s] - normal_avg[s] for s in sensors]
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Average Contributions", "Differences (Fault - Normal)"],
            vertical_spacing=0.2
        )
        
        # Plot 1: Average contributions
        fig.add_trace(
            go.Bar(
                name='Fault',
                x=sensors,
                y=fault_values,
                marker_color=self.color_palette['critical']
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='Normal',
                x=sensors,
                y=normal_values,
                marker_color=self.color_palette['normal']
            ),
            row=1, col=1
        )
        
        # Plot 2: Differences
        colors = [
            self.color_palette['positive'] if d > 0 else self.color_palette['negative']
            for d in differences
        ]
        
        fig.add_trace(
            go.Bar(
                x=sensors,
                y=differences,
                marker_color=colors,
                text=[f"{d:.3f}" for d in differences],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            barmode='group',
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Contribution", row=1, col=1)
        fig.update_yaxes(title_text="Difference", row=2, col=1)
        
        return fig
    
    def plot_interactive_feature_analysis(self, 
                                         explanations: List[Dict],
                                         feature_names: List[str]) -> go.Figure:
        """Create interactive feature analysis visualization"""
        
        # Prepare data
        data = []
        for i, exp in enumerate(explanations):
            for feature in exp.get('top_contributors', []):
                data.append({
                    'sample': i,
                    'feature': feature['feature_name'],
                    'feature_type': feature['feature_type'],
                    'contribution': feature['contribution'],
                    'absolute_contribution': abs(feature['contribution']),
                    'feature_value': feature['feature_value']
                })
        
        df = pd.DataFrame(data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='feature',
            y='contribution',
            color='feature_type',
            size='absolute_contribution',
            hover_data=['feature_value', 'sample'],
            color_discrete_map={
                'vibration': self.color_palette['vibration'],
                'thermal': self.color_palette['thermal'],
                'acoustic': self.color_palette['acoustic'],
                'pressure': self.color_palette['pressure'],
                'aircraft': self.color_palette['aircraft']
            },
            title="Interactive Feature Contribution Analysis"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title="Feature",
            yaxis_title="Contribution",
            template="plotly_white",
            xaxis={'categoryorder': 'total descending'}
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def plot_time_series_explanations(self, 
                                     time_series_data: Dict,
                                     explanations: Dict,
                                     title: str = "Time Series with Explanations") -> go.Figure:
        """Plot time series data with explanation overlays"""
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Sensor Readings", "Explanation Contributions"],
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Plot 1: Sensor readings
        sensor_colors = {
            'vibration': self.color_palette['vibration'],
            'thermal': self.color_palette['thermal'],
            'acoustic': self.color_palette['acoustic'],
            'pressure': self.color_palette['pressure']
        }
        
        for sensor, data in time_series_data.items():
            if sensor in sensor_colors:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(data))),
                        y=data,
                        name=sensor,
                        mode='lines',
                        line=dict(color=sensor_colors[sensor], width=2)
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Explanation contributions
        timestamps = list(explanations.keys())
        contributions = []
        
        for ts in timestamps:
            exp = explanations[ts]
            total_contrib = sum(exp.get('feature_contributions', {}).values())
            contributions.append(total_contrib)
        
        # Color contributions based on value
        colors = []
        for contrib in contributions:
            if contrib > 0.3:
                colors.append(self.color_palette['critical'])
            elif contrib > 0.1:
                colors.append(self.color_palette['warning'])
            else:
                colors.append(self.color_palette['normal'])
        
        fig.add_trace(
            go.Bar(
                x=timestamps,
                y=contributions,
                name='Total Contribution',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=700,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text="Sensor Value", row=1, col=1)
        fig.update_yaxes(title_text="Contribution", row=2, col=1)
        fig.update_xaxes(title_text="Time Step", row=2, col=1)
        
        return fig
    
    def create_dashboard(self, 
                        explanations: Dict,
                        save_path: str = "xai_dashboard.html") -> go.Figure:
        """Create comprehensive XAI dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Feature Importance",
                "SHAP Summary",
                "Sensor Contributions",
                "Fault Patterns",
                "Time Series Analysis",
                "Interactive Analysis"
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Placeholder for actual plots
        # In practice, you would call the specific plotting methods
        # and add their traces to the subplots
        
        # Update layout
        fig.update_layout(
            title="Comprehensive XAI Dashboard",
            height=1200,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save dashboard
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Dashboard saved to {save_path}")
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filepath: str, format: str = None):
        """Save visualization to file"""
        
        if format is None:
            format = self.config['save_format']
        
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath, width=self.config['figure_size'][0] * 100,
                           height=self.config['figure_size'][1] * 100)
        elif format == 'pdf':
            fig.write_image(filepath, format='pdf')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✅ Visualization saved to {filepath}")


# Test function
def test_xai_visualizer():
    """Test XAI visualizer"""
    
    print("Testing XAI Visualizer...")
    
    # Create visualizer
    visualizer = XAIVisualizer()
    
    # Generate test data
    np.random.seed(42)
    
    # Test 1: Feature importance
    print("\n1. Testing feature importance plot...")
    feature_names = [f"feature_{i}" for i in range(20)]
    importance_values = np.random.randn(20)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(importance_values)
    }).sort_values('importance', ascending=False)
    
    fig1 = visualizer.plot_feature_importance(feature_importance, title="Test Feature Importance")
    visualizer.save_visualization(fig1, "test_feature_importance.html")
    
    # Test 2: SHAP summary
    print("\n2. Testing SHAP summary plot...")
    # Simulated SHAP values
    shap_values = np.random.randn(100, 1853)
    fig2 = visualizer.plot_shap_summary(shap_values, feature_names[:1853], max_display=15)
    visualizer.save_visualization(fig2, "test_shap_summary.html")
    
    # Test 3: Waterfall plot
    print("\n3. Testing waterfall plot...")
    explanation = {
        'expected_value': 0.5,
        'prediction': 0.8,
        'top_contributors': [
            {'feature_name': f'feature_{i}', 'contribution': np.random.randn() * 0.1}
            for i in range(10)
        ]
    }
    fig3 = visualizer.plot_shap_waterfall(explanation, title="Test Waterfall Plot")
    visualizer.save_visualization(fig3, "test_waterfall.html")
    
    # Test 4: Multi-sensor contributions
    print("\n4. Testing multi-sensor contributions plot...")
    explanations = []
    for _ in range(20):
        explanations.append({
            'feature_contributions': {
                'vibration': np.random.randn(),
                'thermal': np.random.randn(),
                'acoustic': np.random.randn(),
                'pressure': np.random.randn(),
                'aircraft': np.random.randn()
            }
        })
    
    fig4 = visualizer.plot_multi_sensor_contributions(explanations, title="Test Sensor Contributions")
    visualizer.save_visualization(fig4, "test_sensor_contributions.html")
    
    # Test 5: Fault pattern comparison
    print("\n5. Testing fault pattern comparison...")
    fault_explanations = {}
    normal_explanations = {}
    
    for i in range(10):
        fault_explanations[f'fault_{i}'] = {
            'feature_contributions': {
                'vibration': np.random.randn() * 0.5 + 0.3,
                'thermal': np.random.randn() * 0.3 + 0.1,
                'acoustic': np.random.randn() * 0.4 + 0.2,
                'pressure': np.random.randn() * 0.2 + 0.1
            }
        }
        
        normal_explanations[f'normal_{i}'] = {
            'feature_contributions': {
                'vibration': np.random.randn() * 0.2,
                'thermal': np.random.randn() * 0.1,
                'acoustic': np.random.randn() * 0.15,
                'pressure': np.random.randn() * 0.1
            }
        }
    
    fig5 = visualizer.plot_fault_pattern_comparison(
        fault_explanations, normal_explanations,
        title="Test Fault vs Normal Comparison"
    )
    visualizer.save_visualization(fig5, "test_fault_comparison.html")
    
    # Test 6: Interactive feature analysis
    print("\n6. Testing interactive feature analysis...")
    test_explanations = []
    for _ in range(50):
        test_explanations.append({
            'top_contributors': [
                {
                    'feature_name': f'feature_{np.random.randint(0, 100)}',
                    'feature_type': np.random.choice(['vibration', 'thermal', 'acoustic', 'pressure']),
                    'contribution': np.random.randn() * 0.2,
                    'feature_value': np.random.randn()
                }
                for _ in range(np.random.randint(3, 8))
            ]
        })
    
    fig6 = visualizer.plot_interactive_feature_analysis(
        test_explanations, feature_names[:100]
    )
    visualizer.save_visualization(fig6, "test_interactive_analysis.html")
    
    # Test 7: Dashboard creation
    print("\n7. Testing dashboard creation...")
    dashboard_data = {
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'feature_names': feature_names[:1853],
        'explanations': explanations,
        'fault_explanations': fault_explanations,
        'normal_explanations': normal_explanations
    }
    
    # Note: In practice, you would create a real dashboard with actual data
    print("   Dashboard creation would require actual model explanations")
    
    print("\n✅ XAI visualizer test completed!")
    print("\nGenerated visualizations:")
    print("  - test_feature_importance.html")
    print("  - test_shap_summary.html")
    print("  - test_waterfall.html")
    print("  - test_sensor_contributions.html")
    print("  - test_fault_comparison.html")
    print("  - test_interactive_analysis.html")
    
    return visualizer


if __name__ == "__main__":
    test_xai_visualizer()