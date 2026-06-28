"""
Real-Time Monitoring System for Edge Devices
Monitors system performance, sensor data, and inference results in real-time
"""

import time
import threading
import queue
import numpy as np
import json
import os
import psutil
import GPUtil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class RealTimeMonitor:
    """Real-time monitoring system for edge deployment"""
    
    def __init__(self, 
                 config_path: str = "monitor_config.json",
                 update_interval: float = 1.0):
        
        self.config = self._load_config(config_path)
        self.update_interval = update_interval
        self.running = False
        
        # Data storage
        self.metrics_history = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'temperature': [],
            'inference_time': [],
            'throughput': [],
            'fault_count': [],
            'timestamps': []
        }
        
        # Alerts
        self.alerts = []
        self.alert_thresholds = {
            'cpu_usage': 90.0,      # %
            'memory_usage': 85.0,    # %
            'gpu_usage': 95.0,       # %
            'gpu_memory': 90.0,      # %
            'temperature': 85.0,     # °C
            'inference_time': 15.0,  # ms
            'throughput': 50.0       # samples/sec
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_inferences': 0,
            'total_faults': 0,
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'uptime': 0.0
        }
        
        # Queues for external data
        self.inference_queue = queue.Queue(maxsize=1000)
        self.sensor_queue = queue.Queue(maxsize=1000)
        
        # Threads
        self.monitor_thread = None
        self.alert_thread = None
        
        # Start time
        self.start_time = time.time()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitor configuration"""
        
        default_config = {
            "metrics_to_monitor": [
                "cpu_usage",
                "memory_usage", 
                "gpu_usage",
                "gpu_memory",
                "temperature",
                "inference_time",
                "throughput",
                "fault_count"
            ],
            "history_size": 3600,  # 1 hour at 1-second intervals
            "alert_cooldown": 60,   # seconds between same alert
            "log_file": "logs/monitor.log",
            "dashboard_update_interval": 1.0,
            "performance_report_interval": 300,  # 5 minutes
            "data_retention_days": 7
        }
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Monitor config {config_path} not found. Using defaults.")
        
        return default_config
    
    def get_system_metrics(self) -> Dict:
        """Collect system metrics"""
        
        metrics = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters(),
            'process_count': len(psutil.pids()),
            'uptime': time.time() - self.start_time
        }
        
        # GPU metrics (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                metrics.update({
                    'gpu_usage': gpu.load * 100,
                    'gpu_memory': gpu.memoryUtil * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_temperature': gpu.temperature
                })
        except:
            metrics.update({
                'gpu_usage': 0.0,
                'gpu_memory': 0.0,
                'gpu_temperature': 0.0
            })
        
        # Temperature (Linux systems)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                metrics['temperature'] = temp
        except:
            metrics['temperature'] = 0.0
        
        return metrics
    
    def add_inference_result(self, inference_result: Dict):
        """Add inference result to queue"""
        try:
            self.inference_queue.put_nowait(inference_result)
        except queue.Full:
            # Remove oldest result if queue is full
            try:
                self.inference_queue.get_nowait()
                self.inference_queue.put_nowait(inference_result)
            except queue.Empty:
                pass
    
    def add_sensor_data(self, sensor_data: Dict):
        """Add sensor data to queue"""
        try:
            self.sensor_queue.put_nowait(sensor_data)
        except queue.Full:
            # Remove oldest data if queue is full
            try:
                self.sensor_queue.get_nowait()
                self.sensor_queue.put_nowait(sensor_data)
            except queue.Empty:
                pass
    
    def process_inference_data(self):
        """Process inference data from queue"""
        inference_results = []
        
        # Get all available inference results
        while not self.inference_queue.empty():
            try:
                result = self.inference_queue.get_nowait()
                inference_results.append(result)
            except queue.Empty:
                break
        
        if not inference_results:
            return
        
        # Calculate metrics
        inference_times = [r.get('inference_time_ms', 0) for r in inference_results]
        fault_counts = [len(r.get('fault_predictions', [])) for r in inference_results]
        
        # Update performance stats
        self.performance_stats['total_inferences'] += len(inference_results)
        self.performance_stats['total_faults'] += sum(fault_counts)
        
        if inference_times:
            avg_time = np.mean(inference_times)
            max_time = np.max(inference_times)
            
            # Update rolling average
            if self.performance_stats['avg_inference_time'] == 0:
                self.performance_stats['avg_inference_time'] = avg_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.performance_stats['avg_inference_time'] = (
                    alpha * avg_time + 
                    (1 - alpha) * self.performance_stats['avg_inference_time']
                )
            
            self.performance_stats['max_inference_time'] = max(
                self.performance_stats['max_inference_time'],
                max_time
            )
        
        # Store in history
        current_time = time.time()
        if inference_times:
            self.metrics_history['inference_time'].append(np.mean(inference_times))
            self.metrics_history['throughput'].append(1000 / np.mean(inference_times) 
                                                    if np.mean(inference_times) > 0 else 0)
            self.metrics_history['fault_count'].append(sum(fault_counts))
            self.metrics_history['timestamps'].append(current_time)
        
        # Check for limit and trim
        history_size = self.config['history_size']
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > history_size:
                self.metrics_history[key] = self.metrics_history[key][-history_size:]
    
    def process_sensor_data(self):
        """Process sensor data from queue"""
        sensor_readings = {}
        
        # Get latest sensor readings
        while not self.sensor_queue.empty():
            try:
                data = self.sensor_queue.get_nowait()
                sensor_name = data.get('sensor', 'unknown')
                sensor_readings[sensor_name] = data
            except queue.Empty:
                break
        
        # Store sensor anomalies
        for sensor_name, data in sensor_readings.items():
            anomalies = data.get('anomalies', [])
            if anomalies:
                self._create_alert(
                    level='warning',
                    source=f'sensor_{sensor_name}',
                    message=f"Sensor anomalies detected: {', '.join(anomalies[:3])}",
                    data=data
                )
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        last_performance_report = time.time()
        performance_interval = self.config['performance_report_interval']
        
        while self.running:
            try:
                cycle_start = time.time()
                
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Store in history
                for metric in self.config['metrics_to_monitor']:
                    if metric in system_metrics:
                        self.metrics_history[metric].append(system_metrics[metric])
                
                self.metrics_history['timestamps'].append(system_metrics['timestamp'])
                
                # Process inference and sensor data
                self.process_inference_data()
                self.process_sensor_data()
                
                # Check thresholds and create alerts
                self._check_thresholds(system_metrics)
                
                # Trim history if needed
                history_size = self.config['history_size']
                for key in self.metrics_history:
                    if len(self.metrics_history[key]) > history_size:
                        self.metrics_history[key] = self.metrics_history[key][-history_size:]
                
                # Generate performance report periodically
                current_time = time.time()
                if current_time - last_performance_report >= performance_interval:
                    self._generate_performance_report()
                    last_performance_report = current_time
                
                # Log to file
                self._log_metrics(system_metrics)
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _alert_loop(self):
        """Alert processing loop"""
        
        last_alert_times = {}
        alert_cooldown = self.config['alert_cooldown']
        
        while self.running:
            try:
                # Process alerts (e.g., send notifications, log, etc.)
                if self.alerts:
                    alert = self.alerts[-1]  # Get latest alert
                    alert_key = f"{alert['source']}_{alert['level']}"
                    
                    # Check cooldown
                    current_time = time.time()
                    last_time = last_alert_times.get(alert_key, 0)
                    
                    if current_time - last_time >= alert_cooldown:
                        # Process alert
                        self._process_alert(alert)
                        last_alert_times[alert_key] = current_time
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in alert loop: {e}")
                time.sleep(5)
    
    def _check_thresholds(self, metrics: Dict):
        """Check metrics against thresholds"""
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                self._create_alert(
                    level='critical' if metrics[metric] > threshold * 1.2 else 'warning',
                    source='system',
                    message=f"{metric.replace('_', ' ').title()} exceeded threshold: "
                           f"{metrics[metric]:.1f} > {threshold:.1f}",
                    data={metric: metrics[metric], 'threshold': threshold}
                )
    
    def _create_alert(self, level: str, source: str, message: str, data: Dict = None):
        """Create a new alert"""
        
        alert = {
            'timestamp': time.time(),
            'level': level,
            'source': source,
            'message': message,
            'data': data or {}
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _process_alert(self, alert: Dict):
        """Process an alert (log, notify, etc.)"""
        
        timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {alert['level'].upper()}: {alert['message']}"
        
        # Log to console
        if alert['level'] == 'critical':
            print(f"🚨 {log_message}")
        elif alert['level'] == 'warning':
            print(f"⚠️  {log_message}")
        else:
            print(f"ℹ️  {log_message}")
        
        # Log to file
        self._log_alert(alert)
        
        # Here you could add notification methods:
        # - Send email
        # - Send SMS
        # - Push notification
        # - MQTT message
        # - Webhook
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to file"""
        
        log_file = self.config['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        timestamp = datetime.fromtimestamp(metrics['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            'timestamp': timestamp,
            'metrics': {k: v for k, v in metrics.items() if k != 'timestamp'}
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_alert(self, alert: Dict):
        """Log alert to file"""
        
        alert_file = self.config['log_file'].replace('.log', '_alerts.log')
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        
        timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        alert['formatted_timestamp'] = timestamp
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def _generate_performance_report(self):
        """Generate performance report"""
        
        report = {
            'timestamp': time.time(),
            'performance_stats': self.performance_stats.copy(),
            'recent_alerts': len([a for a in self.alerts 
                                 if time.time() - a['timestamp'] < 3600]),
            'system_metrics': self._get_recent_averages()
        }
        
        # Calculate uptime
        report['uptime_hours'] = (time.time() - self.start_time) / 3600
        
        # Save report
        report_file = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Performance report saved to {report_file}")
        
        return report
    
    def _get_recent_averages(self, minutes: int = 5) -> Dict:
        """Get average metrics over recent period"""
        
        recent_cutoff = time.time() - (minutes * 60)
        recent_indices = [i for i, t in enumerate(self.metrics_history['timestamps']) 
                         if t >= recent_cutoff]
        
        averages = {}
        for metric, values in self.metrics_history.items():
            if metric != 'timestamps' and values:
                recent_values = [values[i] for i in recent_indices if i < len(values)]
                if recent_values:
                    averages[metric] = np.mean(recent_values)
        
        return averages
    
    def get_current_status(self) -> Dict:
        """Get current system status"""
        
        system_metrics = self.get_system_metrics()
        recent_averages = self._get_recent_averages(minutes=1)
        
        # Determine overall status
        critical_alerts = len([a for a in self.alerts 
                              if a['level'] == 'critical' and 
                              time.time() - a['timestamp'] < 300])  # Last 5 minutes
        
        warning_alerts = len([a for a in self.alerts 
                             if a['level'] == 'warning' and 
                             time.time() - a['timestamp'] < 300])
        
        if critical_alerts > 0:
            status = 'critical'
        elif warning_alerts > 0:
            status = 'warning'
        else:
            status = 'normal'
        
        return {
            'timestamp': time.time(),
            'status': status,
            'system_metrics': system_metrics,
            'recent_averages': recent_averages,
            'performance_stats': self.performance_stats,
            'active_alerts': {
                'critical': critical_alerts,
                'warning': warning_alerts,
                'total': len(self.alerts)
            },
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def get_historical_data(self, metric: str, minutes: int = 60) -> Tuple[List, List]:
        """Get historical data for a specific metric"""
        
        cutoff = time.time() - (minutes * 60)
        
        if metric not in self.metrics_history:
            return [], []
        
        timestamps = []
        values = []
        
        for i, ts in enumerate(self.metrics_history['timestamps']):
            if ts >= cutoff and i < len(self.metrics_history[metric]):
                timestamps.append(ts)
                values.append(self.metrics_history[metric][i])
        
        return timestamps, values
    
    def start(self):
        """Start monitoring system"""
        
        if self.running:
            print("Monitor already running")
            return
        
        print("Starting real-time monitor...")
        self.running = True
        
        # Create log directory
        os.makedirs('logs', exist_ok=True)
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="monitor_loop"
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Start alert thread
        self.alert_thread = threading.Thread(
            target=self._alert_loop,
            name="alert_loop"
        )
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        print(f"✅ Real-time monitor started (update interval: {self.update_interval}s)")
    
    def stop(self):
        """Stop monitoring system"""
        
        print("Stopping real-time monitor...")
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=2.0)
        
        # Generate final report
        self._generate_performance_report()
        
        print("✅ Real-time monitor stopped")


# Test function
def test_real_time_monitor():
    """Test the real-time monitor"""
    
    print("Testing Real-Time Monitor...")
    
    # Create monitor
    monitor = RealTimeMonitor(update_interval=0.5)
    
    # Start monitor
    monitor.start()
    
    # Simulate some data
    print("\nSimulating data for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        # Simulate inference results
        inference_result = {
            'inference_time_ms': np.random.uniform(5, 20),
            'fault_predictions': [] if np.random.random() > 0.1 else ['fault_1', 'fault_2'],
            'timestamp': time.time()
        }
        
        monitor.add_inference_result(inference_result)
        
        # Simulate sensor data
        sensor_data = {
            'sensor': 'vibration',
            'value': np.random.uniform(0.1, 0.5),
            'anomalies': [] if np.random.random() > 0.05 else ['high_vibration'],
            'timestamp': time.time()
        }
        
        monitor.add_sensor_data(sensor_data)
        
        # Get current status
        if np.random.random() > 0.8:  # 20% chance
            status = monitor.get_current_status()
            print(f"  Status: {status['status']}, "
                  f"CPU: {status['system_metrics']['cpu_usage']:.1f}%, "
                  f"Inferences: {status['performance_stats']['total_inferences']}")
        
        time.sleep(0.1)
    
    # Get historical data
    print("\nHistorical Data (last 1 minute):")
    timestamps, values = monitor.get_historical_data('cpu_usage', minutes=1)
    if timestamps and values:
        print(f"  CPU Usage samples: {len(values)}, "
              f"Avg: {np.mean(values):.1f}%, "
              f"Max: {np.max(values):.1f}%")
    
    # Get alerts
    print(f"\nTotal Alerts: {len(monitor.alerts)}")
    if monitor.alerts:
        print("Recent Alerts:")
        for alert in monitor.alerts[-3:]:  # Last 3 alerts
            time_str = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
            print(f"  [{time_str}] {alert['level']}: {alert['message']}")
    
    # Stop monitor
    monitor.stop()
    
    # Final status
    final_status = monitor.get_current_status()
    print(f"\nFinal Status:")
    print(f"  Uptime: {final_status['uptime_hours']:.2f} hours")
    print(f"  Total Inferences: {final_status['performance_stats']['total_inferences']}")
    print(f"  Total Faults: {final_status['performance_stats']['total_faults']}")
    print(f"  Avg Inference Time: {final_status['performance_stats']['avg_inference_time']:.2f}ms")
    
    print("\n✅ Real-time monitor test completed!")
    return monitor


if __name__ == "__main__":
    # Run test
    monitor = test_real_time_monitor()
    
    # Save test data
    test_results = {
        'performance_stats': monitor.performance_stats,
        'total_alerts': len(monitor.alerts),
        'metrics_history_summary': {
            metric: len(values) for metric, values in monitor.metrics_history.items()
        }
    }
    
    with open("monitor_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to monitor_test_results.json")