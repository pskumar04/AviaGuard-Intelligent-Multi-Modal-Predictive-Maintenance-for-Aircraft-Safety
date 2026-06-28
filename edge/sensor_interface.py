"""
Multi-Sensor Interface for Edge Devices
Handles real-time data acquisition from vibration, thermal, acoustic, and pressure sensors
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import hardware-specific libraries
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import smbus2
    import board
    import busio
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False

try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class SensorType(Enum):
    """Types of sensors supported"""
    VIBRATION = "vibration"
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    PRESSURE = "pressure"
    STRAIN = "strain"
    DISPLACEMENT = "displacement"

class SensorInterface:
    """Interface for multi-sensor data acquisition"""
    
    def __init__(self, config_path: str = "sensor_config.json"):
        self.config = self._load_config(config_path)
        self.sensors = {}
        self.data_buffers = {}
        self.running = False
        self.threads = []
        
        # Data queues
        self.raw_data_queue = queue.Queue(maxsize=10000)
        self.processed_data_queue = queue.Queue(maxsize=1000)
        
        # Initialize sensors
        self._initialize_sensors()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load sensor configuration"""
        
        default_config = {
            "sampling_rates": {
                "vibration": 1000,
                "thermal": 10,
                "acoustic": 44100,
                "pressure": 100
            },
            "sensor_interfaces": {
                "vibration": "i2c",
                "thermal": "i2c",
                "acoustic": "usb",
                "pressure": "spi"
            },
            "calibration": {
                "vibration": {"scale": 1.0, "offset": 0.0},
                "thermal": {"scale": 1.0, "offset": 0.0},
                "acoustic": {"scale": 1.0, "offset": 0.0},
                "pressure": {"scale": 1.0, "offset": 0.0}
            },
            "buffer_sizes": {
                "vibration": 1000,
                "thermal": 100,
                "acoustic": 4410,
                "pressure": 100
            },
            "simulation_mode": True  # Use simulated data if hardware not available
        }
        
        # Try to load user config
        try:
            import json
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
        
        return default_config
    
    def _initialize_sensors(self):
        """Initialize all sensors based on configuration"""
        
        print("Initializing sensors...")
        
        for sensor_type in SensorType:
            sensor_name = sensor_type.value
            
            if sensor_name in self.config["sampling_rates"]:
                sampling_rate = self.config["sampling_rates"][sensor_name]
                interface = self.config["sensor_interfaces"].get(sensor_name, "simulated")
                buffer_size = self.config["buffer_sizes"].get(sensor_name, 100)
                
                # Initialize buffer
                self.data_buffers[sensor_name] = {
                    'data': np.zeros(buffer_size),
                    'timestamps': np.zeros(buffer_size),
                    'index': 0,
                    'size': buffer_size
                }
                
                # Initialize sensor based on interface
                if interface == "i2c" and I2C_AVAILABLE:
                    self._initialize_i2c_sensor(sensor_name)
                elif interface == "spi" and SPI_AVAILABLE:
                    self._initialize_spi_sensor(sensor_name)
                elif interface == "usb" and SERIAL_AVAILABLE:
                    self._initialize_serial_sensor(sensor_name)
                elif interface == "audio" and AUDIO_AVAILABLE:
                    self._initialize_audio_sensor(sensor_name)
                else:
                    # Use simulated sensor
                    self._initialize_simulated_sensor(sensor_name)
                    print(f"  {sensor_name}: Using simulated sensor")
    
    def _initialize_i2c_sensor(self, sensor_name: str):
        """Initialize I2C sensor"""
        try:
            # Common I2C addresses for different sensors
            i2c_addresses = {
                "vibration": 0x68,  # MPU6050
                "thermal": 0x48,    # ADS1115 with thermocouple
                "pressure": 0x76    # BMP280
            }
            
            address = i2c_addresses.get(sensor_name, 0x00)
            
            # Initialize I2C bus
            bus = smbus2.SMBus(1)  # I2C bus 1 on Raspberry Pi/Jetson
            
            self.sensors[sensor_name] = {
                'type': 'i2c',
                'bus': bus,
                'address': address,
                'initialized': True
            }
            
            print(f"  {sensor_name}: I2C sensor initialized at 0x{address:02x}")
            
        except Exception as e:
            print(f"  {sensor_name}: Failed to initialize I2C sensor: {e}")
            self._initialize_simulated_sensor(sensor_name)
    
    def _initialize_spi_sensor(self, sensor_name: str):
        """Initialize SPI sensor"""
        try:
            # Initialize SPI
            spi = spidev.SpiDev()
            spi.open(0, 0)  # Bus 0, Device 0
            spi.max_speed_hz = 1000000
            spi.mode = 0
            
            self.sensors[sensor_name] = {
                'type': 'spi',
                'device': spi,
                'initialized': True
            }
            
            print(f"  {sensor_name}: SPI sensor initialized")
            
        except Exception as e:
            print(f"  {sensor_name}: Failed to initialize SPI sensor: {e}")
            self._initialize_simulated_sensor(sensor_name)
    
    def _initialize_serial_sensor(self, sensor_name: str):
        """Initialize serial/USB sensor"""
        try:
            # Try common serial ports
            ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyS0']
            serial_port = None
            
            for port in ports:
                try:
                    ser = serial.Serial(port, baudrate=9600, timeout=1)
                    serial_port = ser
                    break
                except:
                    continue
            
            if serial_port:
                self.sensors[sensor_name] = {
                    'type': 'serial',
                    'port': serial_port,
                    'initialized': True
                }
                print(f"  {sensor_name}: Serial sensor initialized on {port}")
            else:
                raise Exception("No serial ports available")
                
        except Exception as e:
            print(f"  {sensor_name}: Failed to initialize serial sensor: {e}")
            self._initialize_simulated_sensor(sensor_name)
    
    def _initialize_audio_sensor(self, sensor_name: str):
        """Initialize audio sensor (microphone)"""
        try:
            # Check audio devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if input_devices:
                self.sensors[sensor_name] = {
                    'type': 'audio',
                    'device_id': input_devices[0]['index'],
                    'sampling_rate': self.config["sampling_rates"][sensor_name],
                    'initialized': True
                }
                print(f"  {sensor_name}: Audio sensor initialized")
            else:
                raise Exception("No audio input devices found")
                
        except Exception as e:
            print(f"  {sensor_name}: Failed to initialize audio sensor: {e}")
            self._initialize_simulated_sensor(sensor_name)
    
    def _initialize_simulated_sensor(self, sensor_name: str):
        """Initialize simulated sensor"""
        self.sensors[sensor_name] = {
            'type': 'simulated',
            'initialized': True,
            'simulation_params': self._get_simulation_params(sensor_name)
        }
    
    def _get_simulation_params(self, sensor_name: str) -> Dict:
        """Get parameters for simulated sensor data"""
        
        base_params = {
            "vibration": {
                "base_frequency": 50,  # Hz
                "amplitude": 0.3,      # g
                "noise_level": 0.05    # g
            },
            "thermal": {
                "base_value": 400,     # °C
                "variation": 50,       # °C
                "noise_level": 5       # °C
            },
            "acoustic": {
                "base_value": 75,      # dB
                "variation": 10,       # dB
                "noise_level": 2       # dB
            },
            "pressure": {
                "base_value": 3000,    # PSI
                "variation": 200,      # PSI
                "noise_level": 20      # PSI
            }
        }
        
        return base_params.get(sensor_name, {
            "base_value": 0.0,
            "variation": 1.0,
            "noise_level": 0.1
        })
    
    def read_i2c_sensor(self, sensor_name: str) -> float:
        """Read data from I2C sensor"""
        sensor = self.sensors[sensor_name]
        
        try:
            if sensor_name == "vibration":
                # MPU6050: Read accelerometer data
                # This is a simplified version
                accel_x = sensor['bus'].read_i2c_block_data(sensor['address'], 0x3B, 2)
                accel_y = sensor['bus'].read_i2c_block_data(sensor['address'], 0x3D, 2)
                accel_z = sensor['bus'].read_i2c_block_data(sensor['address'], 0x3F, 2)
                
                # Convert to g (simplified)
                vibration = np.sqrt(
                    (accel_x[0] << 8 | accel_x[1])**2 +
                    (accel_y[0] << 8 | accel_y[1])**2 +
                    (accel_z[0] << 8 | accel_z[1])**2
                ) / 16384.0  # MPU6050 sensitivity
                
                return vibration
                
            elif sensor_name == "thermal":
                # ADS1115: Read temperature
                # This is a simplified version
                data = sensor['bus'].read_i2c_block_data(sensor['address'], 0x00, 2)
                raw_temp = (data[0] << 8) | data[1]
                temperature = raw_temp * 0.0625  # 12-bit resolution, 0.0625°C per LSB
                
                return temperature
                
            else:
                # Generic I2C read
                data = sensor['bus'].read_byte(sensor['address'])
                return float(data)
                
        except Exception as e:
            print(f"Error reading I2C sensor {sensor_name}: {e}")
            return 0.0
    
    def read_spi_sensor(self, sensor_name: str) -> float:
        """Read data from SPI sensor"""
        sensor = self.sensors[sensor_name]
        
        try:
            # SPI read (simplified)
            # For pressure sensors like BMP280
            data = sensor['device'].xfer2([0x00, 0x00, 0x00])
            
            if sensor_name == "pressure":
                # Convert to pressure (simplified)
                raw_pressure = (data[0] << 16) | (data[1] << 8) | data[2]
                pressure = raw_pressure / 100.0  # Convert to PSI (simplified)
                return pressure
            else:
                return float(data[0])
                
        except Exception as e:
            print(f"Error reading SPI sensor {sensor_name}: {e}")
            return 0.0
    
    def read_serial_sensor(self, sensor_name: str) -> float:
        """Read data from serial sensor"""
        sensor = self.sensors[sensor_name]
        
        try:
            # Read line from serial port
            line = sensor['port'].readline().decode('utf-8').strip()
            
            if sensor_name == "acoustic":
                # Parse acoustic data (simplified)
                if "dB" in line:
                    value = float(line.split("dB")[0].strip())
                    return value
                else:
                    return float(line)
            else:
                return float(line)
                
        except Exception as e:
            print(f"Error reading serial sensor {sensor_name}: {e}")
            return 0.0
    
    def read_audio_sensor(self, sensor_name: str) -> float:
        """Read data from audio sensor"""
        sensor = self.sensors[sensor_name]
        
        try:
            # Record audio snippet
            duration = 0.1  # 100ms
            samples = int(duration * sensor['sampling_rate'])
            
            recording = sd.rec(
                samples,
                samplerate=sensor['sampling_rate'],
                channels=1,
                device=sensor['device_id']
            )
            sd.wait()
            
            # Calculate RMS (simplified sound pressure level)
            rms = np.sqrt(np.mean(recording**2))
            
            # Convert to dB (simplified)
            db = 20 * np.log10(rms + 1e-10) + 94  # Reference: 20 μPa
            
            return db
            
        except Exception as e:
            print(f"Error reading audio sensor {sensor_name}: {e}")
            return 0.0
    
    def read_simulated_sensor(self, sensor_name: str) -> float:
        """Generate simulated sensor data"""
        params = self.sensors[sensor_name]['simulation_params']
        
        # Add time-based variation
        t = time.time()
        
        if sensor_name == "vibration":
            # Simulate vibration with harmonics
            base_freq = params["base_frequency"]
            amplitude = params["amplitude"]
            noise = params["noise_level"]
            
            # Create vibration signal
            vibration = (
                amplitude * np.sin(2 * np.pi * base_freq * t) +
                0.3 * amplitude * np.sin(2 * np.pi * 2 * base_freq * t) +
                0.1 * amplitude * np.sin(2 * np.pi * 3 * base_freq * t) +
                np.random.randn() * noise
            )
            
            # Add occasional fault
            if np.random.random() < 0.001:  # 0.1% chance of fault
                vibration += np.random.uniform(0.5, 1.5)
            
            return abs(vibration)  # Vibration is always positive
            
        elif sensor_name == "thermal":
            # Simulate temperature with slow drift
            base_value = params["base_value"]
            variation = params["variation"]
            noise = params["noise_level"]
            
            temperature = (
                base_value +
                variation * np.sin(2 * np.pi * 0.001 * t) +  # Slow drift
                np.random.randn() * noise
            )
            
            # Add occasional overheating
            if np.random.random() < 0.0005:  # 0.05% chance
                temperature += np.random.uniform(50, 200)
            
            return temperature
            
        elif sensor_name == "acoustic":
            # Simulate acoustic levels
            base_value = params["base_value"]
            variation = params["variation"]
            noise = params["noise_level"]
            
            acoustic = (
                base_value +
                variation * np.sin(2 * np.pi * 0.1 * t) +  # 0.1Hz variation
                np.random.randn() * noise
            )
            
            # Add occasional spike
            if np.random.random() < 0.0002:  # 0.02% chance
                acoustic += np.random.uniform(20, 40)
            
            return max(0, acoustic)  # dB can't be negative
            
        elif sensor_name == "pressure":
            # Simulate pressure
            base_value = params["base_value"]
            variation = params["variation"]
            noise = params["noise_level"]
            
            pressure = (
                base_value +
                variation * np.sin(2 * np.pi * 0.01 * t) +  # 0.01Hz variation
                np.random.randn() * noise
            )
            
            # Add occasional pressure drop
            if np.random.random() < 0.0003:  # 0.03% chance
                pressure -= np.random.uniform(500, 1000)
            
            return max(0, pressure)  # Pressure can't be negative
            
        else:
            # Generic simulated sensor
            return params["base_value"] + np.random.randn() * params["noise_level"]
    
    def read_sensor(self, sensor_name: str) -> float:
        """Read data from a specific sensor"""
        
        if sensor_name not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        sensor = self.sensors[sensor_name]
        
        if not sensor['initialized']:
            return 0.0
        
        # Read based on sensor type
        if sensor['type'] == 'i2c':
            return self.read_i2c_sensor(sensor_name)
        elif sensor['type'] == 'spi':
            return self.read_spi_sensor(sensor_name)
        elif sensor['type'] == 'serial':
            return self.read_serial_sensor(sensor_name)
        elif sensor['type'] == 'audio':
            return self.read_audio_sensor(sensor_name)
        elif sensor['type'] == 'simulated':
            return self.read_simulated_sensor(sensor_name)
        else:
            return 0.0
    
    def _sensor_reading_thread(self, sensor_name: str):
        """Thread for continuous sensor reading"""
        
        sampling_rate = self.config["sampling_rates"][sensor_name]
        interval = 1.0 / sampling_rate
        
        buffer = self.data_buffers[sensor_name]
        
        while self.running:
            try:
                start_time = time.perf_counter()
                
                # Read sensor
                value = self.read_sensor(sensor_name)
                timestamp = time.time()
                
                # Apply calibration
                calibration = self.config["calibration"][sensor_name]
                calibrated_value = value * calibration["scale"] + calibration["offset"]
                
                # Store in buffer
                buffer['data'][buffer['index']] = calibrated_value
                buffer['timestamps'][buffer['index']] = timestamp
                buffer['index'] = (buffer['index'] + 1) % buffer['size']
                
                # Put in raw data queue
                self.raw_data_queue.put_nowait({
                    'sensor': sensor_name,
                    'value': calibrated_value,
                    'timestamp': timestamp,
                    'raw_value': value
                })
                
                # Calculate sleep time to maintain sampling rate
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error in sensor thread {sensor_name}: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def _data_processing_thread(self):
        """Thread for processing sensor data"""
        
        while self.running:
            try:
                # Get raw data from queue
                raw_data = self.raw_data_queue.get(timeout=1.0)
                
                # Process data (feature extraction, filtering, etc.)
                processed_data = self._process_sensor_data(raw_data)
                
                # Put in processed queue
                self.processed_data_queue.put_nowait(processed_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in data processing thread: {e}")
    
    def _process_sensor_data(self, raw_data: Dict) -> Dict:
        """Process raw sensor data"""
        
        sensor_name = raw_data['sensor']
        value = raw_data['value']
        timestamp = raw_data['timestamp']
        
        # Get buffer for this sensor
        buffer = self.data_buffers[sensor_name]
        
        # Calculate features
        recent_data = buffer['data'][:buffer['index']]
        
        if len(recent_data) > 0:
            features = {
                'value': value,
                'mean': np.mean(recent_data),
                'std': np.std(recent_data),
                'max': np.max(recent_data),
                'min': np.min(recent_data),
                'rms': np.sqrt(np.mean(recent_data**2)),
                'peak_to_peak': np.max(recent_data) - np.min(recent_data)
            }
        else:
            features = {
                'value': value,
                'mean': value,
                'std': 0.0,
                'max': value,
                'min': value,
                'rms': value,
                'peak_to_peak': 0.0
            }
        
        # Check for anomalies
        anomalies = self._detect_anomalies(sensor_name, features)
        
        return {
            'sensor': sensor_name,
            'timestamp': timestamp,
            'features': features,
            'anomalies': anomalies,
            'raw_value': raw_data['raw_value']
        }
    
    def _detect_anomalies(self, sensor_name: str, features: Dict) -> List[str]:
        """Detect anomalies in sensor data"""
        
        anomalies = []
        
        # Define thresholds for different sensors
        thresholds = {
            'vibration': {'max': 1.0, 'std': 0.2},
            'thermal': {'max': 700, 'std': 50},
            'acoustic': {'max': 95, 'std': 10},
            'pressure': {'max': 3500, 'min': 2500, 'std': 200}
        }
        
        sensor_thresholds = thresholds.get(sensor_name, {})
        
        # Check each threshold
        if 'max' in sensor_thresholds and features['max'] > sensor_thresholds['max']:
            anomalies.append(f"Maximum value {features['max']:.2f} exceeds threshold {sensor_thresholds['max']}")
        
        if 'min' in sensor_thresholds and features['min'] < sensor_thresholds['min']:
            anomalies.append(f"Minimum value {features['min']:.2f} below threshold {sensor_thresholds['min']}")
        
        if 'std' in sensor_thresholds and features['std'] > sensor_thresholds['std']:
            anomalies.append(f"High variability: std={features['std']:.2f}")
        
        # Check for sudden changes
        if features['peak_to_peak'] > 2 * features['std']:
            anomalies.append(f"Large peak-to-peak variation: {features['peak_to_peak']:.2f}")
        
        return anomalies
    
    def start(self):
        """Start all sensor threads"""
        
        if self.running:
            print("Sensor interface already running")
            return
        
        print("Starting sensor interface...")
        self.running = True
        
        # Start sensor reading threads
        for sensor_name in self.sensors.keys():
            thread = threading.Thread(
                target=self._sensor_reading_thread,
                args=(sensor_name,),
                name=f"sensor_{sensor_name}"
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            print(f"  Started {sensor_name} sensor thread")
        
        # Start data processing thread
        processing_thread = threading.Thread(
            target=self._data_processing_thread,
            name="data_processing"
        )
        processing_thread.daemon = True
        processing_thread.start()
        self.threads.append(processing_thread)
        
        print(f"✅ Sensor interface started with {len(self.sensors)} sensors")
    
    def stop(self):
        """Stop all sensor threads"""
        
        print("Stopping sensor interface...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        self.threads = []
        print("✅ Sensor interface stopped")
    
    def get_latest_data(self, sensor_name: Optional[str] = None) -> Dict:
        """Get latest processed data"""
        
        if sensor_name:
            # Get specific sensor data
            buffer = self.data_buffers.get(sensor_name)
            if buffer and buffer['index'] > 0:
                latest_index = (buffer['index'] - 1) % buffer['size']
                return {
                    'sensor': sensor_name,
                    'value': buffer['data'][latest_index],
                    'timestamp': buffer['timestamps'][latest_index]
                }
            else:
                return {}
        else:
            # Get data for all sensors
            all_data = {}
            for name, buffer in self.data_buffers.items():
                if buffer['index'] > 0:
                    latest_index = (buffer['index'] - 1) % buffer['size']
                    all_data[name] = {
                        'value': buffer['data'][latest_index],
                        'timestamp': buffer['timestamps'][latest_index]
                    }
            return all_data
    
    def get_processed_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get processed data from queue"""
        try:
            return self.processed_data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_sensor_status(self) -> Dict:
        """Get status of all sensors"""
        
        status = {}
        for name, sensor in self.sensors.items():
            buffer = self.data_buffers[name]
            status[name] = {
                'type': sensor['type'],
                'initialized': sensor['initialized'],
                'sampling_rate': self.config["sampling_rates"].get(name, 0),
                'buffer_fill': buffer['index'],
                'buffer_size': buffer['size'],
                'latest_value': self.get_latest_data(name).get('value', 0.0)
            }
        
        return status
    
    def calibrate_sensor(self, sensor_name: str, reference_values: List[float]):
        """Calibrate sensor using reference values"""
        
        if sensor_name not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        # Collect sensor readings
        readings = []
        for _ in range(len(reference_values)):
            readings.append(self.read_sensor(sensor_name))
            time.sleep(0.1)
        
        # Calculate calibration parameters (simplified linear calibration)
        sensor_mean = np.mean(readings)
        reference_mean = np.mean(reference_values)
        
        scale = reference_mean / sensor_mean if sensor_mean != 0 else 1.0
        offset = reference_mean - sensor_mean * scale
        
        # Update calibration
        self.config["calibration"][sensor_name] = {
            "scale": float(scale),
            "offset": float(offset)
        }
        
        print(f"Calibrated {sensor_name}: scale={scale:.4f}, offset={offset:.4f}")
        
        return scale, offset


# Test function
def test_sensor_interface():
    """Test the sensor interface"""
    
    print("Testing Sensor Interface...")
    
    # Create sensor interface
    interface = SensorInterface()
    
    # Test single sensor reading
    print("\nTesting single sensor readings:")
    for sensor_name in ['vibration', 'thermal', 'acoustic', 'pressure']:
        value = interface.read_sensor(sensor_name)
        print(f"  {sensor_name}: {value:.2f}")
    
    # Test starting interface
    print("\nStarting sensor interface...")
    interface.start()
    
    # Collect some data
    print("\nCollecting sensor data for 5 seconds...")
    collected_data = []
    start_time = time.time()
    
    while time.time() - start_time < 5:
        processed_data = interface.get_processed_data(timeout=0.1)
        if processed_data:
            collected_data.append(processed_data)
            print(f"  {processed_data['sensor']}: "
                  f"{processed_data['features']['value']:.2f} "
                  f"(anomalies: {len(processed_data['anomalies'])})")
    
    # Get sensor status
    print("\nSensor Status:")
    status = interface.get_sensor_status()
    for sensor_name, sensor_status in status.items():
        print(f"  {sensor_name}: {sensor_status['type']}, "
              f"rate={sensor_status['sampling_rate']}Hz, "
              f"fill={sensor_status['buffer_fill']}/{sensor_status['buffer_size']}")
    
    # Get latest data
    print("\nLatest Sensor Values:")
    latest_data = interface.get_latest_data()
    for sensor_name, data in latest_data.items():
        print(f"  {sensor_name}: {data['value']:.2f}")
    
    # Test calibration
    print("\nTesting calibration...")
    try:
        # Simulate calibration with known values
        reference_values = [0.3, 0.31, 0.29, 0.32]
        scale, offset = interface.calibrate_sensor('vibration', reference_values)
        print(f"  Calibration successful: scale={scale:.4f}, offset={offset:.4f}")
    except Exception as e:
        print(f"  Calibration failed: {e}")
    
    # Stop interface
    print("\nStopping sensor interface...")
    interface.stop()
    
    # Statistics
    print(f"\nData Collection Statistics:")
    print(f"  Total samples collected: {len(collected_data)}")
    
    if collected_data:
        sensors = {}
        for data in collected_data:
            sensor = data['sensor']
            if sensor not in sensors:
                sensors[sensor] = []
            sensors[sensor].append(data)
        
        for sensor, data_list in sensors.items():
            print(f"  {sensor}: {len(data_list)} samples")
    
    print("\n✅ Sensor interface test completed!")
    return interface


if __name__ == "__main__":
    # Run test
    interface = test_sensor_interface()
    
    # Save test data
    import json
    test_results = {
        'sensor_status': interface.get_sensor_status(),
        'config': interface.config
    }
    
    with open("sensor_interface_test.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to sensor_interface_test.json")