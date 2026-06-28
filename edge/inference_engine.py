"""
<10ms Inference Engine for Edge Deployment
Real-time multi-modal fault detection optimized for edge devices
"""

import time
import numpy as np
import threading
import queue
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import edge-optimized libraries
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class InferenceMode(Enum):
    """Inference modes for different hardware"""
    TFLITE = "tflite"
    TENSORRT = "tensorrt"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"

class InferenceResult:
    """Container for inference results"""
    
    def __init__(self):
        self.timestamp = time.time()
        self.fault_predictions = []
        self.severity_score = 0.0
        self.confidence = 0.0
        self.inference_time_ms = 0.0
        self.model_used = ""
        self.sensor_readings = {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'fault_predictions': self.fault_predictions,
            'severity_score': self.severity_score,
            'confidence': self.confidence,
            'inference_time_ms': self.inference_time_ms,
            'model_used': self.model_used,
            'meets_10ms_target': self.inference_time_ms < 10
        }
    
    def __str__(self) -> str:
        return (f"InferenceResult(time={self.inference_time_ms:.2f}ms, "
                f"faults={len(self.fault_predictions)}, "
                f"severity={self.severity_score:.2f})")

class EdgeInferenceEngine:
    """<10ms inference engine for edge deployment"""
    
    def __init__(self, 
                 model_path: str = "models/optimized",
                 inference_mode: InferenceMode = InferenceMode.TFLITE,
                 target_device: str = "auto"):
        
        self.model_path = model_path
        self.inference_mode = inference_mode
        self.target_device = target_device
        self.model = None
        self.interpreter = None
        self.engine = None
        self.context = None
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        self.avg_inference_time = 0.0
        
        # Real-time constraints
        self.max_inference_time_ms = 10.0  # <10ms target
        self.warning_threshold_ms = 8.0
        
        # Thread safety
        self.lock = threading.Lock()
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue(maxsize=1000)
        
        # Fault thresholds
        self.fault_thresholds = {
            'bearing_wear': 0.7,
            'blade_damage': 0.8,
            'oil_leak': 0.6,
            'compressor_stall': 0.9,
            'ignition_failure': 0.85
        }
        
        # Initialize based on mode
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize inference engine based on mode"""
        
        print(f"Initializing {self.inference_mode.value} inference engine...")
        
        if self.inference_mode == InferenceMode.TFLITE:
            self._initialize_tflite()
        elif self.inference_mode == InferenceMode.TENSORRT:
            self._initialize_tensorrt()
        elif self.inference_mode == InferenceMode.TENSORFLOW:
            self._initialize_tensorflow()
        elif self.inference_mode == InferenceMode.ONNX:
            self._initialize_onnx()
        else:
            raise ValueError(f"Unsupported inference mode: {self.inference_mode}")
        
        print(f"✅ Inference engine initialized")
        print(f"   Target: <{self.max_inference_time_ms}ms inference time")
        
    def _initialize_tflite(self):
        """Initialize TFLite interpreter"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TFLite Runtime not available")
        
        # Find TFLite model
        model_files = [f for f in os.listdir(self.model_path) 
                      if f.endswith('.tflite')]
        
        if not model_files:
            raise FileNotFoundError(f"No TFLite models found in {self.model_path}")
        
        # Load the smallest model (likely most optimized)
        model_files.sort(key=lambda x: os.path.getsize(os.path.join(self.model_path, x)))
        model_file = os.path.join(self.model_path, model_files[0])
        
        print(f"Loading TFLite model: {model_file}")
        
        # Create interpreter
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        
        print(f"   Inputs: {len(self.input_details)}")
        print(f"   Outputs: {len(self.output_details)}")
        
        # Warm up
        self._warmup_tflite()
        
    def _warmup_tflite(self):
        """Warm up TFLite interpreter"""
        print("Warming up TFLite interpreter...")
        
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        for _ in range(10):
            # Set inputs
            for i, detail in enumerate(self.input_details):
                self.interpreter.set_tensor(detail['index'], dummy_input[i])
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs (discard)
            _ = [self.interpreter.get_tensor(detail['index']) 
                 for detail in self.output_details]
        
        print("✅ TFLite warmup complete")
    
    def _initialize_tensorrt(self):
        """Initialize TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        # Find TensorRT engine
        engine_files = [f for f in os.listdir(self.model_path) 
                       if f.endswith('.trt')]
        
        if not engine_files:
            raise FileNotFoundError(f"No TensorRT engines found in {self.model_path}")
        
        engine_file = os.path.join(self.model_path, engine_files[0])
        
        print(f"Loading TensorRT engine: {engine_file}")
        
        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_file, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        print(f"   Engine loaded: {self.engine.num_layers} layers")
        print(f"   Max batch size: {self.engine.max_batch_size}")
        
        # Warm up
        self._warmup_tensorrt()
    
    def _warmup_tensorrt(self):
        """Warm up TensorRT engine"""
        print("Warming up TensorRT engine...")
        
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        # Prepare input data
        input_data = []
        for data in dummy_input:
            input_data.append(data.flatten())
        
        # Copy to input buffers
        for i, data in enumerate(input_data):
            np.copyto(self.inputs[i]['host'], data)
        
        for _ in range(10):
            # Copy inputs to device
            for i in range(len(self.inputs)):
                cuda.memcpy_htod_async(self.inputs[i]['device'], 
                                      self.inputs[i]['host'], 
                                      self.stream)
            
            # Execute
            self.context.execute_async_v2(bindings=self.bindings, 
                                         stream_handle=self.stream.handle)
            
            # Copy outputs from device
            for i in range(len(self.outputs)):
                cuda.memcpy_dtoh_async(self.outputs[i]['host'], 
                                      self.outputs[i]['device'], 
                                      self.stream)
            
            self.stream.synchronize()
        
        print("✅ TensorRT warmup complete")
    
    def _initialize_tensorflow(self):
        """Initialize TensorFlow model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Find TensorFlow model
        model_files = [f for f in os.listdir(self.model_path) 
                      if f.endswith(('.h5', '.keras'))]
        
        if not model_files:
            raise FileNotFoundError(f"No TensorFlow models found in {self.model_path}")
        
        model_file = os.path.join(self.model_path, model_files[0])
        
        print(f"Loading TensorFlow model: {model_file}")
        
        # Load model
        self.model = tf.keras.models.load_model(model_file)
        
        # Optimize for inference
        self.model._make_predict_function()
        
        print(f"   Model loaded: {self.model.name}")
        print(f"   Inputs: {len(self.model.inputs)}")
        print(f"   Outputs: {len(self.model.outputs)}")
        
        # Warm up
        self._warmup_tensorflow()
    
    def _warmup_tensorflow(self):
        """Warm up TensorFlow model"""
        print("Warming up TensorFlow model...")
        
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=0)
        
        print("✅ TensorFlow warmup complete")
    
    def _initialize_onnx(self):
        """Initialize ONNX Runtime"""
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("ONNX Runtime not available")
        
        # Find ONNX model
        model_files = [f for f in os.listdir(self.model_path) 
                      if f.endswith('.onnx')]
        
        if not model_files:
            raise FileNotFoundError(f"No ONNX models found in {self.model_path}")
        
        model_file = os.path.join(self.model_path, model_files[0])
        
        print(f"Loading ONNX model: {model_file}")
        
        # Create inference session
        providers = ['CPUExecutionProvider']
        
        # Try CUDA if available
        try:
            import onnxruntime_gpu
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except ImportError:
            pass
        
        self.session = ort.InferenceSession(model_file, providers=providers)
        
        # Get input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"   Inputs: {self.input_names}")
        print(f"   Outputs: {self.output_names}")
        
        # Warm up
        self._warmup_onnx()
    
    def _warmup_onnx(self):
        """Warm up ONNX Runtime"""
        print("Warming up ONNX Runtime...")
        
        # Create dummy input
        dummy_input = self._create_dummy_input()
        
        # Prepare input dict
        input_dict = {}
        for i, name in enumerate(self.input_names):
            input_dict[name] = dummy_input[i].astype(np.float32)
        
        for _ in range(10):
            _ = self.session.run(self.output_names, input_dict)
        
        print("✅ ONNX warmup complete")
    
    def _create_dummy_input(self) -> List[np.ndarray]:
        """Create dummy input for warmup"""
        return [
            np.random.randn(1, 1000, 1).astype(np.float32),  # vibration
            np.random.randn(1, 500, 1).astype(np.float32),   # thermal
            np.random.randn(1, 250, 1).astype(np.float32),   # acoustic
            np.random.randn(1, 100, 1).astype(np.float32)    # pressure
        ]
    
    def preprocess_sensor_data(self, sensor_data: Dict) -> List[np.ndarray]:
        """Preprocess sensor data for inference"""
        
        # Extract and normalize sensor data
        vibration = sensor_data.get('vibration', 0.0)
        thermal = sensor_data.get('thermal', 25.0)
        acoustic = sensor_data.get('acoustic', 70.0)
        pressure = sensor_data.get('pressure', 100.0)
        
        # Normalize based on expected ranges
        vibration_norm = np.clip(vibration / 0.5, 0, 2)  # 0-0.5g normal range
        thermal_norm = np.clip((thermal - 25) / 625, 0, 1)  # 25-650°C
        acoustic_norm = np.clip((acoustic - 60) / 35, 0, 1)  # 60-95dB
        pressure_norm = np.clip((pressure - 50) / 2950, 0, 1)  # 50-3000 PSI
        
        # Create time series data (replicate value across time steps)
        vibration_series = np.full((1, 1000, 1), vibration_norm, dtype=np.float32)
        thermal_series = np.full((1, 500, 1), thermal_norm, dtype=np.float32)
        acoustic_series = np.full((1, 250, 1), acoustic_norm, dtype=np.float32)
        pressure_series = np.full((1, 100, 1), pressure_norm, dtype=np.float32)
        
        return [vibration_series, thermal_series, acoustic_series, pressure_series]
    
    def infer_tflite(self, preprocessed_data: List[np.ndarray]) -> InferenceResult:
        """Run inference using TFLite"""
        
        result = InferenceResult()
        result.model_used = "TFLite"
        
        start_time = time.perf_counter()
        
        try:
            # Set inputs
            for i, detail in enumerate(self.input_details):
                self.interpreter.set_tensor(detail['index'], preprocessed_data[i])
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            outputs = []
            for detail in self.output_details:
                outputs.append(self.interpreter.get_tensor(detail['index']))
            
            # Process results
            fault_probs = outputs[0][0]  # First output: fault probabilities
            severity = outputs[1][0][0] if len(outputs) > 1 else 0.0  # Second output: severity
            
            # Apply thresholds
            for i, prob in enumerate(fault_probs):
                if prob > 0.5:  # Generic threshold
                    fault_name = f"fault_{i}"
                    result.fault_predictions.append({
                        'name': fault_name,
                        'confidence': float(prob),
                        'severity': 'high' if prob > 0.8 else 'medium'
                    })
            
            result.severity_score = float(severity)
            result.confidence = float(np.max(fault_probs))
            
        except Exception as e:
            print(f"TFLite inference error: {e}")
        
        result.inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def infer_tensorrt(self, preprocessed_data: List[np.ndarray]) -> InferenceResult:
        """Run inference using TensorRT"""
        
        result = InferenceResult()
        result.model_used = "TensorRT"
        
        start_time = time.perf_counter()
        
        try:
            # Prepare input data
            input_data = []
            for data in preprocessed_data:
                input_data.append(data.flatten())
            
            # Copy to input buffers
            for i, data in enumerate(input_data):
                np.copyto(self.inputs[i]['host'], data)
            
            # Copy inputs to device
            for i in range(len(self.inputs)):
                cuda.memcpy_htod_async(self.inputs[i]['device'], 
                                      self.inputs[i]['host'], 
                                      self.stream)
            
            # Execute
            self.context.execute_async_v2(bindings=self.bindings, 
                                         stream_handle=self.stream.handle)
            
            # Copy outputs from device
            for i in range(len(self.outputs)):
                cuda.memcpy_dtoh_async(self.outputs[i]['host'], 
                                      self.outputs[i]['device'], 
                                      self.stream)
            
            self.stream.synchronize()
            
            # Process outputs
            # Note: Output processing depends on model architecture
            # This is a simplified version
            fault_probs = self.outputs[0]['host'].reshape(1, -1)[0]
            
            # Apply thresholds
            for i, prob in enumerate(fault_probs):
                if prob > 0.5:
                    fault_name = f"fault_{i}"
                    result.fault_predictions.append({
                        'name': fault_name,
                        'confidence': float(prob),
                        'severity': 'high' if prob > 0.8 else 'medium'
                    })
            
            result.severity_score = float(np.mean(fault_probs))
            result.confidence = float(np.max(fault_probs))
            
        except Exception as e:
            print(f"TensorRT inference error: {e}")
        
        result.inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def infer_tensorflow(self, preprocessed_data: List[np.ndarray]) -> InferenceResult:
        """Run inference using TensorFlow"""
        
        result = InferenceResult()
        result.model_used = "TensorFlow"
        
        start_time = time.perf_counter()
        
        try:
            # Run inference
            predictions = self.model.predict(preprocessed_data, verbose=0)
            
            # Process results
            if isinstance(predictions, list):
                fault_probs = predictions[0][0]  # First output: fault probabilities
                severity = predictions[1][0][0] if len(predictions) > 1 else 0.0
            else:
                fault_probs = predictions[0]
                severity = 0.0
            
            # Apply thresholds
            for i, prob in enumerate(fault_probs):
                if prob > 0.5:
                    fault_name = f"fault_{i}"
                    result.fault_predictions.append({
                        'name': fault_name,
                        'confidence': float(prob),
                        'severity': 'high' if prob > 0.8 else 'medium'
                    })
            
            result.severity_score = float(severity)
            result.confidence = float(np.max(fault_probs))
            
        except Exception as e:
            print(f"TensorFlow inference error: {e}")
        
        result.inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def infer_onnx(self, preprocessed_data: List[np.ndarray]) -> InferenceResult:
        """Run inference using ONNX Runtime"""
        
        result = InferenceResult()
        result.model_used = "ONNX"
        
        start_time = time.perf_counter()
        
        try:
            # Prepare input dict
            input_dict = {}
            for i, name in enumerate(self.input_names):
                input_dict[name] = preprocessed_data[i].astype(np.float32)
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            
            # Process results
            fault_probs = outputs[0][0]  # First output: fault probabilities
            severity = outputs[1][0][0] if len(outputs) > 1 else 0.0
            
            # Apply thresholds
            for i, prob in enumerate(fault_probs):
                if prob > 0.5:
                    fault_name = f"fault_{i}"
                    result.fault_predictions.append({
                        'name': fault_name,
                        'confidence': float(prob),
                        'severity': 'high' if prob > 0.8 else 'medium'
                    })
            
            result.severity_score = float(severity)
            result.confidence = float(np.max(fault_probs))
            
        except Exception as e:
            print(f"ONNX inference error: {e}")
        
        result.inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def infer(self, sensor_data: Dict) -> InferenceResult:
        """Main inference method"""
        
        with self.lock:
            # Preprocess data
            preprocessed_data = self.preprocess_sensor_data(sensor_data)
            
            # Run inference based on mode
            if self.inference_mode == InferenceMode.TFLITE:
                result = self.infer_tflite(preprocessed_data)
            elif self.inference_mode == InferenceMode.TENSORRT:
                result = self.infer_tensorrt(preprocessed_data)
            elif self.inference_mode == InferenceMode.TENSORFLOW:
                result = self.infer_tensorflow(preprocessed_data)
            elif self.inference_mode == InferenceMode.ONNX:
                result = self.infer_onnx(preprocessed_data)
            else:
                raise ValueError(f"Unsupported inference mode: {self.inference_mode}")
            
            # Store sensor readings
            result.sensor_readings = sensor_data
            
            # Update performance metrics
            self.total_inferences += 1
            self.inference_times.append(result.inference_time_ms)
            
            # Keep only last 1000 measurements
            if len(self.inference_times) > 1000:
                self.inference_times.pop(0)
            
            # Update average
            self.avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
            
            # Check performance
            self._check_performance(result)
            
            return result
    
    def _check_performance(self, result: InferenceResult):
        """Check if performance meets requirements"""
        
        if result.inference_time_ms > self.max_inference_time_ms:
            print(f"⚠️  Inference time {result.inference_time_ms:.2f}ms exceeds {self.max_inference_time_ms}ms target")
        
        elif result.inference_time_ms > self.warning_threshold_ms:
            print(f"⚠️  Inference time {result.inference_time_ms:.2f}ms approaching limit")
        
        # Log slow inferences (for debugging)
        if result.inference_time_ms > 20:
            print(f"❌ Very slow inference: {result.inference_time_ms:.2f}ms")
    
    def start_realtime_loop(self, sensor_callback, interval_ms: float = 100):
        """Start real-time inference loop"""
        
        self.running = True
        self.inference_thread = threading.Thread(
            target=self._realtime_loop,
            args=(sensor_callback, interval_ms)
        )
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
        print(f"Started real-time inference loop ({interval_ms}ms interval)")
    
    def _realtime_loop(self, sensor_callback, interval_ms: float):
        """Real-time inference loop"""
        
        interval_seconds = interval_ms / 1000.0
        
        while self.running:
            try:
                # Get sensor data from callback
                sensor_data = sensor_callback()
                
                if sensor_data:
                    # Run inference
                    result = self.infer(sensor_data)
                    
                    # Put result in output queue
                    try:
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        # Remove oldest if queue is full
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
                # Sleep to maintain interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in real-time loop: {e}")
                time.sleep(1)  # Prevent tight loop on error
    
    def stop_realtime_loop(self):
        """Stop real-time inference loop"""
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        print("Stopped real-time inference loop")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        
        if not self.inference_times:
            return {
                'avg_inference_time_ms': 0,
                'min_inference_time_ms': 0,
                'max_inference_time_ms': 0,
                'throughput_hz': 0,
                'total_inferences': self.total_inferences,
                'meets_10ms_target': False
            }
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'std_inference_time_ms': np.std(self.inference_times),
            'throughput_hz': 1000 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0,
            'total_inferences': self.total_inferences,
            'meets_10ms_target': np.mean(self.inference_times) < 10,
            'inference_mode': self.inference_mode.value
        }
    
    def benchmark(self, num_iterations: int = 1000) -> Dict:
        """Run benchmark tests"""
        
        print(f"Running benchmark ({num_iterations} iterations)...")
        
        # Create test sensor data
        test_data = {
            'vibration': 0.3,
            'thermal': 400.0,
            'acoustic': 75.0,
            'pressure': 3000.0
        }
        
        times = []
        results = []
        
        # Warm up
        for _ in range(10):
            _ = self.infer(test_data)
        
        # Run benchmark
        for i in range(num_iterations):
            start_time = time.perf_counter()
            result = self.infer(test_data)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        # Fault detection statistics
        faults_detected = sum(1 for r in results if r.fault_predictions)
        avg_confidence = np.mean([r.confidence for r in results])
        
        metrics = {
            'iterations': num_iterations,
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'std_inference_time_ms': std_time,
            'throughput_hz': 1000 / avg_time if avg_time > 0 else 0,
            'faults_detected': faults_detected,
            'detection_rate': faults_detected / num_iterations,
            'avg_confidence': avg_confidence,
            'meets_10ms_target': avg_time < 10,
            'inference_mode': self.inference_mode.value
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Avg Inference Time: {avg_time:.2f} ms")
        print(f"  Min Inference Time: {min_time:.2f} ms")
        print(f"  Max Inference Time: {max_time:.2f} ms")
        print(f"  Throughput: {1000/avg_time:.1f} Hz")
        print(f"  Fault Detection Rate: {metrics['detection_rate']:.2%}")
        print(f"  Meets 10ms Target: {'✅' if avg_time < 10 else '❌'}")
        
        return metrics


# Test function
def test_inference_engine():
    """Test the inference engine"""
    
    print("Testing Edge Inference Engine...")
    
    # Create test directory
    os.makedirs("models/optimized", exist_ok=True)
    
    # Try different inference modes
    modes_to_test = []
    
    if TFLITE_AVAILABLE:
        modes_to_test.append(InferenceMode.TFLITE)
    
    if TENSORRT_AVAILABLE:
        modes_to_test.append(InferenceMode.TENSORRT)
    
    if TF_AVAILABLE:
        modes_to_test.append(InferenceMode.TENSORFLOW)
    
    if not modes_to_test:
        print("❌ No inference backends available")
        return
    
    results = {}
    
    for mode in modes_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {mode.value.upper()} mode")
        print('='*50)
        
        try:
            # Create engine
            engine = EdgeInferenceEngine(
                model_path="models/optimized",
                inference_mode=mode,
                target_device="test"
            )
            
            # Run benchmark
            metrics = engine.benchmark(num_iterations=100)
            
            results[mode.value] = metrics
            
            # Test real-time inference
            print("\nTesting real-time inference...")
            
            # Mock sensor callback
            def mock_sensor_callback():
                return {
                    'vibration': 0.25 + np.random.randn() * 0.05,
                    'thermal': 400 + np.random.randn() * 10,
                    'acoustic': 75 + np.random.randn() * 5,
                    'pressure': 3000 + np.random.randn() * 100
                }
            
            # Start real-time loop
            engine.start_realtime_loop(mock_sensor_callback, interval_ms=200)
            
            # Collect some results
            collected_results = []
            for _ in range(10):
                try:
                    result = engine.output_queue.get(timeout=1.0)
                    collected_results.append(result)
                    print(f"  Inference: {result.inference_time_ms:.2f}ms, "
                          f"Faults: {len(result.fault_predictions)}")
                except queue.Empty:
                    break
            
            # Stop loop
            engine.stop_realtime_loop()
            
            # Get performance metrics
            perf_metrics = engine.get_performance_metrics()
            print(f"\nPerformance Metrics:")
            print(f"  Avg Time: {perf_metrics['avg_inference_time_ms']:.2f}ms")
            print(f"  Throughput: {perf_metrics['throughput_hz']:.1f}Hz")
            print(f"  Meets Target: {'✅' if perf_metrics['meets_10ms_target'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Error testing {mode.value}: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for mode, metrics in results.items():
        meets_target = '✅' if metrics['meets_10ms_target'] else '❌'
        print(f"{mode.upper():12} | "
              f"Avg: {metrics['avg_inference_time_ms']:6.2f}ms | "
              f"Throughput: {metrics['throughput_hz']:6.1f}Hz | "
              f"Target: {meets_target}")
    
    print('='*60)
    
    # Check if any mode meets target
    any_meets_target = any(m['meets_10ms_target'] for m in results.values())
    
    if any_meets_target:
        print("✅ SUCCESS: At least one inference mode meets <10ms target!")
    else:
        print("❌ FAILURE: No inference mode meets <10ms target")
    
    return results


if __name__ == "__main__":
    # Run test
    test_results = test_inference_engine()
    
    # Save results
    with open("inference_benchmark.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("\n✅ Inference engine test completed!")