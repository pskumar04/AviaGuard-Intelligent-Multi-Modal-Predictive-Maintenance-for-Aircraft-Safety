"""
Edge Model Optimizer for TensorRT/TFLite Deployment
Optimizes models for <10ms inference on edge devices
"""

import tensorflow as tf
import numpy as np
import os
import json
import time
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EdgeModelOptimizer:
    """Optimizes models for edge deployment with TensorRT and TFLite"""
    
    def __init__(self, model_path: str, target_device: str = "JETSON_TX2"):
        self.model_path = model_path
        self.target_device = target_device
        
        # Device-specific optimization parameters
        self.optimization_params = {
            "JETSON_NANO": {
                "precision": "FP16",
                "max_workspace_size": 1 << 30,  # 1GB
                "min_batch_size": 1,
                "max_batch_size": 8,
                "optimization_level": 3
            },
            "JETSON_TX2": {
                "precision": "FP16",
                "max_workspace_size": 2 << 30,  # 2GB
                "min_batch_size": 1,
                "max_batch_size": 16,
                "optimization_level": 3
            },
            "JETSON_XAVIER": {
                "precision": "FP16",
                "max_workspace_size": 4 << 30,  # 4GB
                "min_batch_size": 1,
                "max_batch_size": 32,
                "optimization_level": 4
            },
            "RASPBERRY_PI": {
                "precision": "INT8",
                "max_workspace_size": 256 << 20,  # 256MB
                "min_batch_size": 1,
                "max_batch_size": 4,
                "optimization_level": 2
            },
            "CPU_ONLY": {
                "precision": "FP32",
                "max_workspace_size": 512 << 20,  # 512MB
                "min_batch_size": 1,
                "max_batch_size": 32,
                "optimization_level": 1
            }
        }
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        try:
            model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def convert_to_tflite(self, 
                         output_path: str = "models/optimized_model.tflite",
                         quantization: str = "float16") -> bool:
        """Convert model to TFLite format"""
        
        if self.model is None:
            print("❌ Model not loaded")
            return False
        
        print(f"Converting model to TFLite ({quantization})...")
        
        try:
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Set optimization parameters
            if quantization == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif quantization == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._get_representative_dataset()
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif quantization == "dynamic_range":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            else:  # float32
                pass  # No optimization
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model size
            model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f"✅ TFLite model saved to {output_path}")
            print(f"   Model size: {model_size:.2f} MB")
            print(f"   Quantization: {quantization}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error converting to TFLite: {e}")
            return False
    
    def _get_representative_dataset(self):
        """Create representative dataset for quantization"""
        
        def representative_data_gen():
            # Generate representative data for calibration
            for _ in range(100):
                # Create dummy input data matching model's input signature
                vibration = np.random.randn(1, 1000, 1).astype(np.float32)
                thermal = np.random.randn(1, 500, 1).astype(np.float32)
                acoustic = np.random.randn(1, 250, 1).astype(np.float32)
                pressure = np.random.randn(1, 100, 1).astype(np.float32)
                
                yield [vibration, thermal, acoustic, pressure]
        
        return representative_data_gen
    
    def convert_to_tensorrt(self,
                           output_path: str = "models/optimized_model.trt",
                           precision: str = "FP16") -> bool:
        """Convert model to TensorRT format"""
        
        print(f"Converting model to TensorRT ({precision})...")
        
        try:
            # Check if TensorRT is available
            import tensorrt as trt
            
            # Create TensorRT converter
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Build engine
            engine = self._build_tensorrt_engine(TRT_LOGGER, precision)
            
            if engine is None:
                print("❌ Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"✅ TensorRT engine saved to {output_path}")
            
            # Save engine metadata
            self._save_tensorrt_metadata(output_path, precision)
            
            return True
            
        except ImportError:
            print("❌ TensorRT not installed. Skipping TensorRT conversion.")
            return False
        except Exception as e:
            print(f"❌ Error converting to TensorRT: {e}")
            return False
    
    def _build_tensorrt_engine(self, logger, precision: str):
        """Build TensorRT engine"""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # First convert to ONNX
        onnx_path = self.model_path.replace('.h5', '.onnx')
        if not os.path.exists(onnx_path):
            self.convert_to_onnx(onnx_path)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('❌ Failed to parse ONNX model')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build configuration
        config = builder.create_builder_config()
        
        # Set precision
        if precision == "FP16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "INT8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Need calibration dataset for INT8
            # config.int8_calibrator = YourCalibrator()
        
        # Set workspace size
        device_params = self.optimization_params.get(self.target_device, 
                                                    self.optimization_params["CPU_ONLY"])
        config.max_workspace_size = device_params["max_workspace_size"]
        
        # Set optimization level
        config.builder_optimization_level = device_params["optimization_level"]
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        return engine
    
    def convert_to_onnx(self, output_path: str = "models/model.onnx") -> bool:
        """Convert model to ONNX format"""
        
        print("Converting model to ONNX...")
        
        try:
            import onnx
            import tf2onnx
            
            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=[
                    tf.TensorSpec((None, 1000, 1), tf.float32, name='vibration_input'),
                    tf.TensorSpec((None, 500, 1), tf.float32, name='thermal_input'),
                    tf.TensorSpec((None, 250, 1), tf.float32, name='acoustic_input'),
                    tf.TensorSpec((None, 100, 1), tf.float32, name='pressure_input')
                ],
                opset=13,
                output_path=output_path
            )
            
            print(f"✅ ONNX model saved to {output_path}")
            return True
            
        except ImportError:
            print("❌ tf2onnx not installed. Install with: pip install tf2onnx")
            return False
        except Exception as e:
            print(f"❌ Error converting to ONNX: {e}")
            return False
    
    def _save_tensorrt_metadata(self, engine_path: str, precision: str):
        """Save TensorRT engine metadata"""
        
        metadata = {
            "engine_path": engine_path,
            "precision": precision,
            "target_device": self.target_device,
            "optimization_params": self.optimization_params.get(self.target_device, {}),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_shapes": [layer.input_shape for layer in self.model.layers if hasattr(layer, 'input_shape')],
            "output_shapes": [layer.output_shape for layer in self.model.layers if hasattr(layer, 'output_shape')]
        }
        
        metadata_path = engine_path.replace('.trt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ TensorRT metadata saved to {metadata_path}")
    
    def benchmark_model(self, 
                       model_format: str = "tflite",
                       model_path: str = None,
                       num_iterations: int = 1000) -> Dict:
        """Benchmark model performance"""
        
        if model_path is None:
            if model_format == "tflite":
                model_path = "models/optimized_model.tflite"
            elif model_format == "tensorrt":
                model_path = "models/optimized_model.trt"
            else:
                model_path = self.model_path
        
        print(f"Benchmarking {model_format} model...")
        
        try:
            if model_format == "original":
                return self._benchmark_original_model(num_iterations)
            elif model_format == "tflite":
                return self._benchmark_tflite_model(model_path, num_iterations)
            elif model_format == "tensorrt":
                return self._benchmark_tensorrt_model(model_path, num_iterations)
            else:
                print(f"❌ Unknown model format: {model_format}")
                return {}
                
        except Exception as e:
            print(f"❌ Error benchmarking model: {e}")
            return {}
    
    def _benchmark_original_model(self, num_iterations: int) -> Dict:
        """Benchmark original TensorFlow model"""
        
        if self.model is None:
            return {}
        
        # Create dummy input
        dummy_input = [
            np.random.randn(1, 1000, 1).astype(np.float32),
            np.random.randn(1, 500, 1).astype(np.float32),
            np.random.randn(1, 250, 1).astype(np.float32),
            np.random.randn(1, 100, 1).astype(np.float32)
        ]
        
        # Warmup
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=0)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            _ = self.model.predict(dummy_input, verbose=0)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        return {
            "format": "original",
            "avg_inference_time_ms": avg_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "std_inference_time_ms": std_time,
            "throughput_samples_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "meets_10ms_target": avg_time < 10
        }
    
    def _benchmark_tflite_model(self, model_path: str, num_iterations: int) -> Dict:
        """Benchmark TFLite model"""
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        dummy_input = [
            np.random.randn(1, 1000, 1).astype(np.float32),
            np.random.randn(1, 500, 1).astype(np.float32),
            np.random.randn(1, 250, 1).astype(np.float32),
            np.random.randn(1, 100, 1).astype(np.float32)
        ]
        
        # Set inputs
        for i, detail in enumerate(input_details):
            interpreter.set_tensor(detail['index'], dummy_input[i])
        
        # Warmup
        for _ in range(10):
            interpreter.invoke()
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            interpreter.invoke()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Get outputs
        outputs = []
        for detail in output_details:
            outputs.append(interpreter.get_tensor(detail['index']))
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        return {
            "format": "tflite",
            "avg_inference_time_ms": avg_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "std_inference_time_ms": std_time,
            "throughput_samples_per_second": 1000 / avg_time if avg_time > 0 else 0,
            "meets_10ms_target": avg_time < 10,
            "output_shapes": [output.shape for output in outputs]
        }
    
    def _benchmark_tensorrt_model(self, model_path: str, num_iterations: int) -> Dict:
        """Benchmark TensorRT model"""
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Load TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with open(model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate buffers
            inputs, outputs, bindings = [], [], []
            stream = cuda.Stream()
            
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                bindings.append(int(device_mem))
                
                if engine.binding_is_input(binding):
                    inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem})
            
            # Create dummy input
            dummy_input = [
                np.random.randn(1, 1000, 1).astype(np.float32),
                np.random.randn(1, 500, 1).astype(np.float32),
                np.random.randn(1, 250, 1).astype(np.float32),
                np.random.randn(1, 100, 1).astype(np.float32)
            ]
            
            # Flatten and copy to input buffer
            input_data = np.concatenate([arr.flatten() for arr in dummy_input])
            np.copyto(inputs[0]['host'], input_data)
            
            # Warmup
            for _ in range(10):
                cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
                stream.synchronize()
            
            # Benchmark
            times = []
            for i in range(num_iterations):
                start_time = time.perf_counter()
                
                cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
                stream.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            return {
                "format": "tensorrt",
                "avg_inference_time_ms": avg_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "std_inference_time_ms": std_time,
                "throughput_samples_per_second": 1000 / avg_time if avg_time > 0 else 0,
                "meets_10ms_target": avg_time < 10,
                "engine_created": engine is not None
            }
            
        except ImportError:
            print("❌ TensorRT or pycuda not installed")
            return {}
        except Exception as e:
            print(f"❌ Error benchmarking TensorRT model: {e}")
            return {}
    
    def optimize_for_edge(self, 
                         output_dir: str = "models/optimized",
                         formats: List[str] = ["tflite_fp16", "tensorrt_fp16"]) -> Dict:
        """Complete edge optimization pipeline"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "original_model": self.model_path,
            "target_device": self.target_device,
            "optimizations": [],
            "benchmarks": {}
        }
        
        # Convert to different formats
        for format_spec in formats:
            format_type, precision = format_spec.split("_") if "_" in format_spec else (format_spec, "")
            
            if format_type == "tflite":
                output_path = os.path.join(output_dir, f"model_{precision}.tflite")
                success = self.convert_to_tflite(output_path, precision)
                
                if success:
                    results["optimizations"].append({
                        "format": "tflite",
                        "precision": precision,
                        "path": output_path,
                        "success": True
                    })
                    
                    # Benchmark
                    benchmark_result = self.benchmark_model(
                        "tflite", output_path, num_iterations=1000
                    )
                    results["benchmarks"][f"tflite_{precision}"] = benchmark_result
            
            elif format_type == "tensorrt":
                output_path = os.path.join(output_dir, f"model_{precision}.trt")
                success = self.convert_to_tensorrt(output_path, precision)
                
                if success:
                    results["optimizations"].append({
                        "format": "tensorrt",
                        "precision": precision,
                        "path": output_path,
                        "success": True
                    })
                    
                    # Benchmark
                    benchmark_result = self.benchmark_model(
                        "tensorrt", output_path, num_iterations=1000
                    )
                    results["benchmarks"][f"tensorrt_{precision}"] = benchmark_result
        
        # Benchmark original model
        original_benchmark = self.benchmark_model("original", num_iterations=100)
        results["benchmarks"]["original"] = original_benchmark
        
        # Save optimization report
        report_path = os.path.join(output_dir, "optimization_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Optimization complete!")
        print(f"   Report saved to {report_path}")
        
        # Print summary
        self._print_optimization_summary(results)
        
        return results
    
    def _print_optimization_summary(self, results: Dict):
        """Print optimization summary"""
        
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"\nTarget Device: {results['target_device']}")
        
        print("\nModel Formats Generated:")
        for opt in results["optimizations"]:
            if opt["success"]:
                print(f"  ✅ {opt['format'].upper()} ({opt['precision']}): {opt['path']}")
        
        print("\nPerformance Benchmarks:")
        for format_name, benchmark in results["benchmarks"].items():
            if benchmark:
                target_met = "✅" if benchmark.get("meets_10ms_target", False) else "❌"
                print(f"  {target_met} {format_name}:")
                print(f"    Avg Inference Time: {benchmark['avg_inference_time_ms']:.2f} ms")
                print(f"    Throughput: {benchmark['throughput_samples_per_second']:.1f} samples/sec")
        
        # Check if any format meets 10ms target
        meets_target = any(
            benchmark.get("meets_10ms_target", False)
            for benchmark in results["benchmarks"].values()
            if benchmark
        )
        
        print("\n" + "="*60)
        if meets_target:
            print("✅ SUCCESS: Models meet <10ms inference target!")
        else:
            print("⚠️  WARNING: No model meets <10ms inference target")
        print("="*60)


# Test the edge optimizer
if __name__ == "__main__":
    print("Testing Edge Model Optimizer...")
    
    # Create a dummy model for testing
    import tensorflow as tf
    
    # Build a simple model
    vibration_input = tf.keras.Input(shape=(1000, 1), name='vibration_input')
    thermal_input = tf.keras.Input(shape=(500, 1), name='thermal_input')
    acoustic_input = tf.keras.Input(shape=(250, 1), name='acoustic_input')
    pressure_input = tf.keras.Input(shape=(100, 1), name='pressure_input')
    
    # Process each input
    vib_features = tf.keras.layers.Conv1D(32, 3, activation='relu')(vibration_input)
    vib_features = tf.keras.layers.GlobalAveragePooling1D()(vib_features)
    
    th_features = tf.keras.layers.Dense(16, activation='relu')(thermal_input)
    th_features = tf.keras.layers.Flatten()(th_features)
    
    ac_features = tf.keras.layers.Dense(16, activation='relu')(acoustic_input)
    ac_features = tf.keras.layers.Flatten()(ac_features)
    
    pr_features = tf.keras.layers.Dense(16, activation='relu')(pressure_input)
    pr_features = tf.keras.layers.Flatten()(pr_features)
    
    # Concatenate features
    concatenated = tf.keras.layers.Concatenate()([vib_features, th_features, ac_features, pr_features])
    
    # Output layers
    fault_output = tf.keras.layers.Dense(16, activation='softmax', name='fault_classification')(concatenated)
    severity_output = tf.keras.layers.Dense(1, activation='linear', name='severity_prediction')(concatenated)
    
    # Create model
    model = tf.keras.Model(
        inputs=[vibration_input, thermal_input, acoustic_input, pressure_input],
        outputs=[fault_output, severity_output]
    )
    
    # Save model
    model.save("test_model.h5")
    print("✅ Test model created and saved")
    
    # Initialize optimizer
    optimizer = EdgeModelOptimizer("test_model.h5", target_device="CPU_ONLY")
    
    # Test TFLite conversion
    print("\nTesting TFLite conversion...")
    optimizer.convert_to_tflite("test_model_fp16.tflite", quantization="float16")
    
    # Test ONNX conversion
    print("\nTesting ONNX conversion...")
    optimizer.convert_to_onnx("test_model.onnx")
    
    # Benchmark models
    print("\nBenchmarking models...")
    original_benchmark = optimizer.benchmark_model("original", num_iterations=10)
    tflite_benchmark = optimizer.benchmark_model("tflite", "test_model_fp16.tflite", num_iterations=10)
    
    print(f"\nOriginal Model: {original_benchmark.get('avg_inference_time_ms', 0):.2f} ms")
    print(f"TFLite Model: {tflite_benchmark.get('avg_inference_time_ms', 0):.2f} ms")
    
    # Cleanup
    import os
    if os.path.exists("test_model.h5"):
        os.remove("test_model.h5")
    if os.path.exists("test_model_fp16.tflite"):
        os.remove("test_model_fp16.tflite")
    if os.path.exists("test_model.onnx"):
        os.remove("test_model.onnx")
    
    print("\n✅ Edge Model Optimizer test completed!")