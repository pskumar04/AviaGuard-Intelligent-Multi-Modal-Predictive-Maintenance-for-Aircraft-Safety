"""
TensorRT Conversion Script
Converts models to TensorRT format for NVIDIA Jetson deployment
"""

import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional
import argparse

class TensorRTConverter:
    """Converts TensorFlow models to TensorRT format"""
    
    def __init__(self, model_path: str, precision: str = "FP16"):
        self.model_path = model_path
        self.precision = precision.upper()
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Precision mapping
        self.precision_map = {
            "FP32": trt.float32,
            "FP16": trt.float16,
            "INT8": trt.int8
        }
        
    def convert_tf_to_onnx(self, output_path: str) -> bool:
        """Convert TensorFlow model to ONNX format"""
        try:
            import tf2onnx
            
            print(f"Converting TensorFlow model to ONNX: {self.model_path}")
            
            # Load TensorFlow model
            model = tf.keras.models.load_model(self.model_path)
            
            # Define input signature for multi-modal model
            input_signature = [
                tf.TensorSpec((None, 1000, 1), tf.float32, name='vibration_input'),
                tf.TensorSpec((None, 500, 1), tf.float32, name='thermal_input'),
                tf.TensorSpec((None, 250, 1), tf.float32, name='acoustic_input'),
                tf.TensorSpec((None, 100, 1), tf.float32, name='pressure_input')
            ]
            
            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
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
    
    def build_tensorrt_engine(self, onnx_path: str, output_path: str) -> Optional[trt.ICudaEngine]:
        """Build TensorRT engine from ONNX model"""
        
        print(f"Building TensorRT engine (precision: {self.precision})...")
        
        try:
            # Create builder and network
            builder = trt.Builder(self.logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create parser
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    print("❌ Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            print(f"✅ ONNX model parsed successfully")
            print(f"   Network layers: {network.num_layers}")
            print(f"   Inputs: {network.num_inputs}")
            print(f"   Outputs: {network.num_outputs}")
            
            # Build configuration
            config = builder.create_builder_config()
            
            # Set precision
            if self.precision == "FP16":
                config.set_flag(trt.BuilderFlag.FP16)
                print("   Using FP16 precision")
            elif self.precision == "INT8":
                config.set_flag(trt.BuilderFlag.INT8)
                print("   Using INT8 precision")
                # Note: INT8 requires calibration dataset
                # config.int8_calibrator = self.create_calibrator()
            else:
                print("   Using FP32 precision")
            
            # Set optimization profiles
            profile = builder.create_optimization_profile()
            
            # Set input shapes for multi-modal model
            # Vibration input: [batch, 1000, 1]
            profile.set_shape(
                "vibration_input",
                min=(1, 1000, 1),
                opt=(8, 1000, 1),
                max=(32, 1000, 1)
            )
            
            # Thermal input: [batch, 500, 1]
            profile.set_shape(
                "thermal_input",
                min=(1, 500, 1),
                opt=(8, 500, 1),
                max=(32, 500, 1)
            )
            
            # Acoustic input: [batch, 250, 1]
            profile.set_shape(
                "acoustic_input",
                min=(1, 250, 1),
                opt=(8, 250, 1),
                max=(32, 250, 1)
            )
            
            # Pressure input: [batch, 100, 1]
            profile.set_shape(
                "pressure_input",
                min=(1, 100, 1),
                opt=(8, 100, 1),
                max=(32, 100, 1)
            )
            
            config.add_optimization_profile(profile)
            
            # Set workspace size (adjust based on available GPU memory)
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Set optimization level
            config.builder_optimization_level = 3
            
            # Build engine
            print("Building TensorRT engine (this may take several minutes)...")
            start_time = time.time()
            engine = builder.build_engine(network, config)
            build_time = time.time() - start_time
            
            if engine is None:
                print("❌ Failed to build TensorRT engine")
                return None
            
            print(f"✅ TensorRT engine built in {build_time:.2f} seconds")
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"✅ TensorRT engine saved to {output_path}")
            
            # Save engine metadata
            self.save_engine_metadata(engine, output_path, build_time)
            
            return engine
            
        except Exception as e:
            print(f"❌ Error building TensorRT engine: {e}")
            return None
    
    def save_engine_metadata(self, engine: trt.ICudaEngine, engine_path: str, build_time: float):
        """Save TensorRT engine metadata"""
        
        metadata = {
            "engine_path": engine_path,
            "precision": self.precision,
            "build_time_seconds": build_time,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": [],
            "outputs": [],
            "layers": engine.num_layers,
            "max_batch_size": engine.max_batch_size
        }
        
        # Collect input information
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                shape = engine.get_binding_shape(i)
                dtype = str(engine.get_binding_dtype(i))
                name = engine.get_binding_name(i)
                
                metadata["inputs"].append({
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype
                })
        
        # Collect output information
        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                shape = engine.get_binding_shape(i)
                dtype = str(engine.get_binding_dtype(i))
                name = engine.get_binding_name(i)
                
                metadata["outputs"].append({
                    "name": name,
                    "shape": list(shape),
                    "dtype": dtype
                })
        
        # Save metadata
        metadata_path = engine_path.replace('.trt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ TensorRT metadata saved to {metadata_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("TensorRT Engine Summary")
        print("="*50)
        print(f"Precision: {self.precision}")
        print(f"Build Time: {build_time:.2f} seconds")
        print(f"Layers: {engine.num_layers}")
        print(f"Max Batch Size: {engine.max_batch_size}")
        
        print("\nInputs:")
        for inp in metadata["inputs"]:
            print(f"  {inp['name']}: {inp['shape']} ({inp['dtype']})")
        
        print("\nOutputs:")
        for out in metadata["outputs"]:
            print(f"  {out['name']}: {out['shape']} ({out['dtype']})")
        print("="*50)
    
    def benchmark_engine(self, engine_path: str, num_iterations: int = 100) -> Dict:
        """Benchmark TensorRT engine performance"""
        
        print(f"\nBenchmarking TensorRT engine...")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate buffers
            inputs, outputs, bindings = [], [], []
            stream = cuda.Stream()
            
            for i in range(engine.num_bindings):
                size = trt.volume(engine.get_binding_shape(i))
                dtype = trt.nptype(engine.get_binding_dtype(i))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                bindings.append(int(device_mem))
                
                if engine.binding_is_input(i):
                    inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem})
            
            # Prepare dummy input data
            # Create random data for each input
            input_data = []
            
            # Vibration input: [batch, 1000, 1]
            vib_data = np.random.randn(1, 1000, 1).astype(np.float32)
            input_data.append(vib_data.flatten())
            
            # Thermal input: [batch, 500, 1]
            th_data = np.random.randn(1, 500, 1).astype(np.float32)
            input_data.append(th_data.flatten())
            
            # Acoustic input: [batch, 250, 1]
            ac_data = np.random.randn(1, 250, 1).astype(np.float32)
            input_data.append(ac_data.flatten())
            
            # Pressure input: [batch, 100, 1]
            pr_data = np.random.randn(1, 100, 1).astype(np.float32)
            input_data.append(pr_data.flatten())
            
            # Copy data to input buffers
            for i, data in enumerate(input_data):
                np.copyto(inputs[i]['host'], data)
            
            # Warmup
            print("  Warmup...")
            for _ in range(10):
                for i in range(len(inputs)):
                    cuda.memcpy_htod_async(inputs[i]['device'], inputs[i]['host'], stream)
                
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                
                for i in range(len(outputs)):
                    cuda.memcpy_dtoh_async(outputs[i]['host'], outputs[i]['device'], stream)
                
                stream.synchronize()
            
            # Benchmark
            print(f"  Running {num_iterations} iterations...")
            times = []
            
            for iteration in range(num_iterations):
                start_time = time.perf_counter()
                
                # Copy inputs
                for i in range(len(inputs)):
                    cuda.memcpy_htod_async(inputs[i]['device'], inputs[i]['host'], stream)
                
                # Execute
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                
                # Copy outputs
                for i in range(len(outputs)):
                    cuda.memcpy_dtoh_async(outputs[i]['host'], outputs[i]['device'], stream)
                
                stream.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            results = {
                "format": "tensorrt",
                "precision": self.precision,
                "avg_inference_time_ms": avg_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "std_inference_time_ms": std_time,
                "throughput_samples_per_second": 1000 / avg_time if avg_time > 0 else 0,
                "meets_10ms_target": avg_time < 10
            }
            
            print("\n" + "="*50)
            print("Benchmark Results")
            print("="*50)
            print(f"Average Inference Time: {avg_time:.2f} ms")
            print(f"Minimum Inference Time: {min_time:.2f} ms")
            print(f"Maximum Inference Time: {max_time:.2f} ms")
            print(f"Throughput: {1000/avg_time:.1f} samples/sec")
            print(f"Meets 10ms Target: {'✅' if avg_time < 10 else '❌'}")
            print("="*50)
            
            return results
            
        except Exception as e:
            print(f"❌ Error benchmarking engine: {e}")
            return {}
    
    def convert(self, output_dir: str = "models/tensorrt") -> Dict:
        """Complete conversion pipeline"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        model_name = os.path.basename(self.model_path).replace('.h5', '')
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        engine_path = os.path.join(output_dir, f"{model_name}_{self.precision}.trt")
        
        results = {
            "model_path": self.model_path,
            "precision": self.precision,
            "success": False,
            "onnx_path": onnx_path,
            "engine_path": engine_path,
            "benchmark": {}
        }
        
        print(f"\nStarting TensorRT conversion for: {self.model_path}")
        print(f"Target precision: {self.precision}")
        print(f"Output directory: {output_dir}")
        
        # Step 1: Convert to ONNX
        print("\n" + "="*50)
        print("Step 1: Converting to ONNX")
        print("="*50)
        
        if not self.convert_tf_to_onnx(onnx_path):
            return results
        
        # Step 2: Build TensorRT engine
        print("\n" + "="*50)
        print("Step 2: Building TensorRT Engine")
        print("="*50)
        
        engine = self.build_tensorrt_engine(onnx_path, engine_path)
        
        if engine is None:
            return results
        
        # Step 3: Benchmark engine
        print("\n" + "="*50)
        print("Step 3: Benchmarking Engine")
        print("="*50)
        
        benchmark_results = self.benchmark_engine(engine_path, num_iterations=100)
        
        results["success"] = True
        results["benchmark"] = benchmark_results
        
        # Cleanup ONNX file if desired
        # os.remove(onnx_path)
        
        return results


def main():
    """Main conversion function"""
    
    parser = argparse.ArgumentParser(description="Convert TensorFlow models to TensorRT format")
    parser.add_argument("--model", "-m", required=True, help="Path to TensorFlow model (.h5)")
    parser.add_argument("--precision", "-p", default="FP16", choices=["FP32", "FP16", "INT8"], 
                       help="Precision for TensorRT engine")
    parser.add_argument("--output", "-o", default="models/tensorrt", 
                       help="Output directory for TensorRT engine")
    parser.add_argument("--benchmark", "-b", action="store_true", 
                       help="Run benchmarking after conversion")
    
    args = parser.parse_args()
    
    print("="*60)
    print("TensorRT Conversion Tool")
    print("="*60)
    
    # Check if TensorRT is available
    try:
        import tensorrt as trt
        print(f"✅ TensorRT version: {trt.__version__}")
    except ImportError:
        print("❌ TensorRT not installed. Please install TensorRT first.")
        print("   For Jetson: sudo apt-get install tensorrt")
        print("   For x86: Follow NVIDIA TensorRT installation guide")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Initialize converter
    converter = TensorRTConverter(args.model, args.precision)
    
    # Run conversion
    results = converter.convert(args.output)
    
    if results["success"]:
        print("\n" + "="*60)
        print("✅ CONVERSION SUCCESSFUL!")
        print("="*60)
        
        print(f"\nOutput Files:")
        print(f"  ONNX Model: {results['onnx_path']}")
        print(f"  TensorRT Engine: {results['engine_path']}")
        
        if results["benchmark"]:
            benchmark = results["benchmark"]
            print(f"\nPerformance:")
            print(f"  Avg Inference Time: {benchmark['avg_inference_time_ms']:.2f} ms")
            print(f"  Throughput: {benchmark['throughput_samples_per_second']:.1f} samples/sec")
            
            if benchmark['meets_10ms_target']:
                print(f"  ✅ Meets <10ms inference target!")
            else:
                print(f"  ⚠️  Does not meet <10ms inference target")
        
        # Save conversion report
        report_path = os.path.join(args.output, "conversion_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Conversion report saved to: {report_path}")
        
    else:
        print("\n" + "="*60)
        print("❌ CONVERSION FAILED")
        print("="*60)


if __name__ == "__main__":
    main()