import time
import json
import glob
import os
import numpy as np
import onnxruntime as ort

# Configure Session Options for Profiling
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.profile_file_prefix = "resnet18_profile"

# Option to set execution mode (Sequential vs Parallel operator execution)
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# define execution providers
# Test with CPU first to see full node-by-node execution, then switch to CoreML
providers = ["CPUExecutionProvider"]
# providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

print(f"Initializing session with providers: {providers}")
session = ort.InferenceSession("resnet18.onnx", sess_options, providers=providers)

# Prepare Dummy Input: [Batch=1, Channels=3, Height=224, Width=224]
input_name = session.get_inputs()[0].name
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warmup Iterations (Crucial for hardware profiling to eliminate initial cache misses)
print("Running warm-up iterations...")
for _ in range(5):
    _ = session.run(None, {input_name: dummy_input})

# Timed Inference Benchmark Loop
num_runs = 20
latencies = []

print(f"Running {num_runs} profiled inference passes...")
for _ in range(num_runs):
    start_time = time.perf_counter()
    _ = session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    latencies.append((end_time - start_time) * 1000.0) # Convert to ms

# End profiling session
profile_file = session.end_profiling()
print(f"\nProfiling complete!")
print(f"Average Latency: {np.mean(latencies):.2f} ms")
print(f"P95 Latency:     {np.percentile(latencies, 95):.2f} ms")
print(f"Throughput:      {1000.0 / np.mean(latencies):.2f} FPS")
print(f"Trace file saved as: {profile_file}")