import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import psutil

def run_benchmark(model_path, output_path):
    process = psutil.Process(os.getpid())
    
    # Measure memory before loading
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Load session
    session = ort.InferenceSession(model_path)
    
    # Measure memory after loading
    mem_after = process.memory_info().rss / (1024 * 1024)
    memory_delta_mb = mem_after - mem_before
    
    input_name = session.get_inputs()[0].name
    
    # CRNN expects ['batch', 1, 32, 'width'] tensor(float)
    dummy_input = np.random.randn(1, 1, 32, 128).astype(np.float32)
    
    # Warmup
    for _ in range(20):
        session.run(None, {input_name: dummy_input})
        
    # Benchmark
    num_runs = 200
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    
    results = {
        "avg_latency_ms": round(avg_latency_ms, 2),
        "memory_delta_mb": round(memory_delta_mb, 2)
    }
    
    print(f"Latency: {results['avg_latency_ms']} ms")
    print(f"Memory Delta: {results['memory_delta_mb']} MB")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to ONNX model")
    args = parser.parse_args()
    
    output_path = "edge/benchmarks/lightweight_ocr.json"
    run_benchmark(args.model_path, output_path)
