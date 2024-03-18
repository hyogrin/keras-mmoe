import tensorflow as tf
import numpy as np
import logging
from time import perf_counter
import csv

logger = logging.getLogger(__name__)

def benchmark_latency(model, input_data, data_length, num_infers=10):
    logger.info(f"Measuring latency for sequence length={data_length}, num_infers={num_infers}")
    payload = generate_sample_inputs(input_data, data_length)
    latencies = []
    # warm up
    for _ in range(5):
        _ = model(payload)
    # Timed run
    for _ in range(num_infers):
        start_time = perf_counter()
        _ =  model(payload)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    time_p99_ms = 1000 * np.percentile(latencies,99)    
    return {"avg_ms": time_avg_ms, "std_ms": time_std_ms, "p95_ms": time_p95_ms, "p99_ms": time_p95_ms}
 
def generate_sample_inputs(input_data, data_length=10):
    input_data = input_data[:data_length]
    return input_data