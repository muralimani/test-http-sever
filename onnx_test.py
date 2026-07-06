import time
import numpy as np
from perch_hoplite.zoo import models_onnx

# Path to the ONNX model
model_path = '/usr/local/share/perch/perch_v2.onnx'
# 1. Benchmark Model Loading
print("Loading model...")
start_time = time.perf_counter()

model = models_onnx.TaxonomyModelOnnx(
    sample_rate=32000,
    model_path=model_path,
    input_name='inputs',
    # Note: Ensure these output names match your ONNX model.
    # Standard Perch v2 ONNX often uses 'label' for logits and 'spectrogram' for frontend.
    output_map={
        'embedding': 'embedding',
        'logits': 'logits',
        'frontend': 'spectogram'
    },
    window_size_s=5.0,
    hop_size_s=5.0,
    target_peak=0.25
)

loading_time = time.perf_counter() - start_time
print(f"Model loaded in {loading_time:.4f} seconds.")

# 2. Prepare Audio Input (5 seconds of random audio)
duration_s = 5.0
# Changed to uniform(-1.0, 1.0) to generate varied audio data
audio = np.random.uniform(-1.0, 1.0, int(duration_s * model.sample_rate)).astype(np.float32)

# 3. Benchmark First Inference (Warmup)
# ONNX Runtime often performs one-time optimizations on the first run.
print("\nRunning warmup inference...")
start_time = time.perf_counter()
output = model.embed(audio)
warmup_time = time.perf_counter() - start_time
print(f"Warmup inference completed in {warmup_time:.4f} seconds.")

# 4. Benchmark Subsequent Inferences (10 iterations)
num_iterations = 10
inference_times = []
print(f"\nRunning {num_iterations} benchmark iterations...")

for i in range(num_iterations):
  start_time = time.perf_counter()
  _ = model.embed(audio)
  elapsed = time.perf_counter() - start_time
  inference_times.append(elapsed)
  print(f"  Iteration {i+1:02d}: {elapsed:.4f} seconds")

# 5. Calculate Statistics
avg_time = np.mean(inference_times)
std_time = np.std(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)

print("\n" + "="*40)
print("BENCHMARK RESULTS")
print("="*40)
print(f"Model Loading Time:      {loading_time:.4f} s")
print(f"Warmup Inference Time:  {warmup_time:.4f} s")
print(f"Average Inference Time:  {avg_time:.4f} s (± {std_time:.4f} s)")
print(f"Min Inference Time:      {min_time:.4f} s")
print(f"Max Inference Time:      {max_time:.4f} s")
print("="*40)

if output.embeddings is not None:
  print(f"Embeddings shape: {output.embeddings.shape}")
else:
  print("No embeddings returned.")
