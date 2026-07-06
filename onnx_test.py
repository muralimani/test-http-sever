import os
import threading
import time

import numpy as np
import psutil
from perch_hoplite.zoo import models_onnx

# Path to the ONNX model
model_path = '/usr/local/share/perch/perch_v2.onnx'

_process = psutil.Process(os.getpid())


def get_cpu_temp_c():
  """Reads CPU temperature. Uses the Raspberry Pi thermal zone file first,
  falling back to psutil's sensor API for other platforms."""
  try:
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
      return int(f.read().strip()) / 1000.0
  except (FileNotFoundError, ValueError, PermissionError):
    pass
  try:
    temps = psutil.sensors_temperatures()
    for entries in temps.values():
      if entries:
        return entries[0].current
  except (AttributeError, NotImplementedError):
    pass
  return None


class ResourceMonitor:
  """Samples process/system CPU, memory (RSS), and CPU temperature on a
  background thread so short-lived spikes aren't missed by before/after
  snapshots."""

  def __init__(self, interval_s=0.05):
    self._interval_s = interval_s
    self._stop_event = threading.Event()
    self._thread = None
    self.samples = []

  def _run(self):
    _process.cpu_percent(interval=None)  # Prime the process counter.
    psutil.cpu_percent(interval=None)  # Prime the system-wide counter.
    while not self._stop_event.is_set():
      proc_cpu = _process.cpu_percent(interval=None)
      sys_cpu = psutil.cpu_percent(interval=None)
      rss_mb = _process.memory_info().rss / (1024 ** 2)
      temp_c = get_cpu_temp_c()
      self.samples.append((proc_cpu, sys_cpu, rss_mb, temp_c))
      time.sleep(self._interval_s)

  def __enter__(self):
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()
    return self

  def __exit__(self, *exc_info):
    self._stop_event.set()
    self._thread.join(timeout=1.0)

  def summary(self):
    if not self.samples:
      return {
          'proc_cpu_avg': 0.0, 'proc_cpu_max': 0.0,
          'sys_cpu_avg': 0.0, 'sys_cpu_max': 0.0,
          'rss_avg_mb': _process.memory_info().rss / (1024 ** 2),
          'rss_max_mb': _process.memory_info().rss / (1024 ** 2),
          'temp_avg_c': None, 'temp_max_c': None,
      }
    proc_cpu, sys_cpu, rss, temps = zip(*self.samples)
    temps = [t for t in temps if t is not None]
    return {
        'proc_cpu_avg': np.mean(proc_cpu),
        'proc_cpu_max': np.max(proc_cpu),
        'sys_cpu_avg': np.mean(sys_cpu),
        'sys_cpu_max': np.max(sys_cpu),
        'rss_avg_mb': np.mean(rss),
        'rss_max_mb': np.max(rss),
        'temp_avg_c': np.mean(temps) if temps else None,
        'temp_max_c': np.max(temps) if temps else None,
    }


def format_temp(value):
  return f"{value:.1f} C" if value is not None else "n/a"


# 1. Benchmark Model Loading
print("Loading model...")
rss_before_load_mb = _process.memory_info().rss / (1024 ** 2)
start_time = time.perf_counter()

with ResourceMonitor() as load_monitor:
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
load_stats = load_monitor.summary()
rss_after_load_mb = _process.memory_info().rss / (1024 ** 2)
print(f"Model loaded in {loading_time:.4f} seconds.")
print(f"  Memory footprint: {rss_after_load_mb - rss_before_load_mb:.1f} MB "
      f"(RSS now {rss_after_load_mb:.1f} MB)")
print(f"  Process CPU avg/max: {load_stats['proc_cpu_avg']:.1f}% / "
      f"{load_stats['proc_cpu_max']:.1f}%")

# 2. Prepare Audio Input (5 seconds of random audio)
duration_s = 5.0
# Changed to uniform(-1.0, 1.0) to generate varied audio data
audio = np.random.uniform(-1.0, 1.0, int(duration_s * model.sample_rate)).astype(np.float32)

# 3. Benchmark First Inference (Warmup)
# ONNX Runtime often performs one-time optimizations on the first run.
print("\nRunning warmup inference...")
start_time = time.perf_counter()
with ResourceMonitor() as warmup_monitor:
  output = model.embed(audio)
warmup_time = time.perf_counter() - start_time
warmup_stats = warmup_monitor.summary()
print(f"Warmup inference completed in {warmup_time:.4f} seconds.")
print(f"  Process CPU avg/max: {warmup_stats['proc_cpu_avg']:.1f}% / "
      f"{warmup_stats['proc_cpu_max']:.1f}%")

# 4. Benchmark Subsequent Inferences (10 iterations)
num_iterations = 10
inference_times = []
iteration_stats = []
print(f"\nRunning {num_iterations} benchmark iterations...")

for i in range(num_iterations):
  start_time = time.perf_counter()
  with ResourceMonitor() as iter_monitor:
    _ = model.embed(audio)
  elapsed = time.perf_counter() - start_time
  inference_times.append(elapsed)
  stats = iter_monitor.summary()
  iteration_stats.append(stats)
  print(f"  Iteration {i+1:02d}: {elapsed:.4f} s | "
        f"CPU {stats['proc_cpu_avg']:.1f}% (sys {stats['sys_cpu_avg']:.1f}%) | "
        f"RSS {stats['rss_max_mb']:.1f} MB | "
        f"Temp {format_temp(stats['temp_max_c'])}")

# 5. Calculate Statistics
avg_time = np.mean(inference_times)
std_time = np.std(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)

proc_cpu_avgs = [s['proc_cpu_avg'] for s in iteration_stats]
sys_cpu_avgs = [s['sys_cpu_avg'] for s in iteration_stats]
rss_maxes = [s['rss_max_mb'] for s in iteration_stats]
temp_maxes = [s['temp_max_c'] for s in iteration_stats if s['temp_max_c'] is not None]
rss_after_inference_mb = _process.memory_info().rss / (1024 ** 2)

print("\n" + "="*40)
print("BENCHMARK RESULTS")
print("="*40)
print(f"Model Loading Time:      {loading_time:.4f} s")
print(f"Warmup Inference Time:   {warmup_time:.4f} s")
print(f"Average Inference Time:  {avg_time:.4f} s (± {std_time:.4f} s)")
print(f"Min Inference Time:      {min_time:.4f} s")
print(f"Max Inference Time:      {max_time:.4f} s")
print("-"*40)
print("RESOURCE USAGE")
print("-"*40)
print(f"Model Memory Footprint:  {rss_after_load_mb - rss_before_load_mb:.1f} MB")
print(f"Peak RSS Memory:         {max(rss_maxes):.1f} MB")
print(f"RSS After All Runs:      {rss_after_inference_mb:.1f} MB "
      f"(delta since warmup: {rss_after_inference_mb - rss_after_load_mb:+.1f} MB)")
print(f"Process CPU (avg/max):   {np.mean(proc_cpu_avgs):.1f}% / {max(s['proc_cpu_max'] for s in iteration_stats):.1f}%")
print(f"System CPU (avg/max):    {np.mean(sys_cpu_avgs):.1f}% / {max(s['sys_cpu_max'] for s in iteration_stats):.1f}%")
if temp_maxes:
  print(f"CPU Temperature (avg/max): {np.mean(temp_maxes):.1f} C / {max(temp_maxes):.1f} C")
else:
  print("CPU Temperature:         n/a (no sensor found)")
print(f"CPU Core Count:          {psutil.cpu_count(logical=True)} logical "
      f"/ {psutil.cpu_count(logical=False)} physical")
print("="*40)

if output.embeddings is not None:
  print(f"Embeddings shape: {output.embeddings.shape}")
else:
  print("No embeddings returned.")
