import numpy as np
from perch_hoplite.zoo import models_onnx
model_path =  '/usr/local/share/perch/perch_v2.onnx'

model = models_onnx.TaxonomyModelOnnx( sample_rate=32000, model_path= model_path, input_name = 'inputs', output_map={'embedding': 'embedding', 'logits': 'logits', 'frontend':'spectogram'}, window_size_s=5.0, hop_size_s=5.0,target_peak=0.25)

duration_s = 5.0
audio = np.random.uniform(-1.0,-1.0,int(duration_s*model.sample_rate)).astype(np.float32)
#output = model.embed(np.zeros([5 * 32_000], dtype=np.float32))
output = model.embed(audio)
print ("Model embeding complete")

if output.embeddings  is not None:
  print("Embeddings shape" , output.embeddings.shape)
else:
  print("No embeddings")
