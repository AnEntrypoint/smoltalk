import sys
import os
import json
import shutil
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

model_id = "Real-Turf/SmolRP-135M-v0.9"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smolrp-135m"
CHUNK_MB = 90

os.makedirs(output_dir, exist_ok=True)

print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

print("Downloading tokenizer files directly...")
for fname in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=fname)
        shutil.copy(path, os.path.join(output_dir, fname))
        print(f"Copied {fname}")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

input_ids = torch.ones((1, 5), dtype=torch.long)
attention_mask = torch.ones((1, 5), dtype=torch.long)

onnx_path = os.path.join(output_dir, "model.onnx")

print("Exporting ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        dynamo=False,
    )

size_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"ONNX: {size_mb:.1f}MB")

chunk_size = CHUNK_MB * 1024 * 1024
with open(onnx_path, 'rb') as f:
    data = f.read()

total_bytes = len(data)
chunks = [data[i:i+chunk_size] for i in range(0, total_bytes, chunk_size)]
os.remove(onnx_path)

for i, chunk in enumerate(chunks):
    path = os.path.join(output_dir, f"model.onnx.part{i}")
    with open(path, 'wb') as f:
        f.write(chunk)
    print(f"  part{i}: {len(chunk)/1024/1024:.1f}MB")

manifest = {"chunks": len(chunks), "total_bytes": total_bytes}
with open(os.path.join(output_dir, "model.onnx.manifest.json"), 'w') as f:
    json.dump(manifest, f)

print(f"Split into {len(chunks)} parts ({total_bytes} bytes total)")
