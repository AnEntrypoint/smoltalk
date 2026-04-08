import sys
import os
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
for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=fname)
        shutil.copy(path, os.path.join(output_dir, fname))
        print(f"Copied {fname}")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

input_ids = torch.ones((1, 5), dtype=torch.long)
attention_mask = torch.ones((1, 5), dtype=torch.long)

fp32_path = os.path.join(output_dir, "model_fp32.onnx")
q4_path = os.path.join(output_dir, "model.onnx")

print("Exporting FP32 ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
        dynamo=False,
    )

print(f"FP32: {os.path.getsize(fp32_path)/1024/1024:.1f}MB")

print("Quantizing to INT4...")
from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
import onnx

q4_model = MatMul4BitsQuantizer(onnx.load(fp32_path), block_size=32, is_symmetric=True, accuracy_level=4)
q4_model.process()
q4_model.model.save_model_to_file(q4_path, use_external_data_format=False)
os.remove(fp32_path)

size_mb = os.path.getsize(q4_path) / 1024 / 1024
print(f"INT4: {size_mb:.1f}MB")

if size_mb > CHUNK_MB:
    print(f"Splitting into {CHUNK_MB}MB chunks...")
    with open(q4_path, 'rb') as f:
        data = f.read()
    chunk_size = CHUNK_MB * 1024 * 1024
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"model.onnx.part{i}")
        with open(chunk_path, 'wb') as f:
            f.write(chunk)
        print(f"  part{i}: {len(chunk)/1024/1024:.1f}MB")
    os.remove(q4_path)
    manifest = {"chunks": len(chunks), "total_bytes": len(data)}
    import json
    with open(os.path.join(output_dir, "model.onnx.manifest.json"), 'w') as f:
        json.dump(manifest, f)
    print(f"Split into {len(chunks)} parts, manifest written")
else:
    print(f"Model fits in one file ({size_mb:.1f}MB), no split needed")
