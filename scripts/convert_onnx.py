import sys
import os
import json
import shutil
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from onnxruntime.quantization import MatMul4BitsQuantizer

model_id = "Real-Turf/SmolRP-135M-v0.9"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smolrp-135m"

os.makedirs(output_dir, exist_ok=True)

print(f"Loading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

print("Downloading tokenizer files...")
for fname in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=fname)
        shutil.copy(path, os.path.join(output_dir, fname))
        print(f"Copied {fname}")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

tok_path = os.path.join(output_dir, "tokenizer.json")
with open(tok_path, "r", encoding="utf-8") as f:
    tok = json.load(f)
if "model" in tok and "merges" in tok["model"]:
    tok["model"]["merges"] = [
        " ".join(m) if isinstance(m, list) else m
        for m in tok["model"]["merges"]
    ]
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
    print("Fixed tokenizer.json merges format")

input_ids = torch.ones((1, 5), dtype=torch.long)
attention_mask = torch.ones((1, 5), dtype=torch.long)

fp32_path = os.path.join(output_dir, "model_fp32.onnx")
onnx_path = os.path.join(output_dir, "model.onnx")

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
        opset_version=14,
        dynamo=False,
    )

fp32_mb = os.path.getsize(fp32_path) / 1024 / 1024
print(f"FP32 ONNX: {fp32_mb:.1f}MB")

print("Quantizing to INT4 (Q4F16)...")
quantizer = MatMul4BitsQuantizer(fp32_path, block_size=32, is_symmetric=True, accuracy_level=4)
quantizer.process()
quantizer.model.save_model_to_file(onnx_path, use_external_data_format=False)
os.remove(fp32_path)

q4_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"Q4F16 ONNX: {q4_mb:.1f}MB")
if q4_mb >= 100:
    raise RuntimeError(f"Quantized model is {q4_mb:.1f}MB, exceeds 100MB GitHub Pages limit")

total_bytes = os.path.getsize(onnx_path)
manifest = {"chunks": 1, "total_bytes": total_bytes}
with open(os.path.join(output_dir, "model.onnx.manifest.json"), "w") as f:
    json.dump(manifest, f)

print(f"Done: {onnx_path} ({q4_mb:.1f}MB)")
