import sys
import os
import json
import shutil
import subprocess

model_id = "Real-Turf/SmolRP-135M-v0.9"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smolrp-135m"
tmp_dir = output_dir + "_tmp"
CHUNK_MB = 90

os.makedirs(output_dir, exist_ok=True)

print(f"Exporting {model_id} to FP16 ONNX via optimum...")
subprocess.run([
    "optimum-cli", "export", "onnx",
    "--model", model_id,
    "--task", "text-generation",
    "--dtype", "fp16",
    "--trust-remote-code",
    tmp_dir
], check=True)

fp16_path = os.path.join(tmp_dir, "model.onnx")
if not os.path.exists(fp16_path):
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith(".onnx"):
                fp16_path = os.path.join(root, f)
                break

fp16_mb = os.path.getsize(fp16_path) / 1024 / 1024
print(f"FP16 ONNX: {fp16_mb:.1f}MB")

from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
onnx_path = os.path.join(tmp_dir, "model_q4.onnx")

print("Quantizing FP16 to INT4 (MatMulNBits)...")
quantizer = MatMulNBitsQuantizer(fp16_path, 4, 32, is_symmetric=True)
quantizer.process()
quantizer.model.save_model_to_file(onnx_path, use_external_data_format=False)

q_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"INT4 ONNX: {q_mb:.1f}MB")

for fname in ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
    src = os.path.join(tmp_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(output_dir, fname))
        print(f"Copied {fname}")

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

chunk_size = CHUNK_MB * 1024 * 1024
with open(onnx_path, "rb") as f:
    data = f.read()

total_bytes = len(data)
chunks = [data[i:i+chunk_size] for i in range(0, total_bytes, chunk_size)]

for i, chunk in enumerate(chunks):
    path = os.path.join(output_dir, f"model.onnx.part{i}")
    with open(path, "wb") as f:
        f.write(chunk)
    print(f"  part{i}: {len(chunk)/1024/1024:.1f}MB")

shutil.rmtree(tmp_dir)

manifest = {"chunks": len(chunks), "total_bytes": total_bytes}
with open(os.path.join(output_dir, "model.onnx.manifest.json"), "w") as f:
    json.dump(manifest, f)

print(f"Done: {len(chunks)} parts, {q_mb:.1f}MB total (INT4)")
