import sys
import os
import json
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
onnx_repo = "onnx-community/SmolLM2-135M-Instruct-ONNX"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smollm2-135m-instruct"
CHUNK_MB = 90

os.makedirs(output_dir, exist_ok=True)

print("Downloading tokenizer and config files...")
for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "config.json", "generation_config.json"]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy(path, dst)
        print(f"Copied {fname}")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

tok_path = os.path.join(output_dir, "tokenizer.json")
if os.path.exists(tok_path):
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

print(f"Downloading q4 ONNX from {onnx_repo}...")
onnx_file = "onnx/model_q4.onnx"
path = hf_hub_download(repo_id=onnx_repo, filename=onnx_file)
onnx_path = os.path.join(output_dir, "model.onnx")
shutil.copy(path, onnx_path)
print(f"Downloaded ONNX")

q_mb = os.path.getsize(onnx_path) / 1024 / 1024
print(f"ONNX size: {q_mb:.1f}MB")

chunk_size = CHUNK_MB * 1024 * 1024
with open(onnx_path, "rb") as f:
    data = f.read()

total_bytes = len(data)
chunks = [data[i:i+chunk_size] for i in range(0, total_bytes, chunk_size)]
os.remove(onnx_path)

for i, chunk in enumerate(chunks):
    path = os.path.join(output_dir, f"model.onnx.part{i}")
    with open(path, "wb") as f:
        f.write(chunk)
    print(f"  part{i}: {len(chunk)/1024/1024:.1f}MB")

manifest = {"chunks": len(chunks), "total_bytes": total_bytes}
with open(os.path.join(output_dir, "model.onnx.manifest.json"), "w") as f:
    json.dump(manifest, f)

print(f"Done: {len(chunks)} parts, {q_mb:.1f}MB total")
