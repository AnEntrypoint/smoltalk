import sys
import os
import json
import shutil
from optimum.exporters.onnx import main_export
from huggingface_hub import hf_hub_download

model_id = "Real-Turf/SmolRP-135M-v0.9"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smolrp-135m"
CHUNK_MB = 90

os.makedirs(output_dir, exist_ok=True)

print("Exporting with optimum (KV cache, FP32)...")
main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="text-generation-with-past",
    dtype="fp32",
)
print("optimum export done")

print("Downloading extra tokenizer files...")
for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "config.json", "generation_config.json"]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            shutil.copy(path, dst)
            print(f"Copied {fname}")
        else:
            print(f"Already present: {fname}")
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

onnx_path = os.path.join(output_dir, "model.onnx")
merged_path = os.path.join(output_dir, "model_merged.onnx")
if not os.path.exists(onnx_path) and os.path.exists(merged_path):
    os.rename(merged_path, onnx_path)
    print("Renamed model_merged.onnx -> model.onnx")

if not os.path.exists(onnx_path):
    raise RuntimeError(f"No model.onnx found in {output_dir}. Files: {os.listdir(output_dir)}")

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
