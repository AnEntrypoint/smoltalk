import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Real-Turf/SmolRP-135M-v0.9"
output_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/smolrp-135m"

os.makedirs(output_dir, exist_ok=True)

print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

tokenizer.save_pretrained(output_dir)

sample_input = tokenizer("Hello", return_tensors="pt")
input_ids = sample_input["input_ids"]
attention_mask = sample_input["attention_mask"]

print("Exporting ONNX (legacy TorchScript path)...")
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        os.path.join(output_dir, "model.onnx"),
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

print(f"ONNX model saved to {output_dir}/model.onnx")
size_mb = os.path.getsize(os.path.join(output_dir, "model.onnx")) / 1024 / 1024
print(f"Model size: {size_mb:.1f}MB")
