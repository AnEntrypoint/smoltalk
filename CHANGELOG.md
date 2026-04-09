# Changelog

## 2026-04-09 (2)

### Fixed
- Pass prompt as chat messages array instead of raw string so SmolLM2-Instruct applies its chat template — fixes incoherent base-model-style output
- Extract assistant response from structured message array in results

## 2026-04-09

### Fixed
- Switch dtype from q4f16 to q4: q4f16 causes ORT WASM abort (264782152) on Windows D3D12 even when shader-f16 is reported available; q4 uses fp32 compute which works reliably on WebGPU across platforms

## 2026-04-08 (5)

### Changed
- Switched from @xenova/transformers (v2, WASM) to @huggingface/transformers (v3, WebGPU)
- Removed unused @mlc-ai/web-llm dependency
- convert_onnx.py: export with dtype=q4f16 (~67MB) instead of fp32 (~500MB)
- modelLoader.js: pipeline now uses device:'webgpu', dtype:'q4f16'

## 2026-04-08 (4)

### Fixed
- convert_onnx.py: replaced torch.onnx.export (logits-only, no KV cache) with optimum main_export (text-generation-with-past) so Transformers.js gets proper past_key_values for autoregressive decoding — fixes incoherent/repetitive generation
- deploy.yml: install optimum[onnxruntime] instead of bare torch+onnxruntime stack

## 2026-04-08

### Fixed
- tokenizer.json merges now post-processed from arrays to strings after CI download (fixes Transformers.js BPE `e.split is not a function`)
- modelLoader.js cache now intercepts `url.includes('/onnx/')` to match actual Transformers.js ONNX request path
- modelLoader.js manifest fetch failure now throws instead of silently falling back

## 2026-04-08 (2)

### Fixed
- modelLoader.js: set env.useBrowserCache=false so Transformers.js uses custom cache instead of browser Cache API for ONNX loading

## 2026-04-08 (3)

### Fixed
- convert_onnx.py: use QUInt8 dynamic quantization (browser WASM compatible) instead of MatMulNBits (com.microsoft op, not supported in ort-wasm)
- FP32 export + INT8 quantize_dynamic → ~155MB in 2 × 90MB chunks, fully browser-runnable
