# Changelog

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
