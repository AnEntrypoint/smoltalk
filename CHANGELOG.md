# Changelog

## 2026-04-08

### Fixed
- tokenizer.json merges now post-processed from arrays to strings after CI download (fixes Transformers.js BPE `e.split is not a function`)
- modelLoader.js cache now intercepts `url.includes('/onnx/')` to match actual Transformers.js ONNX request path
- modelLoader.js manifest fetch failure now throws instead of silently falling back

## 2026-04-08 (2)

### Fixed
- modelLoader.js: set env.useBrowserCache=false so Transformers.js uses custom cache instead of browser Cache API for ONNX loading

## 2026-04-08 (3)

### Changed
- convert_onnx.py: export FP16 ONNX via optimum then apply MatMulNBitsQuantizer INT4 — 128MB total in 2 × 90MB chunks
- deploy.yml: optimum[onnxruntime] + onnx-ir deps for FP16 export and INT4 quantization
- model.onnx.manifest.json chunks:2 (FP16+INT4 quantized, each part under 90MB)
