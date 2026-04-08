# Changelog

## 2026-04-08

### Fixed
- tokenizer.json merges now post-processed from arrays to strings after CI download (fixes Transformers.js BPE `e.split is not a function`)
- modelLoader.js cache now intercepts `url.includes('/onnx/')` to match actual Transformers.js ONNX request path
- modelLoader.js manifest fetch failure now throws instead of silently falling back
