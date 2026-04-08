# SmolLM2-135M Emotion Classifier with web-llm

A browser-based emotion classification application using the [SmolLM2-135M-emotions](https://huggingface.co/dianak12/SmolLM2-135M-emotions) model from Hugging Face, powered by [Transformers.js](https://xenova.github.io/transformers.js/) and [web-llm](https://github.com/mlc-ai/web-llm).

## Features

- 🚀 **In-Browser Processing**: No server required, all inference runs in your browser
- 🤖 **Emotion Classification**: Analyze text to detect emotions (joy, sadness, anger, fear, surprise, etc.)
- ⚡ **Fast Inference**: Optimized for 135M parameter model
- 🎨 **Modern UI**: Clean, responsive interface with visual emotion probability bars
- 🔧 **web-llm Integration**: Full setup with web-llm engine for extensibility

## Architecture

This project combines two powerful libraries:

1. **Transformers.js** - Loads the HuggingFace SmolLM2 emotion model directly in the browser
2. **web-llm** - Provides LLM capabilities and serves as the foundation for this application

The emotion classifier runs entirely client-side using ONNX.js for inference.

## Installation

### Prerequisites
- Node.js 16+ and npm
- Modern browser with WebGPU or WebAssembly support

### Setup

```bash
# Clone or download the project
cd smollm2-emotions-demo

# Install dependencies
npm install

# Start development server
npm run dev
```

The dev server will start at `https://localhost:5173` (Vite uses HTTPS for SharedArrayBuffer support).

## Usage

1. Open the app in your browser
2. Wait for the emotion model to load (first run may take a moment as it downloads the model)
3. Enter text in the textarea
4. Click "Analyze" or press Enter
5. View the emotion probabilities with visual bars

### Example Inputs

- "I'm so happy today!" → Joy
- "This is terrible and I hate it" → Anger
- "I can't believe what just happened" → Surprise
- "Life is meaningless" → Sadness

## Project Structure

```
smollm2-emotions-demo/
├── index.html          # Main UI
├── package.json        # Dependencies
├── vite.config.js      # Build configuration
├── src/
│   ├── main.js         # Application logic
│   └── modelLoader.js  # Model initialization
└── dist/              # Built output (after `npm run build`)
```

## Model Information

**Model**: SmolLM2-135M-emotions
- **Source**: [dianak12/SmolLM2-135M-emotions](https://huggingface.co/dianak12/SmolLM2-135M-emotions)
- **Parameters**: 135M
- **Task**: Text Classification (Emotion Detection)
- **Framework**: Transformers (PyTorch)

### Output Format

The model returns emotion classifications with confidence scores:

```javascript
[
  { label: 'joy', score: 0.85 },
  { label: 'neutral', score: 0.10 },
  { label: 'sadness', score: 0.03 },
  { label: 'anger', score: 0.02 }
]
```

## Performance Notes

- **First Load**: 10-30 seconds (downloads ~400MB model files from HuggingFace)
- **Subsequent Loads**: Cached in browser (localStorage and IndexedDB)
- **Inference Speed**: 100-500ms per text (depends on text length and hardware)
- **Browser Support**: Chrome, Firefox, Edge (requires WebAssembly and SharedArrayBuffer)

## Troubleshooting

### Model fails to load
- **Check your internet connection** - First download is large (~400MB)
- **Clear browser cache** - Try `Ctrl+Shift+Delete` and retry
- **Update browser** - Ensure you have the latest version
- **Check console** - Press F12 and look for error messages

### "SharedArrayBuffer is not defined"
- Your browser doesn't support SharedArrayBuffer
- Try a different browser (Chrome recommended)
- Ensure you're accessing via HTTPS (the dev server uses HTTPS)

### Slow inference
- Close other browser tabs to free up resources
- Reduce text length for faster analysis
- Check browser console for warnings

### CORS or model download errors
- The HuggingFace model CDN is being accessed
- Ensure you're online and CDN is accessible
- Try clearing IndexedDB: Open DevTools → Application → Storage → IndexedDB

## Development

### Build for Production

```bash
npm run build
```

Output goes to `dist/` directory.

### Development Server

```bash
npm run dev
```

Starts Vite dev server with hot reload support.

## Configuration

### Vite Config (`vite.config.js`)

Key settings for web-llm compatibility:
- COOP/COEP headers for SharedArrayBuffer
- ES2020 target for modern JavaScript
- Web worker support for parallel processing

### Model Configuration (`src/modelLoader.js`)

Customize the emotion model or add additional models:

```javascript
// Use a different emotion model from HuggingFace
emotionPipeline = await pipeline(
  'text-classification',
  'your-username/your-emotion-model'
);
```

## Extended Usage with web-llm

The project includes optional web-llm integration for text generation:

```javascript
import { initWebLLM } from './modelLoader.js';

// Initialize web-llm engine
const engine = await initWebLLM();

// Generate text based on detected emotion
const message = await engine.generate('Continue this thought...');
```

Currently configured with the Phi-2-q4f32 model. See [web-llm models](https://github.com/mlc-ai/web-llm?tab=readme-ov-file#supported-models) for alternatives.

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 91+ | ✅ Full | Recommended |
| Firefox 89+ | ✅ Full | Good support |
| Edge 91+ | ✅ Full | Chromium-based |
| Safari 16+ | ⚠️ Limited | WebGPU support varies |

## Dependencies

- **@mlc-ai/web-llm**: ^0.2.0 - Language model inference in browser
- **@xenova/transformers**: ^2.6.0 - Transformer models for NLP tasks
- **vite**: ^5.0.0 - Build tool
- **@vitejs/plugin-basic-ssl**: ^1.1.0 - HTTPS for dev server

## Performance Optimization

To use web-llm models instead of Transformers.js (smaller, quantized models):

1. Convert SmolLM2-135M using [mlc-llm](https://github.com/mlc-ai/mlc-llm) tools
2. Add the converted model to web-llm model zoo
3. Update `modelLoader.js` to load via web-llm

This approach yields smaller bundle sizes but requires pre-conversion.

## License

This demo project is provided as-is. Check licenses for:
- SmolLM2 model (original license on HuggingFace)
- web-llm (Apache 2.0)
- Transformers.js (MIT)

## Resources

- [web-llm Documentation](https://mlc.ai/web-llm/)
- [Transformers.js Docs](https://xenova.github.io/transformers.js/)
- [SmolLM2 Model Card](https://huggingface.co/dianak12/SmolLM2-135M-emotions)
- [HuggingFace Models](https://huggingface.co)

## Next Steps

Consider extending this demo with:
- [ ] Multi-language emotion detection
- [ ] Sentiment analysis alongside emotions
- [ ] Text generation based on emotion
- [ ] Export results to CSV/JSON
- [ ] Real-time emotion tracking for long texts
- [ ] Voice input using Web Audio API
