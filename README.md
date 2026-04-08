# SmolTalk - Text Generation Demo

A browser-based text generation application powered by [Transformers.js](https://xenova.github.io/transformers.js/) and [web-llm](https://github.com/mlc-ai/web-llm).

**Note**: Currently configured with GPT-2 for broad browser compatibility. Can be swapped to [SmolRP-135M-v0.9](https://huggingface.co/Real-Turf/SmolRP-135M-v0.9) or other ONNX-optimized models.

## Features

- 🚀 **In-Browser Processing**: No server required, all inference runs in your browser
- 📝 **Text Generation**: Continue prompts with contextual, creative text completions
- ⚡ **Fast Inference**: Optimized for 135M parameter model
- 🎨 **Modern UI**: Clean interface with temperature control for output variety
- 🔧 **web-llm Integration**: Full setup with web-llm engine for extensibility

## Architecture

This project combines two powerful libraries:

1. **Transformers.js** - Loads the HuggingFace SmolRP-135M model directly in the browser
2. **web-llm** - Provides LLM capabilities and serves as the foundation for this application

The text generation runs entirely client-side using ONNX.js for inference.

## Installation

### Prerequisites
- Node.js 16+ and npm
- Modern browser with WebGPU or WebAssembly support

### Setup

```bash
# Clone or download the project
cd smoltalk

# Install dependencies
npm install

# Start development server
npm run dev
```

The dev server will start at `https://localhost:5173` (Vite uses HTTPS for SharedArrayBuffer support).

## Usage

1. Open the app in your browser
2. Wait for the text generation model to load (first run downloads ~350MB)
3. Enter a prompt in the textarea
4. Adjust temperature slider for creativity (0.1 = deterministic, 2.0 = very creative)
5. Click "Generate" or press Ctrl+Enter
6. View the generated text continuation

### Example Prompts

- "The sun rose over the mountain and"
- "In a galaxy far far away, there was"
- "She opened the ancient book and found"
- "The best time to visit Paris is"

### Temperature Guide

- **0.1 - 0.5**: Focused, coherent, predictable outputs
- **0.5 - 0.9**: Balanced creativity and coherence
- **0.9 - 1.5**: Creative, varied outputs
- **1.5 - 2.0**: Very creative, may be less coherent

## Project Structure

```
smoltalk/
├── index.html          # Main UI
├── package.json        # Dependencies
├── vite.config.js      # Build configuration
├── src/
│   ├── main.js         # Application logic
│   └── modelLoader.js  # Model initialization
└── dist/              # Built output (after `npm run build`)
```

## Model Information

**Current Model**: GPT-2
- **Source**: [OpenAI GPT-2](https://huggingface.co/gpt2)
- **Parameters**: 124M
- **Task**: Text Generation / Language Modeling
- **Format**: ONNX-optimized for browser inference

### Using Custom Models

Edit `src/modelLoader.js` to use different models:

```javascript
const modelId = 'your-username/your-model-name';
textGenPipeline = await pipeline('text-generation', modelId);
```

Recommended browser-compatible models:
- `gpt2` - Fast, lightweight
- `distilgpt2` - Even smaller
- `facebook/opt-125m` - Larger capacity
- Custom ONNX-converted models

### Output Format

The model returns generated text continuation:

```javascript
[
  { generated_text: "Your prompt here continued text..." },
  { generated_text: "Alternative continuation..." }
]
```

## Performance Notes

- **First Load**: 10-30 seconds (downloads ~400MB model files from HuggingFace)
- **Subsequent Loads**: Cached in browser (localStorage and IndexedDB)
- **Generation Speed**: 1-5 seconds per output (depends on token count and hardware)
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

### Slow generation
- Close other browser tabs to free up resources
- Reduce temperature for faster outputs
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

Customize generation parameters:

```javascript
const results = await generateText(prompt, {
  max_new_tokens: 150,      // Maximum tokens to generate
  temperature: 0.7,          // Sampling temperature
  top_p: 0.95,               // Nucleus sampling
  top_k: 50,                 // Top-k sampling
});
```

### Use Different Model

```javascript
// Use a different model from HuggingFace
textGenPipeline = await pipeline(
  'text-generation',
  'username/model-name'
);
```

## Extended Usage with web-llm

The project includes optional web-llm integration for additional capabilities:

```javascript
import { initWebLLM } from './modelLoader.js';

// Initialize web-llm engine
const engine = await initWebLLM();

// Use web-llm models for different tasks
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

## License

This demo project is provided as-is. Check licenses for:
- SmolRP-135M model (original license on HuggingFace)
- web-llm (Apache 2.0)
- Transformers.js (MIT)

## Resources

- [web-llm Documentation](https://mlc.ai/web-llm/)
- [Transformers.js Docs](https://xenova.github.io/transformers.js/)
- [SmolRP Model Card](https://huggingface.co/Real-Turf/SmolRP-135M-v0.9)
- [HuggingFace Models](https://huggingface.co)

## Next Steps

Consider extending this demo with:
- [ ] Multiple model support
- [ ] Output sampling strategies
- [ ] Conversation history
- [ ] Export generations to file
- [ ] Fine-tuning on specific datasets
- [ ] Voice input/output using Web Audio API
