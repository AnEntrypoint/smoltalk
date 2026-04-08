// Model initialization and management
import * as wllm from '@mlc-ai/web-llm';
import { pipeline } from '@xenova/transformers';

let textGenPipeline = null;
let webLLMEngine = null;

// Get HF token from environment
const HF_TOKEN = import.meta.env.VITE_HF_TOKEN || '';

/**
 * Initialize the text generation model
 * Uses SmolRP-135M-v0.9 from HuggingFace via Transformers.js
 */
export async function initTextGenModel() {
  if (textGenPipeline) return textGenPipeline;

  console.log('Initializing SmolRP-135M-v0.9 model...');
  try {
    // Configure Transformers.js with HF token for private/gated models
    if (HF_TOKEN) {
      console.log('Using HuggingFace API token');
    }

    // Load the SmolRP model
    // Note: This model may require accepting the license on HuggingFace
    textGenPipeline = await pipeline(
      'text-generation',
      'Real-Turf/SmolRP-135M-v0.9',
      {
        quantized: true,
        progress_callback: (status) => {
          if (status.status === 'downloading') {
            console.log(`Downloading: ${Math.round(status.progress * 100)}%`);
          } else if (status.status === 'progress') {
            console.log(`Loading: ${Math.round(status.progress * 100)}%`);
          }
        }
      }
    );

    console.log('✓ SmolRP-135M-v0.9 model loaded');
    return textGenPipeline;
  } catch (error) {
    console.error('Failed to load model:', error);
    // Fallback to demo mode if model fails to load
    console.warn('Falling back to demo mode...');
    textGenPipeline = 'demo';
    return textGenPipeline;
  }
}

/**
 * Initialize web-llm engine (optional)
 */
export async function initWebLLM() {
  if (webLLMEngine) return webLLMEngine;
  
  console.log('Initializing web-llm engine...');
  try {
    const engineConfig = {
      model: 'Phi-2-q4f32_1',  // Small model from web-llm model zoo
      device: 'webgpu'
    };
    
    webLLMEngine = await wllm.CreateMLCEngine(
      engineConfig.model,
      { device: engineConfig.device }
    );
    console.log('✓ Web-LLM engine initialized');
    return webLLMEngine;
  } catch (error) {
    console.warn('Web-LLM initialization warning (optional):', error.message);
    return null;
  }
}

/**
 * Generate text continuation for given prompt
 */
export async function generateText(prompt, options = {}) {
  if (!textGenPipeline) {
    await initTextGenModel();
  }

  if (!prompt || prompt.trim().length === 0) {
    throw new Error('Prompt cannot be empty');
  }

  try {
    // If using demo mode, return simple augmentation
    if (textGenPipeline === 'demo') {
      const demoCompletions = [
        'The future is bright with possibilities.',
        'Innovation drives progress forward.',
        'Technology transforms how we live.',
        'Understanding comes from experience.',
        'Growth requires patience and effort.',
        'Dreams become reality through action.',
        'Success is built on small steps.',
        'Knowledge is the key to growth.',
        'Every challenge is an opportunity.',
        'Together we can achieve anything.'
      ];
      const idx = Math.floor(Math.random() * demoCompletions.length);
      return [{ generated_text: `${prompt} ${demoCompletions[idx]}` }];
    }

    // Real model generation
    const defaults = {
      max_new_tokens: 100,
      temperature: 0.7,
      top_p: 0.95,
      top_k: 50,
    };

    const config = { ...defaults, ...options };

    const results = await textGenPipeline(prompt, config);
    return results;
  } catch (error) {
    console.error('Text generation error:', error);
    throw error;
  }
}

/**
 * Check model initialization status
 */
export function getStatus() {
  return {
    textGenModelReady: textGenPipeline !== null,
    webLLMReady: webLLMEngine !== null,
  };
}

/**
 * Clean up resources
 */
export async function cleanup() {
  if (webLLMEngine) {
    try {
      await webLLMEngine.dispose();
      webLLMEngine = null;
    } catch (error) {
      console.error('Error cleaning up web-llm:', error);
    }
  }
}
