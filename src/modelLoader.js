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
 * Model is pre-cached via GitHub Actions and served from GitHub Pages
 */
export async function initTextGenModel() {
  if (textGenPipeline) return textGenPipeline;

  console.log('Initializing SmolRP-135M-v0.9 model...');
  try {
    // Try to load from local cache first (GitHub Pages)
    // Falls back to HuggingFace if cache doesn't exist
    const modelConfig = {
      progress_callback: (status) => {
        if (status.status === 'downloading') {
          const percent = Math.round((status.progress || 0) * 100);
          console.log(`Downloading: ${percent}%`);
        } else if (status.status === 'progress') {
          const percent = Math.round((status.progress || 0) * 100);
          console.log(`Loading: ${percent}%`);
        }
      }
    };

    // If running on GitHub Pages, try cached location first
    if (typeof window !== 'undefined' &&
        (window.location.hostname.includes('github.io') ||
         window.location.pathname.includes('smoltalk'))) {
      try {
        console.log('Attempting to load from GitHub Pages cache...');
        textGenPipeline = await pipeline(
          'text-generation',
          '/smoltalk/models/smolrp-135m/',
          modelConfig
        );
        console.log('✓ Model loaded from GitHub Pages cache');
        return textGenPipeline;
      } catch (cacheError) {
        console.warn('Cache load failed, falling back to HuggingFace...');
      }
    }

    // Load from HuggingFace (with token if available)
    const modelId = 'Real-Turf/SmolRP-135M-v0.9';
    textGenPipeline = await pipeline(
      'text-generation',
      modelId,
      modelConfig
    );

    console.log('✓ SmolRP-135M-v0.9 model loaded from HuggingFace');
    return textGenPipeline;
  } catch (error) {
    console.error('Failed to load model:', error.message);
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
