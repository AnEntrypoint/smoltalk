// Model initialization and management
import * as wllm from '@mlc-ai/web-llm';
import { pipeline } from '@xenova/transformers';

let textGenPipeline = null;
let webLLMEngine = null;

/**
 * Initialize the text generation pipeline (Transformers.js)
 */
export async function initTextGenModel() {
  if (textGenPipeline) return textGenPipeline;
  
  console.log('Initializing text generation model...');
  try {
    textGenPipeline = await pipeline(
      'text-generation',
      'Real-Turf/SmolRP-135M-v0.9'
    );
    console.log('✓ SmolRP-135M model loaded');
    return textGenPipeline;
  } catch (error) {
    console.error('Failed to load text generation model:', error);
    throw error;
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
