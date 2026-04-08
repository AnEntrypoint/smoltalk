// Model initialization and management
import * as wllm from '@mlc-ai/web-llm';
import { pipeline } from '@xenova/transformers';

let emotionPipeline = null;
let webLLMEngine = null;

/**
 * Initialize the emotion classification pipeline (Transformers.js)
 */
export async function initEmotionModel() {
  if (emotionPipeline) return emotionPipeline;
  
  console.log('Initializing emotion classification model...');
  try {
    emotionPipeline = await pipeline(
      'text-classification',
      'dianak12/SmolLM2-135M-emotions'
    );
    console.log('✓ Emotion model loaded');
    return emotionPipeline;
  } catch (error) {
    console.error('Failed to load emotion model:', error);
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
 * Classify emotion for given text
 */
export async function classifyEmotion(text) {
  if (!emotionPipeline) {
    await initEmotionModel();
  }
  
  if (!text || text.trim().length === 0) {
    throw new Error('Text cannot be empty');
  }
  
  try {
    const results = await emotionPipeline(text, {
      top_k: 5  // Get top 5 emotions
    });
    return results;
  } catch (error) {
    console.error('Emotion classification error:', error);
    throw error;
  }
}

/**
 * Check model initialization status
 */
export function getStatus() {
  return {
    emotionModelReady: emotionPipeline !== null,
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
