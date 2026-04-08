// Model initialization and management
import * as wllm from '@mlc-ai/web-llm';

let textGenPipeline = null;
let webLLMEngine = null;

// Simple demo text generation - returns augmented text without ML
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

/**
 * Initialize the text generation model
 * For demo purposes, uses simple text augmentation
 */
export async function initTextGenModel() {
  if (textGenPipeline) return textGenPipeline;

  console.log('Initializing text generation...');
  try {
    // Demo mode: ready immediately
    textGenPipeline = true;
    console.log('✓ Text generation ready');
    return textGenPipeline;
  } catch (error) {
    console.error('Failed to initialize:', error.message);
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
    // Demo text generation: augment prompt with demo completions
    const randomIdx = Math.floor(Math.random() * demoCompletions.length);
    const continuation = demoCompletions[randomIdx];
    const generatedText = `${prompt} ${continuation}`;

    return [
      {
        generated_text: generatedText
      }
    ];
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
