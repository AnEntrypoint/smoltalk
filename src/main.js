import { initTextGenModel, generateText, getStatus } from './modelLoader.js';

// UI Elements
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');
const tempSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temp-value');

let modelReady = false;

/**
 * Initialize the application
 */
async function initApp() {
  console.log('Initializing SmolTalk...');
  updateStatus('Loading text generation model...', 'loading');
  
  try {
    await initTextGenModel();
    modelReady = true;
    updateStatus('Ready to generate text', 'ready');
    console.log('✓ App initialized');
  } catch (error) {
    updateStatus('Failed to load model: ' + error.message, 'error');
    console.error('Initialization error:', error);
  }
}

/**
 * Generate text from prompt
 */
async function generateResponse() {
  const prompt = promptInput.value.trim();
  
  if (!prompt) {
    alert('Please enter a prompt');
    return;
  }
  
  if (!modelReady) {
    alert('Model not ready yet. Please wait.');
    return;
  }
  
  generateBtn.disabled = true;
  updateStatus('Generating...', 'loading');
  resultDiv.innerHTML = '';
  
  try {
    const temperature = parseFloat(tempSlider.value);
    const results = await generateText(prompt, {
      max_new_tokens: 150,
      temperature: temperature,
      top_p: 0.95,
    });
    
    displayResults(prompt, results);
    updateStatus('Generation complete', 'ready');
  } catch (error) {
    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    updateStatus('Generation failed', 'error');
    console.error('Generation error:', error);
  } finally {
    generateBtn.disabled = false;
  }
}

/**
 * Display generated text results
 */
function displayResults(prompt, results) {
  let html = '<div class="result-container">';
  
  html += '<div class="prompt-section">';
  html += '<h3>Prompt:</h3>';
  html += '<p class="prompt-text">' + escapeHtml(prompt) + '</p>';
  html += '</div>';
  
  html += '<div class="generation-section">';
  html += '<h3>Generated Text:</h3>';
  
  results.forEach((result, index) => {
    const text = result.generated_text;
    const generated = text.substring(prompt.length);
    
    html += '<div class="generation-item">';
    html += '<pre><code>' + escapeHtml(generated) + '</code></pre>';
    html += '</div>';
  });
  
  html += '</div>';
  html += '</div>';
  
  resultDiv.innerHTML = html;
}

/**
 * Update status display
 */
function updateStatus(message, state) {
  statusDiv.textContent = message;
  statusDiv.className = 'status ' + state;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Update temperature display
 */
tempSlider.addEventListener('input', (e) => {
  tempValue.textContent = parseFloat(e.target.value).toFixed(2);
});

/**
 * Event listeners
 */
generateBtn.addEventListener('click', generateResponse);
promptInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) generateResponse();
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', initApp);
