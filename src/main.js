import { initEmotionModel, classifyEmotion, getStatus } from './modelLoader.js';

// UI Elements
const inputText = document.getElementById('input-text');
const analyzeBtn = document.getElementById('analyze-btn');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const statusDiv = document.getElementById('status');

let modelReady = false;

/**
 * Initialize the application
 */
async function initApp() {
  console.log('Initializing application...');
  updateStatus('Loading emotion model...', 'loading');
  
  try {
    await initEmotionModel();
    modelReady = true;
    updateStatus('Ready for analysis', 'ready');
    console.log('✓ App initialized');
  } catch (error) {
    updateStatus('Failed to load model: ' + error.message, 'error');
    console.error('Initialization error:', error);
  }
}

/**
 * Analyze text for emotions
 */
async function analyzeText() {
  const text = inputText.value.trim();
  
  if (!text) {
    alert('Please enter some text');
    return;
  }
  
  if (!modelReady) {
    alert('Model not ready yet. Please wait.');
    return;
  }
  
  analyzeBtn.disabled = true;
  updateStatus('Analyzing...', 'loading');
  resultDiv.innerHTML = '';
  
  try {
    const emotions = await classifyEmotion(text);
    displayResults(emotions);
    updateStatus('Analysis complete', 'ready');
  } catch (error) {
    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    updateStatus('Analysis failed', 'error');
    console.error('Analysis error:', error);
  } finally {
    analyzeBtn.disabled = false;
  }
}

/**
 * Display emotion classification results
 */
function displayResults(emotions) {
  let html = '<div class="results-container">';
  
  emotions.forEach((emotion, index) => {
    const label = emotion.label;
    const score = (emotion.score * 100).toFixed(2);
    const barWidth = emotion.score * 100;
    
    html += `
      <div class="emotion-item">
        <div class="emotion-header">
          <span class="emotion-label">${label}</span>
          <span class="emotion-score">${score}%</span>
        </div>
        <div class="emotion-bar">
          <div class="emotion-bar-fill" style="width: ${barWidth}%"></div>
        </div>
      </div>
    `;
  });
  
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
 * Event listeners
 */
analyzeBtn.addEventListener('click', analyzeText);
inputText.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') analyzeText();
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', initApp);
