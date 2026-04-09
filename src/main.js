import { initTextGenModel, generateText, getStatus } from './modelLoader.js'

const promptInput = document.getElementById('prompt-input')
const generateBtn = document.getElementById('generate-btn')
const resultDiv = document.getElementById('result')
const statusDiv = document.getElementById('status')
const tempSlider = document.getElementById('temperature')
const tempValue = document.getElementById('temp-value')

function updateStatus(message, state) {
  statusDiv.textContent = message
  statusDiv.className = 'status ' + state
}

function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

async function initApp() {
  updateStatus('Loading SmolRP-135M-v0.9...', 'loading')

  try {
    await initTextGenModel((pct) => updateStatus('Loading model: ' + pct + '%', 'loading'))
    updateStatus('Ready', 'ready')
    generateBtn.disabled = false
  } catch (error) {
    updateStatus('FAILED: ' + error.message, 'error')
    throw error
  }
}

async function generateResponse() {
  const prompt = promptInput.value.trim()
  if (!prompt) return
  if (!getStatus().ready) return

  generateBtn.disabled = true
  updateStatus('Generating...', 'loading')
  resultDiv.innerHTML = ''

  try {
    const results = await generateText(prompt, {
      max_new_tokens: 150,
      temperature: parseFloat(tempSlider.value),
      top_p: 0.95,
    })

    let html = '<div class="result-container">'
    html += '<div class="prompt-section"><h3>Prompt:</h3>'
    html += '<p class="prompt-text">' + escapeHtml(prompt) + '</p></div>'
    html += '<div class="generation-section"><h3>Generated:</h3>'
    for (const result of results) {
      const msgs = result.generated_text
      const assistantMsg = Array.isArray(msgs) ? msgs.filter(m => m.role === 'assistant').map(m => m.content).join('') : msgs.substring(prompt.length)
      html += '<div class="generation-item"><pre><code>' + escapeHtml(assistantMsg) + '</code></pre></div>'
    }
    html += '</div></div>'
    resultDiv.innerHTML = html
    updateStatus('Done', 'ready')
  } catch (error) {
    resultDiv.innerHTML = '<div class="error">Error: ' + escapeHtml(error.message) + '</div>'
    updateStatus('Generation failed: ' + error.message, 'error')
    throw error
  } finally {
    generateBtn.disabled = false
  }
}

tempSlider.addEventListener('input', (e) => {
  tempValue.textContent = parseFloat(e.target.value).toFixed(2)
})
generateBtn.addEventListener('click', generateResponse)
promptInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) generateResponse()
})

generateBtn.disabled = true
initApp()
