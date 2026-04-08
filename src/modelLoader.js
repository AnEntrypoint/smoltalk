import { pipeline } from '@xenova/transformers'

let textGenPipeline = null

const MODEL_ID = 'Real-Turf/SmolRP-135M-v0.9'

export async function initTextGenModel(onProgress) {
  if (textGenPipeline) return textGenPipeline

  const modelConfig = {}
  if (onProgress) {
    modelConfig.progress_callback = (status) => {
      if (status.status === 'downloading' || status.status === 'progress') {
        onProgress(Math.round((status.progress || 0) * 100))
      }
    }
  }

  textGenPipeline = await pipeline('text-generation', MODEL_ID, modelConfig)
  return textGenPipeline
}

export async function generateText(prompt, options = {}) {
  if (!textGenPipeline) throw new Error('Model not initialized — call initTextGenModel first')
  if (!prompt || prompt.trim().length === 0) throw new Error('Prompt cannot be empty')

  const config = {
    max_new_tokens: 100,
    temperature: 0.7,
    top_p: 0.95,
    top_k: 50,
    ...options
  }

  return await textGenPipeline(prompt, config)
}

export function getStatus() {
  return { ready: textGenPipeline !== null }
}
