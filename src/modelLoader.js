import { pipeline, env } from '@xenova/transformers'

env.localModelPath = window.location.origin + import.meta.env.BASE_URL + 'models/'
env.allowRemoteModels = false

let textGenPipeline = null

export async function initTextGenModel(onProgress) {
  if (textGenPipeline) return textGenPipeline

  const config = {}
  if (onProgress) {
    config.progress_callback = (status) => {
      if (status.status === 'downloading' || status.status === 'progress') {
        onProgress(Math.round((status.progress || 0) * 100))
      }
    }
  }

  textGenPipeline = await pipeline('text-generation', 'smolrp-135m', config)
  return textGenPipeline
}

export async function generateText(prompt, options = {}) {
  if (!textGenPipeline) throw new Error('Model not initialized — call initTextGenModel first')
  if (!prompt || prompt.trim().length === 0) throw new Error('Prompt cannot be empty')

  const result = await textGenPipeline(prompt, {
    max_new_tokens: 100,
    temperature: 0.7,
    top_p: 0.95,
    top_k: 50,
    ...options
  })

  return result
}

export function getStatus() {
  return { ready: textGenPipeline !== null }
}
