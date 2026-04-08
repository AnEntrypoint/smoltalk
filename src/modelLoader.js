import { pipeline, env } from '@xenova/transformers'

const BASE = window.location.origin + import.meta.env.BASE_URL + 'models/smolrp-135m/'

async function fetchOnnxModel(url) {
  const manifestUrl = BASE + 'model.onnx.manifest.json'
  const manifestResp = await fetch(manifestUrl, { cache: 'no-store' })
  if (!manifestResp.ok) {
    return fetch(url)
  }
  const { chunks, total_bytes } = await manifestResp.json()
  const parts = []
  let loaded = 0
  for (let i = 0; i < chunks; i++) {
    const r = await fetch(BASE + `model.onnx.part${i}`)
    if (!r.ok) throw new Error(`Failed to fetch model chunk part${i}: ${r.status}`)
    const buf = await r.arrayBuffer()
    parts.push(buf)
    loaded += buf.byteLength
    window.__debug.modelLoadProgress = Math.round(loaded / total_bytes * 100)
  }
  const full = new Uint8Array(total_bytes)
  let offset = 0
  for (const part of parts) {
    full.set(new Uint8Array(part), offset)
    offset += part.byteLength
  }
  return new Response(full.buffer, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
}

const chunkCache = {
  _store: new Map(),
  async match(req) {
    const url = typeof req === 'string' ? req : req.url
    if (this._store.has(url)) return this._store.get(url).clone()
    if (url.endsWith('model.onnx')) {
      const resp = await fetchOnnxModel(url)
      this._store.set(url, resp)
      return resp.clone()
    }
    return undefined
  },
  async put(req, resp) {
    const url = typeof req === 'string' ? req : req.url
    this._store.set(url, resp)
  }
}

env.localModelPath = window.location.origin + import.meta.env.BASE_URL + 'models/'
env.allowRemoteModels = false
env.useCustomCache = true
env.customCache = chunkCache

window.__debug = {}
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
