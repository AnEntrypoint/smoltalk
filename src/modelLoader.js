import { pipeline, env } from '@huggingface/transformers'

const BASE = window.location.origin + import.meta.env.BASE_URL + 'models/smolrp-135m/'
const DB_NAME = 'smoltalk'
const DB_STORE = 'onnx'

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1)
    req.onupgradeneeded = e => e.target.result.createObjectStore(DB_STORE)
    req.onsuccess = e => resolve(e.target.result)
    req.onerror = e => reject(e.target.error)
  })
}

function idbGet(db, key) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(DB_STORE, 'readonly').objectStore(DB_STORE).get(key)
    req.onsuccess = e => resolve(e.target.result)
    req.onerror = e => reject(e.target.error)
  })
}

function idbPut(db, key, val) {
  return new Promise((resolve, reject) => {
    const req = db.transaction(DB_STORE, 'readwrite').objectStore(DB_STORE).put(val, key)
    req.onsuccess = () => resolve()
    req.onerror = e => reject(e.target.error)
  })
}

async function fetchOnnxModel(url) {
  const manifestResp = await fetch(BASE + 'model.onnx.manifest.json', { cache: 'no-store' })
  if (!manifestResp.ok) throw new Error('Manifest fetch failed: ' + manifestResp.status)
  const { chunks, total_bytes } = await manifestResp.json()
  const parts = []
  let loaded = 0
  for (let i = 0; i < chunks; i++) {
    const r = await fetch(BASE + 'model.onnx.part' + i)
    if (!r.ok) throw new Error('Failed to fetch chunk part' + i + ': ' + r.status)
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
  return full.buffer
}

let _db = null
async function getDB() {
  if (!_db) _db = await openDB()
  return _db
}

const chunkCache = {
  async match(req) {
    const url = typeof req === 'string' ? req : req.url
    const db = await getDB()
    const cached = await idbGet(db, url)
    if (cached) return new Response(cached, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
    if (url.includes('/onnx/') || url.endsWith('.onnx')) {
      const buf = await fetchOnnxModel(url)
      await idbPut(db, url, buf)
      return new Response(buf, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } })
    }
    return undefined
  },
  async put(req, resp) {
    const url = typeof req === 'string' ? req : req.url
    const buf = await resp.clone().arrayBuffer()
    const db = await getDB()
    await idbPut(db, url, buf)
  }
}

env.localModelPath = window.location.origin + import.meta.env.BASE_URL + 'models/'
env.allowRemoteModels = false
env.useBrowserCache = false
env.useCustomCache = true
env.customCache = chunkCache

window.__debug = {}
let textGenPipeline = null

export async function initTextGenModel(onProgress) {
  if (textGenPipeline) return textGenPipeline
  const config = { device: 'webgpu', dtype: 'q4f16' }
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
    repetition_penalty: 1.3,
    ...options
  })
  return result
}

export function getStatus() {
  return { ready: textGenPipeline !== null }
}
