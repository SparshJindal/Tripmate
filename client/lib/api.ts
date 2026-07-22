export const ROUTER_URL =
  process.env.NEXT_PUBLIC_ROUTER_URL ?? "https://tripmate-1-0qh8.onrender.com"
export const VISION_URL =
  process.env.NEXT_PUBLIC_VISION_URL ?? "https://tripmate-w0hm.onrender.com"

export type Strategy = "semantic" | "judge"

export type ModelInfo = { name: string; provider: string; model: string }

export type Candidate = {
  name: string
  provider: string
  ok: boolean
  text: string | null
  latency_ms: number | null
  tokens: number | null
  error: string | null
  agreement?: number
  was_selected?: boolean
}

export type AskResult = {
  answer: string | null
  chosen: string | null
  strategy: string
  candidates: Candidate[]
  cached: boolean
  request_id?: string
  judge?: { judge_model: string | null; reason: string }
}

export type ModelStats = {
  model: string
  provider: string
  calls: number
  successes: number
  wins: number
  win_rate: number
  avg_latency_ms: number | null
  avg_agreement: number | null
}

export type SceneResult = {
  ok: boolean
  answer: string | null
  provider: string | null
  model?: string | null
  latency_ms?: number | null
  tokens?: number | null
  cached?: boolean
  errors?: string[]
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${ROUTER_URL}/health`)
    return res.ok
  } catch {
    return false
  }
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${ROUTER_URL}/models`)
  if (!res.ok) throw new Error(`GET /models failed: ${res.status}`)
  return res.json()
}

export async function fetchMetrics(): Promise<ModelStats[]> {
  const res = await fetch(`${ROUTER_URL}/metrics/models`)
  if (!res.ok) throw new Error(`GET /metrics/models failed: ${res.status}`)
  return res.json()
}

export async function ask(
  prompt: string,
  strategy: Strategy,
): Promise<AskResult> {
  const res = await fetch(`${ROUTER_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, strategy }),
  })
  if (!res.ok) throw new Error(`POST /ask failed: ${res.status}`)
  return res.json()
}

export async function askScene(
  image: File | Blob,
  question: string,
): Promise<SceneResult> {
  const form = new FormData()
  form.append("image", image, "frame.jpg")
  form.append("question", question)
  const res = await fetch(`${VISION_URL}/scene`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw new Error(`POST /scene failed: ${res.status}`)
  return res.json()
}

export async function streamAsk(
  prompt: string,
  onDelta: (token: string) => void,
  onError: (message: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${ROUTER_URL}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`POST /stream failed: ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split("\n\n")
    buffer = events.pop() ?? ""
    for (const event of events) {
      const line = event.split("\n").find((l) => l.startsWith("data:"))
      if (!line) continue
      const data = line.slice("data:".length).trim()
      if (data === "[DONE]") return
      try {
        const parsed = JSON.parse(data)
        if (parsed.delta) onDelta(parsed.delta)
        if (parsed.error) onError(parsed.error)
      } catch {
        // ignore malformed chunks
      }
    }
  }
}
