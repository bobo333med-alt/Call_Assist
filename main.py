import os, asyncio, logging
from typing import Literal
import httpx

logger = logging.getLogger("ai_router")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OR_URL   = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT, MAX_RETRIES = 30, 2

def _env(k: str, d: str = "") -> str:
    return os.getenv(k, d).strip()

async def _post(url: str, headers: dict, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                r = await c.post(url, headers=headers, json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"{r.status_code}: {r.text[:200]}"
                    await asyncio.sleep(0.6 * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                last = str(e)
                await asyncio.sleep(0.6 * (2 ** attempt))
        raise RuntimeError(f"AI HTTP failed: {last}")

async def _groq(messages, model=None, **kw):
    key = _env("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY missing")
    data = await _post(GROQ_URL, {"Authorization": f"Bearer {key}"}, {
        # تحديث النموذج ليدعم 12 لغة منها الإيطالية والإسبانية
        "model": model or _env("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "messages": messages,
        "temperature": kw.get("temperature", 0.3),
        "max_tokens": kw.get("max_tokens", 800),
    })
    data["_provider"] = "groq"
    return data

async def _openrouter(messages, model=None, **kw):
    key = _env("OPENROUTER_API_KEY")
    if not key: raise RuntimeError("OPENROUTER_API_KEY missing")
    data = await _post(OR_URL, {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": _env("APP_REFERER", "https://call-assist.app"),
        "X-Title": "Call-Assist",
    }, {
        # استخدام Qwen3.5 الذي يدعم 201 لغة وله قدرة عالية على التقمص
        "model": model or _env("OPENROUTER_MODEL", "qwen/qwen3.5-9b"),
        "messages": messages,
        "temperature": kw.get("temperature", 0.4),
        "max_tokens": kw.get("max_tokens", 800),
    })
    data["_provider"] = "openrouter"
    return data

async def call_ai(messages, provider: Literal["auto","groq","openrouter"] = None, **kw):
    provider = (provider or _env("AI_PROVIDER", "auto")).lower()
    if provider == "groq": return await _groq(messages, **kw)
    if provider == "openrouter": return await _openrouter(messages, **kw)
    try:
        return await _groq(messages, **kw)
    except Exception as e:
        logger.warning(f"Groq failed ({e}) → OpenRouter")
        return await _openrouter(messages, **kw)

def extract_text(resp):
    try: return (resp["choices"][0]["message"]["content"] or "").strip()
    except: return ""

def extract_usage(resp):
    u = resp.get("usage") or {}
    return {"prompt_tokens": int(u.get("prompt_tokens",0)),
            "completion_tokens": int(u.get("completion_tokens",0)),
            "total_tokens": int(u.get("total_tokens",0))}
