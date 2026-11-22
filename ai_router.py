import os
import asyncio
import logging
from typing import Literal, Optional
import httpx

logger = logging.getLogger("ai_router")

GROQ_URL = "https://groq.com"
OR_URL = "https://openrouter.ai"

# مهلة زمنية قصيرة (8 ثوانٍ) لضمان عدم تجمد المكالمة والانتقال الفوري للبديل
TIMEOUT = 8.0
MAX_RETRIES = 2

# عميل اتصال دائم فائق السرعة (Connection Pooling)
_http_client: Optional[httpx.AsyncClient] = None

def get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )
    return _http_client

def _env(k: str, d: str = "") -> str:
    return os.getenv(k, d).strip()

async def _post(url: str, headers: dict, payload: dict) -> dict:
    client = get_client()
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:150]}"
                await asyncio.sleep(0.3 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"AI Request Failed: {last_err}")

async def _groq(messages, model=None, **kw):
    key = _env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing")
    # استخدام أسرع نموذج في العالم للاستجابة الفورية
    selected_model = model or _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    data = await _post(
        GROQ_URL,
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {
            "model": selected_model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.2),
            "max_tokens": kw.get("max_tokens", 350),
        }
    )
    data["_provider"] = f"groq ({selected_model})"
    return data

async def _openrouter(messages, model=None, **kw):
    key = _env("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    selected_model = model or _env("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    data = await _post(
        OR_URL,
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://call-assist.app",
            "X-Title": "AI-Call-Center",
        },
        {
            "model": selected_model,
            "messages": messages,
            "temperature": kw.get("temperature", 0.2),
            "max_tokens": kw.get("max_tokens", 350),
        }
    )
    data["_provider"] = f"openrouter ({selected_model})"
    return data

async def call_ai(messages, provider: Literal["auto", "groq", "openrouter"] = None, **kw):
    provider = (provider or _env("AI_PROVIDER", "auto")).lower()
    if provider == "groq":
        return await _groq(messages, **kw)
    if provider == "openrouter":
        return await _openrouter(messages, **kw)
    
    # الوضع التلقائي الذكي: Groq أولاً لسرعته الخارقة، ثم OpenRouter كحماية
    try:
        return await _groq(messages, **kw)
    except Exception as e:
        logger.warning(f"Groq fallback triggered: {e}")
        return await _openrouter(messages, **kw)

def extract_text(resp: dict) -> str:
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""

def extract_usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens", 0)),
        "completion_tokens": int(u.get("completion_tokens", 0)),
        "total_tokens": int(u.get("total_tokens", 0)),
    }
