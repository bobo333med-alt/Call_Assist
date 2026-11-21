import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ai_router import call_ai, extract_text, extract_usage

app = FastAPI(title="Call-Assist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    company_memory: str = ""
    mode: str = "translate"
    provider: str = "auto"

@app.get("/")
def read_root():
    return {"status": "Call-Assist Backend is Running"}

@app.get("/call_assist.html")
def get_call_assist():
    return FileResponse("call_assist.html")

@app.get("/admin.html")
def get_admin():
    return FileResponse("admin.html")

@app.post("/ai/translate")
async def ai_translate(req: AIRequest):
    if req.mode == "suggest":
        sys_prompt = (
            f"You are a professional call-center assistant. "
            f"Customer speaks {req.source_lang}, agent replies in {req.target_lang}. "
            f"Suggest ONE concise reply for the agent in {req.target_lang}, "
            f"then a translation of that reply back to {req.source_lang} on a new line prefixed '↳ '. "
            f"Respect company policy."
        )
    else:
        sys_prompt = (
            f"You are a precise translation engine for a call center. "
            f"Translate from {req.source_lang} to {req.target_lang}. "
            f"Output ONLY the translation. No quotes, no explanations."
        )
        
    if req.company_memory.strip():
        sys_prompt += f"\n\nCompany context (terminology only):\n{req.company_memory}"

    resp = await call_ai(
        [{"role":"system","content":sys_prompt},
         {"role":"user","content":req.text}],
        provider=req.provider, temperature=0.1, max_tokens=600,
    )
    
    return {
        "translated": extract_text(resp),
        "provider": resp.get("_provider"),
        "usage": extract_usage(resp),
    }
