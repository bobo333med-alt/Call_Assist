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
    # برومبت صارم يمنع الهلوسة ويجبر النموذج على استخدام اللغة المطلوبة فقط
    sys_prompt = f"""You are an elite, highly empathetic customer support agent for a professional company. 
Your goal is to assist the customer based on the following company policies and knowledge base:
--- COMPANY MEMORY ---
{req.company_memory if req.company_memory.strip() else "No specific company rules provided. Be professional and helpful."}
--- END ---

CRITICAL LANGUAGE RULES:
1. The customer speaks {req.source_lang}.
2. You MUST reply strictly and ONLY in the language code '{req.target_lang}'. 
3. Do NOT use English, French, or any other language unless the target language code is exactly that language.
4. If the target language is 'de', your entire response must be in German. If it is 'es', it must be Spanish.

Your task:
Formulate a natural, professional, and emotionally intelligent reply in {req.target_lang}, addressing the customer's message. Ensure it sounds human and respects the company memory.

Customer's message: {req.text}
Your reply (in {req.target_lang} only):"""

    resp = await call_ai(
        [{"role":"system","content":sys_prompt},
         {"role":"user","content":req.text}],
        provider=req.provider, temperature=0.1, max_tokens=800,
    )
    
    return {
        "translated": extract_text(resp),
        "provider": resp.get("_provider"),
        "usage": extract_usage(resp),
    }
