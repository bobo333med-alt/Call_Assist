import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ai_router import call_ai, extract_text

app = FastAPI(title="Call-Assist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class TurnRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    company_memory: str
    mode: str = "translate"

class SummaryRequest(BaseModel):
    transcript: str
    company_memory: str

@app.get("/")
def read_root(): return {"status": "Call-Assist Backend is Running"}

@app.get("/call_assist.html")
def get_agent_app(): return FileResponse("call_assist.html")

@app.get("/admin.html")
def get_admin_app(): return FileResponse("admin.html")

@app.post("/ai/process_turn")
async def process_turn(req: TurnRequest):
    # برومبت ذكي يطلب الترجمة، تحليل المشاعر، واقتراح رد (Coach Card) في نفس الوقت بصيغة JSON
    sys_prompt = f"""You are an elite, empathetic customer support AI. 
Company Memory: {req.company_memory if req.company_memory else "No specific rules."}

Your tasks based on the customer's message:
1. Translate the message from {req.source_lang} to {req.target_lang}.
2. Analyze customer sentiment (choose ONE: "happy", "neutral", "angry").
3. Suggest a short coaching tip for the human agent in Arabic (max 15 words) to handle this situation.

You MUST respond ONLY with a valid JSON object matching this schema:
{{
  "translation": "Translated text here strictly in {req.target_lang}",
  "sentiment": "happy|neutral|angry",
  "coach_tip": "نصيحة للوكيل بالعربية"
}}"""

    try:
        resp = await call_ai(
            [{"role":"system","content":sys_prompt}, {"role":"user","content":req.text}],
            temperature=0.1, max_tokens=500, json_mode=True
        )
        raw_text = extract_text(resp)
        data = json.loads(raw_text)
        return {
            "translation": data.get("translation", ""),
            "sentiment": data.get("sentiment", "neutral"),
            "coach_tip": data.get("coach_tip", ""),
            "provider": resp.get("_provider")
        }
    except Exception as e:
        return {"translation": "Error", "sentiment": "neutral", "coach_tip": "Server Error", "error": str(e)}

@app.post("/ai/summary")
async def generate_summary(req: SummaryRequest):
    sys_prompt = f"""You are an AI call center supervisor. Summarize this call transcript.
Company Memory context: {req.company_memory}

Respond ONLY with a valid JSON object:
{{
  "summary": "Brief 2-sentence summary of the call.",
  "action_items": ["Action 1", "Action 2"],
  "resolution": "Resolved|Pending|Escalated"
}}"""
    try:
        resp = await call_ai(
            [{"role":"system","content":sys_prompt}, {"role":"user","content":req.transcript}],
            temperature=0.1, max_tokens=300, json_mode=True
        )
        return json.loads(extract_text(resp))
    except Exception as e:
        return {"summary": "Failed to generate summary.", "action_items": [], "resolution": "Unknown"}
