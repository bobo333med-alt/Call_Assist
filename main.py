import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ai_router import call_ai, extract_text, extract_usage, get_client

app = FastAPI(title="AI Autonomous Call Center & Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    client = get_client()
    if not client.is_closed:
        await client.aclose()

# --- نماذج البيانات ---

class AIRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    company_memory: str = ""
    mode: str = "translate"
    provider: str = "auto"

class ChatTurn(BaseModel):
    role: str
    content: str

class AgentChatRequest(BaseModel):
    history: List[ChatTurn]
    agent_name: str = "سارة"
    agent_lang: str = "ar-TN"
    company_memory: str = ""
    customer_name: Optional[str] = ""
    customer_note: Optional[str] = ""
    provider: str = "auto"

class SummarizeRequest(BaseModel):
    history: List[ChatTurn]
    provider: str = "auto"

# --- مسارات الصفحات (الرابط الأساسي يفتح التطبيق مباشرة) ---

@app.get("/")
@app.get("/call_assist.html")
def open_app_directly():
    return FileResponse("call_assist.html")

@app.get("/admin.html")
def get_admin():
    return FileResponse("admin.html")

@app.get("/agent_call.html")
def get_agent_call():
    return FileResponse("agent_call.html")

@app.get("/health")
def health_check():
    return {"status": "online", "engine": "Ultra-Fast AI Call Center"}

# --- مسار الوكيل الصوتي للهاتف الحقيقي (Vapi Custom LLM Endpoint) ---

@app.post("/vapi/chat/completions")
async def vapi_custom_llm(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    
    resp = await call_ai(
        messages,
        provider="auto",
        temperature=0.25,
        max_tokens=200,
    )
    reply_text = extract_text(resp)

    return {
        "id": "call-turn-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "ai-call-center-agent",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_text
                },
                "finish_reason": "stop"
            }
        ]
    }

# --- مسار الوكيل للمتصفح ---

@app.post("/ai/agent-chat")
async def ai_agent_chat(req: AgentChatRequest):
    lang_rules = {
        "ar-TN": "تحدث بالدارجة التونسية المهذبة والواضحة والمفهومة جداً (مثل: عسلامة، مرحبا بيك، يعيشك، تفضل، شنوة نجم نعاونك).",
        "ar": "تحدث باللغة العربية الفصحى المبسطة والودودة.",
        "fr": "Parle uniquement en français professionnel, chaleureux et naturel.",
        "en": "Speak strictly in natural, warm, and professional English."
    }
    selected_lang = lang_rules.get(req.agent_lang, lang_rules["ar-TN"])

    customer_info = ""
    if req.customer_name:
        customer_info += f"\n- اسم الحريف: {req.customer_name}"
    if req.customer_note:
        customer_info += f"\n- ملاحظة عن الحريف: {req.customer_note}"

    sys_prompt = f"""أنت وكيل مركز نداء ذكي (AI Call Center Agent) محترف جداً واسمك '{req.agent_name}'.
مهمتك هي الرد على الحرفاء أو الاتصال بهم وإقناعهم وخدمتهم بدقة بناءً على ذاكرة الشركة فقط.

=== ذاكرة ومعلومات الشركة ===
{req.company_memory if req.company_memory.strip() else "لا توجد معطيات محددة حالياً. رحب بالحريف واسأله بلطف كيف يمكنك مساعدته."}
=== نهاية الذاكرة ===
{customer_info}

قواعد صارمة:
1. اللهجة واللغة: {selected_lang}
2. اجعل ردك قصيراً ومركّزاً (جملة أو جملتين فقط) ليكون الحوار سريعاً وطبيعياً كالبشر.
3. لا تستخدم أبداً رموز مثل (* أو # أو - أو إيموجي).
4. إذا سألك الحريف عن شيء غير موجود في ذاكرة الشركة، لا تؤلف أبداً! قل له أنك ستسجل طلبه ليتصل به المختص.
5. إذا انتهت المكالمة وودعك الحريف، أضف [END_CALL] في نهاية ردك."""

    messages = [{"role": "system", "content": sys_prompt}]
    for turn in req.history:
        messages.append({"role": turn.role, "content": turn.content})

    if not req.history:
        messages.append({
            "role": "user",
            "content": "الحريف رد على الهاتف الآن وقال: ألو؟ (ابدأ المكالمة بالترحيب والتعريف بنفسك وبسبب الاتصال باختصار)."
        })

    try:
        start_time = time.perf_counter()
        resp = await call_ai(messages, provider=req.provider, temperature=0.25, max_tokens=250)
        elapsed_sec = time.perf_counter() - start_time
        usage = extract_usage(resp)
        tps = round(usage["completion_tokens"] / elapsed_sec, 1) if elapsed_sec > 0 else 0

        raw_reply = extract_text(resp)
        should_end = "[END_CALL]" in raw_reply
        clean_reply = raw_reply.replace("[END_CALL]", "").strip()

        return {
            "reply": clean_reply,
            "end_call": should_end,
            "provider": resp.get("_provider"),
            "latency_ms": round(elapsed_sec * 1000),
            "speed_tokens_per_sec": tps,
            "usage": usage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/summarize-call")
async def ai_summarize_call(req: SummarizeRequest):
    if not req.history:
        return {"summary": "لم يتم التحدث في المكالمة"}
    convo = "\n".join([f"{'الحريف' if t.role=='user' else 'الوكيل'}: {t.content}" for t in req.history])
    sys_prompt = "لخص نتيجة هذه المكالمة في جملة واحدة قصيرة وواضحة بالعربية لصاحب مركز النداء:\n\n" + convo
    try:
        resp = await call_ai([{"role": "system", "content": sys_prompt}], provider=req.provider, temperature=0.1, max_tokens=80)
        return {"summary": extract_text(resp)}
    except Exception as e:
        return {"summary": "تعذر التلخيص: " + str(e)}

@app.post("/ai/translate")
async def ai_translate(req: AIRequest):
    sys_prompt = f"""You are an elite, ultra-fast live call translator.
Company context: {req.company_memory if req.company_memory.strip() else "General professional support."}
Translate/formulate from '{req.source_lang}' strictly into '{req.target_lang}'. Output ONLY the final text."""
    try:
        resp = await call_ai([{"role": "system", "content": sys_prompt}, {"role": "user", "content": req.text}], provider=req.provider, temperature=0.1, max_tokens=400)
        return {"translated": extract_text(resp), "provider": resp.get("_provider"), "usage": extract_usage(resp)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
