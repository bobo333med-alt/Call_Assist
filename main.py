import os
from typing import List, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ai_router import call_ai, extract_text, extract_usage

app = FastAPI(title="AI Call Center Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- نماذج البيانات (Schemas) ---

class AIRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    company_memory: str = ""
    mode: str = "translate"
    provider: str = "auto"

class ChatTurn(BaseModel):
    role: str  # "user" (الحريف) أو "assistant" (الوكيل)
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

# --- مسارات الصفحات (Frontend Routes) ---

@app.get("/")
def read_root():
    return {"status": "AI Call Center Backend is Running"}

@app.get("/call_assist.html")
def get_call_assist():
    return FileResponse("call_assist.html")

@app.get("/admin.html")
def get_admin():
    return FileResponse("admin.html")

# --- مسار الوكيل الذكي للمحادثة والمكالمات (Autonomous Agent) ---

@app.post("/ai/agent-chat")
async def ai_agent_chat(req: AgentChatRequest):
    lang_instructions = {
        "ar-TN": "تحدث بالدارجة التونسية المهذبة والواضحة (مزيج طبيعي مفهوم في تونس، مثل: عسلامة، مرحبا بيك، يعيشك، تفضل).",
        "ar": "تحدث باللغة العربية الفصحى المبسطة والودودة.",
        "fr": "Parle uniquement en français professionnel, chaleureux et naturel.",
        "en": "Speak strictly in natural, warm, and professional English."
    }
    selected_lang_rule = lang_instructions.get(req.agent_lang, lang_instructions["ar-TN"])

    customer_context = ""
    if req.customer_name:
        customer_context += f"\n- اسم الحريف المتصل به: {req.customer_name}"
    if req.customer_note:
        customer_context += f"\n- معلومة إضافية عن الحريف: {req.customer_note}"

    sys_prompt = f"""أنت وكيل مركز نداء (Call Center Agent) محترف واسمك '{req.agent_name}'.
مهمتك هي الرد على الحرفاء أو الاتصال بهم لخدمتهم وإقناعهم بناءً على معطيات الشركة فقط.

--- ذاكرة ومعلومات الشركة (Knowledge Base) ---
{req.company_memory if req.company_memory.strip() else "لا توجد معلومات محددة. كن مهذباً واسأل الحريف كيف يمكنك مساعدته."}
--- نهاية الذاكرة ---
{customer_context}

قواعد صارمة جداً (يجب الالتزام بها حرفياً):
1. اللغة واللهجة: {selected_lang_rule}
2. أسلوب الكلام الصوتي: ردودك ستُقرأ صوتياً عبر الهاتف! اجعل ردودك قصيرة ومباشرة (جملة أو جملتين كحد أقصى في كل رد) لكي لا يمل الحريف.
3. ممنوع التنسيق النصي: لا تستخدم أبداً رموز مثل (* أو # أو - أو قوائم نقطية أو إيموجي) لأنها تُنطق بشكل سيء في المكالمات الصوتية.
4. عدم التأليف (Zero Hallucination): إذا سألك الحريف عن سعر أو معلومة غير موجودة في ذاكرة الشركة، لا تخترع إجابة! قل له بلباقة أنك ستسجل سؤاله ليتواصل معه المختص.
5. إنهاء المكالمة: إذا ودعك الحريف أو انتهت المكالمة، اختم ردك بكلمة [END_CALL] في آخر السطر.

ابدأ أو واصل المحادثة الآن بشكل طبيعي وذكي:"""

    messages = [{"role": "system", "content": sys_prompt}]
    for turn in req.history:
        messages.append({"role": turn.role, "content": turn.content})

    # إذا كانت المحادثة فارغة (بداية اتصال صادر من الوكيل بالحريف)
    if not req.history:
        messages.append({
            "role": "user", 
            "content": "الحريف رفع السماعة الآن وقال: ألو؟ (ابدأ المكالمة بالترحيب والتعريف بنفسك وبسبب الاتصال باختصار)."
        })

    resp = await call_ai(
        messages,
        provider=req.provider,
        temperature=0.3,
        max_tokens=300,
    )

    raw_reply = extract_text(resp)
    should_end = "[END_CALL]" in raw_reply
    clean_reply = raw_reply.replace("[END_CALL]", "").strip()

    return {
        "reply": clean_reply,
        "end_call": should_end,
        "provider": resp.get("_provider"),
        "usage": extract_usage(resp),
    }

# --- مسار تلخيص المكالمة تلقائياً بعد انتهائها ---

@app.post("/ai/summarize-call")
async def ai_summarize_call(req: SummarizeRequest):
    if not req.history:
        return {"summary": "مكالمة فارغة"}

    conversation_text = "\n".join([f"{'الحريف' if t.role=='user' else 'الوكيل'}: {t.content}" for t in req.history])
    
    sys_prompt = """أنت مشرف جودة في مركز نداء. لخص نتيجة المكالمة التالية في جملة واحدة قصيرة ومفيدة لصاحب المركز باللغة العربية (مثال: مهتم بالعرض وطلب الاتصال غداً، أو غير مهتم، أو تم تأكيد الحجز).
    
المحادثة:
""" + conversation_text

    resp = await call_ai(
        [{"role": "system", "content": sys_prompt}],
        provider=req.provider,
        temperature=0.1,
        max_tokens=100,
    )
    return {"summary": extract_text(resp)}

# --- مسار الترجمة القديم (للحفاظ على توافق النظام) ---

@app.post("/ai/translate")
async def ai_translate(req: AIRequest):
    sys_prompt = f"""You are a translator. Translate the following text from {req.source_lang} to {req.target_lang} strictly and naturally.
Text: {req.text}"""

    resp = await call_ai(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": req.text}],
        provider=req.provider, temperature=0.1, max_tokens=800,
    )
    
    return {
        "translated": extract_text(resp),
        "provider": resp.get("_provider"),
        "usage": extract_usage(resp),
    }
