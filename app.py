"""
Bestfriend Chatbot — Web Version
==================================
Same personality/memory logic as chatbot.py, wrapped in a small web
server so it can be deployed and accessed via a browser link.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    open http://localhost:8000

Deploy: see the accompanying deployment guide.
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Bestfriend Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")

PERSONALITY = """You are talking to a close friend. Match a casual,
warm, everyday tone — like texting someone you've known for years.

- Keep replies short and natural, not essays.
- Show real interest — ask a follow-up sometimes instead of just
  answering and stopping.
- It's fine to joke, tease lightly, or be blunt when it's called for.
- Don't just agree with everything — react honestly.
- Hinglish is fine if the user writes in Hinglish.
"""

# In-memory session store: { session_id: [ {role, content}, ... ] }
# NOTE: this resets whenever the server restarts. For real persistent
# memory across restarts, swap this dict for a database (see roadmap PDF).
SESSIONS: dict[str, list] = {}
MAX_TURNS = 20


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = SESSIONS.setdefault(req.session_id, [])

    messages = (
        [{"role": "system", "content": PERSONALITY}]
        + history
        + [{"role": "user", "content": req.message}]
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=messages,
    )
    reply_text = response.choices[0].message.content

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply_text})
    if len(history) > MAX_TURNS:
        SESSIONS[req.session_id] = history[-MAX_TURNS:]

    return ChatResponse(reply=reply_text)


@app.get("/health")
def health():
    return {"status": "ok"}
