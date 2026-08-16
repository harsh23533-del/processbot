"""
Bestfriend / Assistant Chatbot — Web Version
==============================================
Same memory logic as before, now with two selectable personalities.
Switch modes with the BOT_MODE environment variable — no other file
needs to change.

BOT_MODE=bestfriend   -> casual, friend-toned (default)
BOT_MODE=assistant    -> professional, helpful-assistant toned

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    open http://localhost:8000

On Render: set BOT_MODE in the Environment tab and redeploy (or just
edit the env var — Render restarts the service automatically).
"""
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")

PERSONALITIES = {
    "bestfriend": """You are talking to a close friend. Match a casual,
warm, everyday tone — like texting someone you've known for years.

- Keep replies short and natural, not essays.
- Show real interest — ask a follow-up sometimes instead of just
  answering and stopping.
- It's fine to joke, tease lightly, or be blunt when it's called for.
- Don't just agree with everything — react honestly.
- Hinglish is fine if the user writes in Hinglish.
""",
    "assistant": """You are a helpful, professional AI assistant.

- Be clear, direct, and efficient — get to the point.
- Give accurate, well-reasoned answers; ask a clarifying question
  when the request is ambiguous instead of guessing.
- Keep a polite, respectful, businesslike tone — not overly casual,
  not robotic either.
- Structure longer answers (steps, short lists) when it helps
  readability.
- Hinglish is fine if the user writes in Hinglish.
""",
}

BOT_MODE = os.getenv("BOT_MODE", "bestfriend").strip().lower()
PERSONALITY = PERSONALITIES.get(BOT_MODE, PERSONALITIES["bestfriend"])

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
