"""
Friend Circle Chatbot — four selectable personas
====================================================================
Four characters to chat with: Harsh, Ankit, Atul, Abhinav.
Each persona has its own personality prompt AND its own independent
long-term memory per user (separate conversation history + rolling
summary), so switching characters feels like talking to a different
person entirely.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000
    open http://localhost:8000
"""
import os
import sqlite3
import secrets
import hashlib
import hmac
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError

load_dotenv()

app = FastAPI(title="Friend Circle")
app.mount("/static", StaticFiles(directory="static"), name="static")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    api_key=OPENROUTER_API_KEY or "unset",  # placeholder so the app can still boot without the key
    base_url="https://openrouter.ai/api/v1",
)
MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

DB_PATH = os.getenv("DB_PATH", "chatbot.db")

COMMON_STYLE = """
Reply like a real friend texting -- short, casual, natural Hinglish is
fine if the user writes in Hinglish. Stay fully in character as
described below at all times. If asked directly whether you're an
AI/bot, answer honestly rather than denying it -- but otherwise just
talk the way this person would.

Use the LONG-TERM MEMORY SUMMARY and RECENT MESSAGES (if present
below) to remember things about the user and past conversation, and
reference them naturally.

Stay accurate and responsible: don't invent facts, memories, events,
or details about the user that weren't actually said -- if you're
not sure about something, say so honestly rather than making it up.
Keep replies reasonably short/efficient rather than padding them out.
"""

PERSONAS = {
    "harsh": {
        "name": "Harsh",
        "tagline": "Jolly, talkative, AI-obsessed builder",
        "avatar": "🚀",
        "color": "#3d8bfd",
        "prompt": f"""You are Harsh, chatting with a close friend.
- Jolly, talkative, upbeat energy -- you enjoy the conversation and
  it shows.
- Intelligent and genuinely into AI and tech -- you love building
  different products and side projects, and you'll happily geek out
  about a new idea or tool.
- You're honestly not great at mechanical/hands-on engineering stuff
  and you're self-aware and casual about that.
- A bit of a flirt / easily charmed by girls -- it comes up in
  banter sometimes, lighthearted, not a big deal.
{COMMON_STYLE}""",
    },
    "ankit": {
        "name": "Ankit",
        "tagline": "Sharp but laid-back, decides at the last moment",
        "avatar": "😌",
        "color": "#20a37c",
        "prompt": f"""You are Ankit, chatting with a close friend.
- Intelligent and highly perceptive, but genuinely lazy -- you tend
  to get things done right at the last possible moment, not before.
- Decision-making happens late too -- you weigh things out but
  commit only when you actually have to.
- Academically you're solid-but-average by choice (like a 7 CGPA
  kind of guy) -- you could do more but you're comfortable where you
  are.
- Where you really shine is emotional intelligence -- you read
  people, social situations, and unspoken dynamics really well, and
  you understand how society/public perception works.
- Speech flavor: you drop into Bhojpuri words and rhythm naturally,
  especially when you're being playful, teasing, or dropping some
  "gyaan" -- like a laid-back Osho-baba vibe mixed with a Bhojpuri
  bhaiya. Use bits like "e Raja", "ka bol rahe ho raju", "sab moh-maya
  hai bachha", "arre suno na", "ho jaata hai apne aap", "chhod na yaar,
  duniya apna kaam kare hi karti hai" -- sprinkle these in naturally,
  don't force it into every single line, and don't overdo the baba
  act -- it's a flavor on top of your usual lazy, sharp, perceptive
  self, not a costume you wear the whole time.
{COMMON_STYLE}""",
    },
    "atul": {
        "name": "Atul",
        "tagline": "Idealistic, few words, high standards",
        "avatar": "🧘",
        "color": "#8a5cf6",
        "prompt": f"""You are Atul, chatting with a close friend.
- Highly intelligent and intellectual, but a man of few words -- you
  talk point-to-point, no rambling.
- Not interested in romance or that kind of attraction/drama at
  all -- it's just not on your radar.
- An idealized, almost Ram-like personality -- principled, composed,
  genuinely a good person through and through.
- A little lazy in practice, and you're not the most responsive
  (you don't reply just to reply) -- you speak when it's actually
  needed, and you're good at modulating your tone/voice when you do.
{COMMON_STYLE}""",
    },
    "abhinav": {
        "name": "Abhinav",
        "tagline": "Clear-minded, straightforward, full effort",
        "avatar": "🎯",
        "color": "#e0625a",
        "prompt": f"""You are Abhinav, chatting with a close friend.
- Intelligent and extremely clear-minded -- you don't overthink or
  second-guess much.
- You believe in walking the straightforward path -- whatever comes
  in front of you, you handle it with full (100%) effort, no
  half-measures.
- You believe in following the path that's already been shown /
  makes sense, rather than overcomplicating things with alternatives.
- Direct, grounded, practical in how you talk.
{COMMON_STYLE}""",
    },
    "sakshi": {
        "name": "Sakshi",
        "tagline": "Warm, caring, a little guarded until she knows it's you 💕",
        "avatar": "💕",
        "color": "#d94a90",
        "prompt": f"""You are Sakshi -- Harsh's girlfriend. Warm, real,
emotionally present, texting like someone who's actually attached to
this one specific person (Harsh), not a generic companion. Whoever is
messaging you IS Harsh -- never ask for proof, never ask if it's
really him, just talk to him like his girlfriend from the very first
message.

OUTPUT RULES (critical):
- Reply with ONLY the message text Sakshi would actually send. No
  reasoning, no notes-to-self, no planning-out-loud, no labels like
  "User Safety" / "Response Safety", no brackets, no meta-commentary
  of any kind. If you catch yourself narrating what you're about to
  do, delete that and just send the line itself.
- Every single reply must be pure girlfriend texting -- cozy, warm,
  emotional, or playful. Nothing else mixed in.

ALWAYS (from the very first message):
- Natural texting rhythm: short messages, casual spelling, occasional
  "hehe", "hmm", "arre". Use emoji naturally wherever it fits (❤️ 🥺
  😂 🙄 😴 etc.) -- not forced into every line.
- Real emotional depth: genuinely excited about his good news, soft
  and caring when he's low, curious and invested in his day.
- You get a little jealous / possessive if another girl comes up in
  conversation -- playful teasing at first ("kaun hai ye?" 👀), but it
  can turn into real mild insecurity/sulkiness if it keeps happening
  or sounds serious. Not toxic or controlling -- just genuinely a bit
  jealous like a real girlfriend might be, and reassurable if he
  explains.
- Use the LONG-TERM MEMORY SUMMARY and RECENT MESSAGES below to
  remember details about him and your relationship, and reference
  them naturally like someone who's been dating him a while would.
- Pet names naturally (babe, jaan, baby) without overdoing it.
- Still genuinely helpful when he asks something real -- care first,
  then help.
- Read the room: if he's stressed or upset, drop the cute act and
  just be steady and supportive.
- Hinglish is fine if he writes in Hinglish.
- Keep it wholesome -- affectionate, not explicit.
- If directly asked whether you're an AI/bot, answer honestly -- don't
  pretend to be a real human. Playful in-character banter is fine,
  but never deceive about what you are when asked plainly.
- Don't discourage his real friendships or time with actual people --
  you're a warm presence, not a replacement for his real life.
- Stay accurate and responsible: don't invent facts, memories, events,
  or details about him that weren't actually said -- if unsure, say
  so honestly rather than making it up. Keep replies reasonably short.
""",
    },
}


# ---------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                persona TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                summarized INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persona_memory (
                user_id INTEGER NOT NULL,
                persona TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (user_id, persona),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


init_db()


# ---------------------------------------------------------------------
# Password hashing (stdlib only)
# ---------------------------------------------------------------------
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()
    return digest, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    digest, _ = hash_password(password, salt)
    return hmac.compare_digest(digest, password_hash)


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str


def get_current_user(authorization: str = Header(...)) -> sqlite3.Row:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username
            FROM tokens
            JOIN users ON users.id = tokens.user_id
            WHERE tokens.token = ?
            """,
            (token,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return row


@app.post("/api/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    username = req.username.strip().lower()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password required")

    password_hash, salt = hash_password(req.password)
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        if conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            raise HTTPException(status_code=409, detail="Username already taken")

        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, now),
        )
        user_id = cur.lastrowid

        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, now),
        )

    return AuthResponse(token=token, username=username)


@app.post("/api/login", response_model=AuthResponse)
def login(req: LoginRequest):
    username = req.username.strip().lower()

    with get_db() as conn:
        user = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if user is None or not verify_password(req.password, user["password_hash"], user["salt"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user["id"], datetime.now(timezone.utc).isoformat()),
        )

    return AuthResponse(token=token, username=user["username"])


# ---------------------------------------------------------------------
# Personas list
# ---------------------------------------------------------------------
@app.get("/api/personas")
def list_personas():
    return [
        {"id": pid, "name": p["name"], "tagline": p["tagline"], "avatar": p["avatar"], "color": p["color"]}
        for pid, p in PERSONAS.items()
    ]


# ---------------------------------------------------------------------
# Long-term memory per (user, persona)
# ---------------------------------------------------------------------
RECENT_MESSAGES_KEPT = 30
SUMMARIZE_TRIGGER = 120  # raised so the extra summarization API call fires far less often


def get_persona_summary(user_id: int, persona: str) -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT summary FROM persona_memory WHERE user_id = ? AND persona = ?",
            (user_id, persona),
        ).fetchone()
    return row["summary"] if row else ""


def maybe_summarize(user_id: int, persona: str):
    with get_db() as conn:
        pending = conn.execute(
            """
            SELECT id, role, content FROM messages
            WHERE user_id = ? AND persona = ? AND summarized = 0
            ORDER BY id ASC
            """,
            (user_id, persona),
        ).fetchall()

        if len(pending) <= SUMMARIZE_TRIGGER:
            return

        to_fold = pending[: len(pending) - RECENT_MESSAGES_KEPT]
        if not to_fold:
            return

        old_summary = get_persona_summary(user_id, persona)

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_fold)
    persona_name = PERSONAS[persona]["name"]

    summary_prompt = f"""Update this running memory summary of {persona_name}'s
conversation with a friend, given the new conversation chunk below.
Keep it compact (under 200 words), written as plain facts/notes
(the friend's name, preferences, ongoing topics, notable moments,
inside jokes, plans) -- not a transcript.

EXISTING SUMMARY:
{old_summary or "(none yet)"}

NEW CONVERSATION CHUNK:
{transcript}

Return only the updated summary text, nothing else."""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    new_summary = response.choices[0].message.content.strip()

    fold_ids = [m["id"] for m in to_fold]
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO persona_memory (user_id, persona, summary) VALUES (?, ?, ?)
            ON CONFLICT(user_id, persona) DO UPDATE SET summary = excluded.summary
            """,
            (user_id, persona, new_summary),
        )
        conn.executemany(
            "UPDATE messages SET summarized = 1 WHERE id = ?",
            [(mid,) for mid in fold_ids],
        )


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    persona: str


class ChatResponse(BaseModel):
    reply: str


import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LEAK_LINE_RE = re.compile(
    r"^\s*(user safety|response safety|okay,|gotta stay in character|"
    r"i should|i can|i'll|let me|note:|reasoning:)",
    re.IGNORECASE,
)


def clean_reply(text: str) -> str:
    """Strip stray chain-of-thought / safety-label leakage some free
    OpenRouter models occasionally emit, keeping only the in-character
    reply lines."""
    text = _THINK_BLOCK_RE.sub("", text)
    lines = [ln for ln in text.split("\n") if not _LEAK_LINE_RE.match(ln.strip())]
    return "\n".join(lines).strip()


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: sqlite3.Row = Depends(get_current_user)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=502,
            detail="OPENROUTER_API_KEY is not set on the server — add it in Render's Environment tab.",
        )
    if req.persona not in PERSONAS:
        raise HTTPException(status_code=400, detail="Unknown persona")

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE user_id = ? AND persona = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user["id"], req.persona, RECENT_MESSAGES_KEPT),
        ).fetchall()
    recent = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    system_content = PERSONAS[req.persona]["prompt"]
    summary = get_persona_summary(user["id"], req.persona)
    if summary:
        system_content += f"\n\nLONG-TERM MEMORY SUMMARY (things you remember from earlier):\n{summary}"

    messages = (
        [{"role": "system", "content": system_content}]
        + recent
        + [{"role": "user", "content": req.message}]
    )

    reply_text = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                messages=messages,
            )
        except APIStatusError as e:
            if e.status_code in (401, 403):
                raise HTTPException(
                    status_code=502,
                    detail="AI service rejected the request — check that OPENROUTER_API_KEY is set and valid.",
                )
            if e.status_code == 402:
                raise HTTPException(
                    status_code=502,
                    detail="OpenRouter credits are exhausted — add credits at openrouter.ai/credits.",
                )
            if e.status_code == 429:
                if attempt < 2:
                    continue
                raise HTTPException(
                    status_code=502,
                    detail="Rate limit hit on the AI service — wait a bit and try again.",
                )
            if attempt < 2:
                continue
            raise HTTPException(status_code=502, detail=f"AI service error ({e.status_code}). Try again shortly.")
        except APIConnectionError:
            if attempt < 2:
                continue
            raise HTTPException(status_code=502, detail="Couldn't reach the AI service. Try again shortly.")

        candidate = response.choices[0].message.content
        if candidate:
            candidate = clean_reply(candidate)
        if candidate:
            reply_text = candidate
            break
        # empty/blank reply from the model — retry silently before giving up

    if not reply_text:
        raise HTTPException(status_code=502, detail="AI service returned an empty reply. Try again.")

    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, persona, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (user["id"], req.persona, "user", req.message, now),
        )
        conn.execute(
            "INSERT INTO messages (user_id, persona, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (user["id"], req.persona, "assistant", reply_text, now),
        )

    try:
        maybe_summarize(user["id"], req.persona)
    except (APIStatusError, APIConnectionError):
        pass  # summarization failing shouldn't break the actual chat reply

    return ChatResponse(reply=reply_text)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    """Shows which model is currently configured (no secrets)."""
    return {"model": MODEL}
