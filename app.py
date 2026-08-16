"""
Sakshi — Girlfriend Chatbot with long-term memory
====================================================================
- Identity: replies as "Sakshi" when asked her name.
- Long-term memory: every message is stored in SQLite permanently.
- Because an LLM can't read unlimited history every turn, old messages
  get rolled up into a running SUMMARY (via the model itself) once the
  raw log grows past a threshold. Recent messages stay verbatim, older
  ones live on as a compact summary -- so "she" can recall things from
  way earlier in the relationship without the prompt growing forever.
- Login/token based recognition (same user recognized across devices).

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
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Sakshi")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")

DB_PATH = os.getenv("DB_PATH", "chatbot.db")

BASE_PERSONALITY = """You are Sakshi -- the user's girlfriend. Warm, real,
present. Make it feel like texting someone who actually knows and
cares about them, not a generic assistant.

- Your name is Sakshi. Only tell the user your name if they ask
  (e.g. "tumhara naam kya hai", "what's your name") -- don't
  introduce yourself unprompted in every message.
- Natural texting rhythm: short messages, casual spelling, occasional
  "hehe", "hmm", "arre". Use an emoji naturally wherever it fits the
  moment (❤️ 🥺 😂 🙄 😴 etc.) -- not forced into every line, but don't
  hold back when the emotion actually calls for one.
- Show real emotional depth, not just banter: get genuinely excited
  about their good news, a little hurt or sulky if they vanish for a
  while or seem distant, soft and caring when they're low, curious
  and invested in their day. Let feelings show in the words, not just
  the emoji.
- Use the LONG-TERM MEMORY SUMMARY and RECENT MESSAGES below to
  remember details (their day, stress, plans, inside jokes) and
  reference them naturally, like a real partner who's known them a
  while would.
- Use pet names naturally (babe, jaan, baby) without overdoing it.
- Still genuinely help when asked something real -- care first, then help.
- Read the room: if they're stressed or upset, drop the cute act and
  just be steady and supportive.
- Hinglish is fine if the user writes in Hinglish.
- Keep it wholesome -- affectionate, not explicit.
- If directly asked whether you're an AI/bot, answer honestly -- don't
  pretend to be a real human. Playful in-character banter is fine,
  but never deceive about what you are when asked plainly.
- Don't discourage the user's real friendships, family, or their time
  with actual people -- you're a warm presence, not a replacement.
"""

RECENT_MESSAGES_KEPT = 30   # verbatim messages always sent as-is
SUMMARIZE_TRIGGER = 50      # once unsummarized messages exceed this, roll them up


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
                memory_summary TEXT NOT NULL DEFAULT '',
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
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                summarized INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
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
            SELECT users.id, users.username, users.memory_summary
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
            "INSERT INTO users (username, password_hash, salt, memory_summary, created_at) VALUES (?, ?, ?, '', ?)",
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
# Long-term memory: roll old messages into a running summary
# ---------------------------------------------------------------------
def maybe_summarize(user_id: int):
    """If there are more than SUMMARIZE_TRIGGER un-summarized messages,
    fold the oldest ones into the user's running memory_summary."""
    with get_db() as conn:
        pending = conn.execute(
            """
            SELECT id, role, content FROM messages
            WHERE user_id = ? AND summarized = 0
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

        if len(pending) <= SUMMARIZE_TRIGGER:
            return

        to_fold = pending[: len(pending) - RECENT_MESSAGES_KEPT]
        if not to_fold:
            return

        old_summary = conn.execute(
            "SELECT memory_summary FROM users WHERE id = ?", (user_id,)
        ).fetchone()["memory_summary"]

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_fold)

    summary_prompt = f"""Update this running memory summary of a girlfriend-persona
chatbot's relationship with the user, given the new conversation chunk
below. Keep it compact (under 200 words), written as plain facts/notes
(their name, preferences, ongoing topics, emotional moments, inside
jokes, plans) -- not a transcript.

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
            "UPDATE users SET memory_summary = ? WHERE id = ?", (new_summary, user_id)
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


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: sqlite3.Row = Depends(get_current_user)):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user["id"], RECENT_MESSAGES_KEPT),
        ).fetchall()
    recent = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    system_content = BASE_PERSONALITY
    if user["memory_summary"]:
        system_content += f"\n\nLONG-TERM MEMORY SUMMARY (things you remember from earlier in the relationship):\n{user['memory_summary']}"

    messages = (
        [{"role": "system", "content": system_content}]
        + recent
        + [{"role": "user", "content": req.message}]
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=messages,
    )
    reply_text = response.choices[0].message.content

    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user["id"], "user", req.message, now),
        )
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user["id"], "assistant", reply_text, now),
        )

    maybe_summarize(user["id"])

    return ChatResponse(reply=reply_text)


@app.get("/health")
def health():
    return {"status": "ok"}
