"""Fallback brain for commands the local ML model does not recognise.

online + GROQ_API_KEY set -> Groq Chat Completions API
otherwise (or API failure)  -> local Ollama model
Standard library only (urllib) - no extra pip installs.
"""
import json
import os
import re
import socket
import time
import urllib.request
from pathlib import Path

SYSTEM = ("You are the Hyundai in-car AI voice assistant. Reply in the user's language "
          "(English or Hinglish) in 1-3 short, spoken-friendly sentences. No markdown, "
          "no lists, no emojis. Never claim you controlled a car function.")

_history = []
_net_cache = (0.0, False)


def _load_env():
    p = Path(__file__).parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


_load_env()


def is_online() -> bool:
    """Internet reachable AND API key configured (cached 10 s)."""
    global _net_cache
    if not os.getenv("GROQ_API_KEY"):
        return False
    now = time.time()
    if now - _net_cache[0] < 10:
        return _net_cache[1]
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1.5).close()
        ok = True
    except OSError:
        ok = False
    _net_cache = (now, ok)
    return ok


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _messages(text):
    return [{"role": "system", "content": SYSTEM}, *_history[-6:], {"role": "user", "content": text}]


def ask_online(text):
    data = _post("https://api.groq.com/openai/v1/chat/completions",
                 {"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "messages": _messages(text)},
                 {"Content-Type": "application/json",
                  "Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
                 float(os.getenv("GROQ_TIMEOUT", "20")))
    return data["choices"][0]["message"]["content"]


def ask_ollama(text):
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    data = _post(base + "/api/chat",
                 {"model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                  "messages": _messages(text), "stream": False},
                 {"Content-Type": "application/json"},
                 float(os.getenv("OLLAMA_TIMEOUT", "90")))
    return data["message"]["content"]


def _clean(t):
    t = re.sub(r"[*_`#>]+", "", t or "")
    return re.sub(r"\s+", " ", t).strip()


def answer(text):
    """Return (reply, source). source: 'online-api' | 'offline-ollama' | 'error: ...'."""
    attempts = []
    if is_online():
        attempts.append(("online-api", ask_online))
    attempts.append(("offline-ollama", ask_ollama))
    last = None
    for source, fn in attempts:
        try:
            reply = _clean(fn(text))
            if reply:
                _history.extend([{"role": "user", "content": text},
                                 {"role": "assistant", "content": reply}])
                del _history[:-12]
                return reply, source
        except Exception as exc:
            last = f"{source} failed: {type(exc).__name__}: {exc}"
    return ("Sorry, I could not reach any AI model right now. "
            "Please check the internet or start Ollama."), "error: " + str(last)
