#!/usr/bin/env python3
import os
import re
import sys
import json
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory, make_response
from dotenv import load_dotenv

# load .env from project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# --- third-party ---
try:
    import yt_dlp
    import requests
except Exception as e:
    print("❌ Missing packages. Install with:\n    pip install -r requirements.txt")
    print("Import error:", e)
    sys.exit(1)

# =========================
# LLM config (set here or via env vars)
# =========================
# User must set the API keys here (or export corresponding environment variables).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google").lower()
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "google": os.environ.get("GOOGLE_API_KEY", ""),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    "groq": os.environ.get("GROQ_API_KEY", ""),
}
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "models/gemini-2.5-flash").strip()

def get_api_key_for_provider(provider: str) -> Optional[str]:
    # Strip whitespace and any surrounding quotes (common .env mistake)
    key = (API_KEYS.get(provider) or "").strip()
    if key.startswith('"') and key.endswith('"'):
        key = key[1:-1]
    if key.startswith("'") and key.endswith("'"):
        key = key[1:-1]
    return key.strip()

# =========================
# Existing helper functions
# =========================

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|vi=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?.*v=([A-Za-z0-9_-]{11})",
        r"([A-Za-z0-9_-]{11})$",
    ]
    url = url.strip()
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Invalid YouTube URL — could not extract video id.")

def choose_subtitle(info: dict, pref_lang: str):
    preferred = [pref_lang] if pref_lang else []
    preferred += ["en", "hi"]

    for source in ("subtitles", "automatic_captions"):
        subs = info.get(source) or {}
        for lang in preferred:
            if lang in subs and subs[lang]:
                entry = subs[lang][0]
                return source, lang, entry.get("url"), entry.get("ext")
        if subs:
            first_lang = next(iter(subs))
            entry = subs[first_lang][0]
            return source, first_lang, entry.get("url"), entry.get("ext")
    return None

def fetch_url_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def parse_vtt(text: str):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if "-->" in line:
            lines.append({"ts": line, "text": None})
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if lines and lines[-1].get("text") is None:
            lines[-1]["text"] = line
        else:
            lines.append({"ts": None, "text": line})
    out = []
    ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")
    last_ts = None
    for item in lines:
        if item["ts"] and "-->" in item["ts"]:
            m = ts_pattern.search(item["ts"])
            if m:
                last_ts = time_str_to_seconds(m.group(1))
        if item["text"]:
            out.append((last_ts, item["text"]))
    return out

def parse_srt(text: str):
    out = []
    last_ts = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-->" in line:
            times = line.split("-->")
            if times:
                last_ts = time_str_to_seconds(times[0].replace(",", ".").strip())
            continue
        if re.fullmatch(r"\d+", line):
            continue
        out.append((last_ts, line))
    return out

def parse_ttml(text: str):
    out = []
    for m in re.finditer(r'<p[^>]*begin="([^"]+)"[^>]*>(.*?)</p>', text, flags=re.DOTALL | re.IGNORECASE):
        begin = m.group(1).strip()
        inner = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        out.append((time_str_to_seconds(begin), inner))
    if out:
        return out
    plain = re.sub(r"<[^>]+>", "", text)
    for s in re.split(r'(?<=[.!?])\s+', plain):
        s = s.strip()
        if s:
            out.append((None, s))
    return out

def parse_json3(text: str):
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r'"events"\s*:\s*(\[[\s\S]*\])', text)
        if m:
            events_text = m.group(1)
            try:
                events = json.loads(events_text)
            except Exception:
                return []
        else:
            return []
    else:
        events = obj.get("events") or obj.get("body") or []

    out = []
    for ev in events:
        start = ev.get("tStartMs") or ev.get("start") or None
        start_s = (int(start) / 1000.0) if start is not None else None
        segs = ev.get("segs") or []
        txts = []
        for seg in segs:
            if isinstance(seg, dict):
                txt = seg.get("utf8") or seg.get("utf-8") or seg.get("text")
                if txt:
                    txts.append(txt)
            elif isinstance(seg, str):
                txts.append(seg)
        full = " ".join(t.strip() for t in txts if t and t.strip())
        if full:
            out.append((start_s, full))
    return out

def time_str_to_seconds(s: str):
    s = s.strip()
    try:
        parts = s.split(":")
        parts = [p.replace(",", ".") for p in parts]
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        else:
            return float(parts[0])
    except Exception:
        try:
            ms = float(s)
            if ms > 1000:
                return ms / 1000.0
            return ms
        except Exception:
            return None

def format_ts(sec: Optional[float]):
    if sec is None:
        return ""
    sec = float(sec)
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    return f"[{hh:02d}:{mm:02d}:{ss:02d}] "

def parse_and_format(text: str, ext: Optional[str], no_timestamps: bool = False):
    ext = (ext or "").lower()
    if "json" in ext or (text and text.lstrip().startswith("{")):
        items = parse_json3(text)
    elif "vtt" in ext or (text and text.lstrip().startswith("WEBVTT")):
        items = parse_vtt(text)
    elif "srt" in ext:
        items = parse_srt(text)
    elif "ttml" in ext or (text and text.lstrip().startswith("<")):
        items = parse_ttml(text)
    else:
        items = parse_json3(text) or parse_vtt(text)

    lines = []
    for ts, t in items:
        if not t or not t.strip():
            continue
        prefix = "" if no_timestamps else format_ts(ts)
        content = re.sub(r"\s+", " ", t).strip()
        lines.append(prefix + content)
    return "\n".join(lines)

def list_available_subtitles(info):
    results = []
    for source in ("subtitles", "automatic_captions"):
        subs = info.get(source) or {}
        for lang, entries in subs.items():
            for entry in entries:
                results.append({
                    "source": source,
                    "lang": lang,
                    "ext": entry.get("ext"),
                    "url": entry.get("url"),
                })
    return results

def best_thumbnail_url(info: dict) -> Optional[str]:
    thumbs = info.get("thumbnails") or []
    if thumbs:
        try:
            best = max(thumbs, key=lambda t: (t.get("height") or 0, t.get("width") or 0))
            return best.get("url") or info.get("thumbnail")
        except Exception:
            pass
    return info.get("thumbnail")

# =========================
# Server integration (Flask)
# =========================

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fetch_transcript_for_url(url: str, pref_lang: Optional[str] = "en", no_timestamps: bool = True):
    ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    picked = choose_subtitle(info, pref_lang or "en")
    if not picked:
        raise ValueError("No subtitles or automatic captions found for this video.")

    _source, lang, sub_url, ext = picked
    if not sub_url:
        raise ValueError("Subtitle URL not available.")

    raw = fetch_url_text(sub_url)
    if not raw:
        raise RuntimeError("Failed to download subtitle content.")

    out_text = parse_and_format(raw, ext, no_timestamps=no_timestamps)
    meta = {
        "title": info.get("title"),
        "thumbnail": best_thumbnail_url(info),
        # prefer common uploader/channel fields from yt_dlp info
        "channel": info.get("uploader") or info.get("channel") or info.get("uploader_id")
    }
    return out_text.strip(), meta

def summarize_text(text: str, max_sentences: int = 8) -> str:
    # Simple frequency-based extractive summary (no external deps)
    import math
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return "\n".join(sentences)

    stop = set("""
        a an the and or if of to in on for with at by from as is are was were be been being
        this that these those it its it's their there here you your me my we our they them
        do does did doing done not no yes but so than then when where who whom which what why how
    """.split())
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    freq = {}
    for w in words:
        if w in stop or len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    if not freq:
        return "\n".join(sentences[:max_sentences])

    scores = []
    for i, s in enumerate(sentences):
        tokens = re.findall(r"[A-Za-z0-9']+", s.lower())
        score = sum(freq.get(t, 0) for t in tokens) / math.log(len(tokens) + 2)
        scores.append((score, i, s))

    top = sorted(scores, reverse=True)[:max_sentences]
    top_sorted_by_position = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    return "\n".join(top_sorted_by_position)

# -------- Static files --------

@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(BASE_DIR, "style.css")

@app.route("/script.js")
def serve_js():
    return send_from_directory(BASE_DIR, "script.js")

@app.route("/favicon.ico")
def serve_favicon():
    return ("", 204)

# -------- API endpoints --------

@app.post("/get_transcript")
def api_get_transcript():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()
    timestamps = bool(data.get("timestamps", True))
    if not url:
        return jsonify(error="Please provide a YouTube URL."), 400
    try:
        _ = extract_video_id(url)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    try:
        transcript, meta = fetch_transcript_for_url(
            url, pref_lang=lang or "en", no_timestamps=(not timestamps)
        )
        if not transcript:
            return jsonify(error="No transcript found for this video."), 404
        return jsonify(
            transcript=transcript,
            lang=lang,
            title=meta.get("title"),
            thumbnail=meta.get("thumbnail"),
            channel=meta.get("channel"),
        )
    except yt_dlp.utils.DownloadError:
        return jsonify(error="Failed to fetch video info."), 502
    except ValueError as e:
        return jsonify(error=str(e)), 404
    except Exception:
        return jsonify(error="Failed to fetch transcript."), 500

@app.post("/download_transcript")
def api_download_transcript():
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify(error="Transcript is required."), 400

    resp = make_response(transcript)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Content-Disposition"] = "attachment; filename=transcript.txt"
    return resp

@app.post("/summarize_llm")
def api_summarize_llm():
    """
    LLM summarization — server uses configured provider and API key (set in Python / env)
    Client should send only: { "transcript": "...", "max_sentences": 8 }
    """
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    max_sentences = int(data.get("max_sentences") or 8)

    if not transcript:
        return jsonify(error="Transcript is required."), 400

    provider = LLM_PROVIDER
    api_key = get_api_key_for_provider(provider)
    # Debug/log: avoid printing full key - only length
    print(f"LLM provider: {provider}, api_key_loaded: {bool(api_key)}, key_len: {len(api_key) if api_key else 0}")

    if not provider or not api_key:
        return jsonify(error="Server LLM provider or API key not configured."), 500

    # Trim transcript to avoid token limits
    trimmed = transcript[:15000]

    prompt = f"Summarize the following YouTube transcript in at most {max_sentences} concise sentences. Avoid redundancy and keep factual tone.\n\nTranscript:\n{trimmed}"

    try:
        if provider == "google":
            summary = summarize_with_google(prompt, api_key, GOOGLE_MODEL)
        elif provider == "openai":
            summary = summarize_with_openai(prompt, api_key)
        elif provider == "anthropic":
            summary = summarize_with_anthropic(prompt, api_key)
        elif provider == "groq":
            summary = summarize_with_groq(prompt, api_key)
        else:
            return jsonify(error="Unsupported LLM provider configured on server."), 500

        return jsonify(summary=summary)

    except Exception as e:
        # Surface the real error back to client and log it
        print(f"LLM Error: {e}")
        return jsonify(error=f"Failed LLM summarization: {e}"), 500

def summarize_with_google(prompt: str, api_key: str, model_name: str) -> str:
    import requests, time

    base = model_name.strip()
    if not base.startswith("models/"):
        base = "models/" + base

    # Candidate model names (all appear in your available list)
    candidates = [
        base,
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.0-flash",
    ]

    # Prefer GA v1 endpoint, then v1beta fallback.
    api_versions = ["v1", "v1beta"]

    tried = []
    last_error = None

    def call(model_full: str, version: str):
        url = f"https://generativelanguage.googleapis.com/{version}/{model_full}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 600,
            }
        }
        r = requests.post(f"{url}?key={api_key}", json=payload, timeout=45)
        if r.status_code == 404:
            raise FileNotFoundError(f"404 {version}/{model_full}")
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {version}/{model_full}: {r.text[:180]}")
        data = r.json()
        texts = []
        for cand in data.get("candidates", []):
            content = cand.get("content") or {}
            for part in content.get("parts", []):
                t = part.get("text")
                if t:
                    texts.append(t.strip())
        out = "\n".join(texts).strip()
        if not out:
            raise ValueError(f"Empty completion {version}/{model_full}")
        return out

    for m in candidates:
        if m in tried:
            continue
        for ver in api_versions:
            ident = f"{ver}/{m}"
            tried.append(ident)
            try:
                return call(m, ver)
            except Exception as e:
                last_error = str(e)
                continue

    # List available models to help debugging
    avail = "unknown"
    try:
        lr = requests.get(f"https://generativelanguage.googleapis.com/v1/models?key={api_key}", timeout=20)
        if lr.status_code < 400:
            names = [mdl.get("name") for mdl in (lr.json().get("models") or []) if mdl.get("name")]
            avail = ", ".join(names) if names else "none"
    except Exception:
        pass

    raise Exception(f"Gemini summarization failed. Tried: {tried}. Last error: {last_error}. Available (v1) models: {avail}")

def summarize_with_openai(prompt: str, api_key: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        raise Exception(f"OpenAI API error: {r.status_code}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def summarize_with_anthropic(prompt: str, api_key: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        raise Exception(f"Anthropic API error: {r.status_code}")

    data = r.json()
    return data["content"][0]["text"].strip()

def summarize_with_groq(prompt: str, api_key: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code >= 400:
        raise Exception(f"Groq API error: {r.status_code}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
