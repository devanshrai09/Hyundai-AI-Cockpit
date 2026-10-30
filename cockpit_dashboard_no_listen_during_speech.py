"""
Hyundai AI Cockpit dashboard + continuous offline voice assistant.

Run from your project root:
    python cockpit_dashboard.py --model-dir models/indic-transcribe-flex

Requirements expected in the existing project environment:
    Flask, numpy, sounddevice, scipy, pyttsx3, and the dependencies required by
    Indic-Transcribe Flex. This file imports predict() and act() from demo.py.

The TTS worker creates a fresh Windows SAPI5 engine for each response. This
matches the standalone tts_test_v2.py behavior that successfully spoke multiple
consecutive sentences on this machine.
"""

import argparse
import logging
import queue
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import pyttsx3
from scipy.io import wavfile
from flask import Flask, jsonify, render_template

from common import UNKNOWN
from demo import predict, act
from llm_fallback import answer as llm_answer, is_online


SAMPLE_RATE = 16000
RECORD_SECONDS = 7

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hyundai-voice")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
state_lock = threading.Lock()
state = {
    "status": "starting",
    "message": "Starting Hyundai AI Cockpit...",
    "transcript": "",
    "normalized": "",
    "intent": "",
    "response": "",
    "commands_processed": 0,
    "offline": True,
    "source": "local-ml",
    "logs": [],
    "avatar_action": "waiting-spin",
}
stop_event = threading.Event()
speech_queue = queue.Queue()
runtime = {
    "asr": None,
    "language": "hi",
    "seconds": RECORD_SECONDS,
    "model_dir": "models/indic-transcribe-flex",
}


def update(**kwargs):
    with state_lock:
        state.update(kwargs)


def add_log(message):
    line = time.strftime("%H:%M:%S") + " | " + str(message)
    log.info("%s", message)
    with state_lock:
        state["logs"].append(line)
        state["logs"] = state["logs"][-100:]


def speak_once(text: str) -> None:
    """Speak one utterance with a fresh SAPI5 engine, then release it."""
    text = str(text).strip()
    if not text:
        return

    engine = None
    try:
        engine = pyttsx3.init("sapi5" if sys.platform == "win32" else None)
        engine.setProperty("volume", 1.0)
        engine.setProperty("rate", 160)

        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)

        engine.say(text)
        engine.runAndWait()
        # Give Windows SAPI a moment to release playback resources.
        time.sleep(0.2)
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
            del engine


def speak(text):
    """Queue text for speech; actual playback happens in the TTS worker."""
    text = str(text).strip()
    if not text:
        return
    add_log(f"TTS queued: {text}")
    speech_queue.put(text)


def speech_worker():
    """Process speech sequentially, creating a fresh engine for every response."""
    add_log("TTS worker ready (fresh SAPI5 engine per response).")
    while True:
        text = speech_queue.get()
        try:
            if text is None:
                return
            add_log(f"TTS speaking: {text}")
            speak_once(text)
            add_log("TTS playback completed.")
        except Exception as exc:
            add_log(f"TTS ERROR: {type(exc).__name__}: {exc}")
            update(
                message=f"Speech playback failed: {exc}",
                avatar_action="failure",
            )
        finally:
            speech_queue.task_done()


def load_indic_flex(model_dir: str):
    """Load Indic-Transcribe Flex from the local model directory."""
    path = Path(model_dir).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local ASR model directory not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

    try:
        from indic_transcribe import IndicTranscribe
    except Exception as exc:
        import traceback
        add_log("IndicTranscribe import failed; printing the original traceback.")
        traceback.print_exc()
        raise RuntimeError(
            f"Could not import IndicTranscribe from {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    add_log(f"Loading Indic-Transcribe Flex from local directory: {path}")
    model = IndicTranscribe.from_pretrained(str(path))
    add_log("Indic-Transcribe Flex loaded.")
    return model


def normalize_for_classifier(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\b(?:aisi|aisee|esi|esii|a c)\b", "AC", t, flags=re.IGNORECASE)
    t = re.sub(r"\bjya?da+\b", "jyada", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\bkar(do)?\b",
        lambda m: "kar do" if m.group(0).lower() == "kardo" else m.group(0),
        t,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", t).strip()


def record_audio(seconds: int):
    update(
        status="listening",
        message=f"Listening for {seconds} seconds...",
        avatar_action="waiting-wave",
    )
    add_log(f"Recording for {seconds} seconds; speak now.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    audio = np.squeeze(audio)
    add_log(f"Recording complete: {audio.size} samples at {SAMPLE_RATE} Hz")
    return audio


def transcribe(model, audio: np.ndarray, language: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        pcm = np.clip(audio, -1.0, 1.0)
        wavfile.write(str(wav_path), SAMPLE_RATE, (pcm * 32767).astype(np.int16))
        add_log(
            f"Transcribing with Indic-Transcribe Flex "
            f"(language={language}, mode=romanized)..."
        )
        result = model(str(wav_path), lang=language, mode="romanized")
        if isinstance(result, dict):
            result = result.get("text", result.get("transcription", str(result)))
        return str(result).strip()
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            add_log(f"Warning: could not remove temporary audio file: {wav_path}")


def is_shutdown(text: str) -> bool:
    t = re.sub(r"[.!?,]+", "", text.lower()).strip()
    return t in {
        "you may switch off",
        "you may shut off",
        "you can switch off",
        "you may switch it off",
        "you may switch off now",
    }


def voice_worker():
    """Load ASR once and then continuously process command-sized recordings."""
    try:
        update(
            status="loading",
            message="Loading offline speech model...",
            avatar_action="waiting-spin",
        )
        runtime["asr"] = load_indic_flex(runtime["model_dir"])
        update(
            status="ready",
            message="Cockpit ready. Speak a command.",
            avatar_action="success",
        )
        speak(
            "Voice cockpit ready. Give your commands. "
            "Say you may switch off to end the session."
        )

        while not stop_event.is_set():
            try:
                # Do not start microphone recording until every queued spoken
                # response has finished playing. Queue.task_done() is called by
                # speech_worker only after speak_once() returns.
                speech_queue.join()
                if stop_event.is_set():
                    break

                audio = record_audio(runtime["seconds"])
                if audio.size == 0 or float(np.max(np.abs(audio))) < 0.008:
                    add_log("No usable audio detected; waiting for the next command.")
                    speak("I did not hear anything. Please try again.")
                    continue

                update(
                    status="processing",
                    message="Understanding your command...",
                    avatar_action="inspect",
                )
                raw = transcribe(runtime["asr"], audio, runtime["language"])
                add_log("ASR raw romanized transcription: " + (raw or "[no speech recognized]"))

                if not raw:
                    speak("Sorry, I could not understand that. Please try again.")
                    update(
                        status="ready",
                        message="I couldn't understand. Listening again.",
                        avatar_action="waiting-spin",
                    )
                    continue

                utterance = normalize_for_classifier(raw)
                update(transcript=raw, normalized=utterance)
                add_log("Normalized command sent to classifier: " + utterance)

                if is_shutdown(utterance):
                    add_log("Shutdown phrase recognized.")
                    speak("Goodbye. Voice cockpit switching off.")
                    update(
                        status="stopped",
                        message="Voice cockpit switched off.",
                        avatar_action="sleep",
                    )
                    stop_event.set()
                    break

                shutdown_text = re.sub(r"[.!?,]+", "", utterance.lower()).strip()
                if shutdown_text in {"quit", "exit", "stop the demo", "close cockpit"}:
                    speak("Goodbye.")
                    update(
                        status="stopped",
                        message="Voice cockpit stopped.",
                        avatar_action="sleep",
                    )
                    stop_event.set()
                    break

                add_log("Running existing intent classifier and actions...")
                prediction = predict(utterance)
                add_log("Classifier result: " + str(prediction))
                update(intent=str(prediction), avatar_action="waiting-spin")

                if prediction.get("intent") == UNKNOWN and not prediction.get("negated"):
                    # Unknown/non-command request: online API when configured and reachable,
                    # otherwise the local Ollama model. Ollama itself runs on this computer.
                    update(intent="UNKNOWN -> LLM", source="thinking", avatar_action="inspect",
                           message="Not a known cockpit command. Asking the AI assistant...")
                    online_available = is_online()
                    add_log(f"Unknown intent -> LLM fallback; online API available={online_available}")
                    response_text, source = llm_answer(utterance)
                    add_log("LLM response source: " + source)
                    failed = source.startswith("error")
                    update(source=source, offline=(source != "online-api"),
                           avatar_action="failure" if failed else "success")
                else:
                    # Known commands, clarifications, and negated requests stay on local ML.
                    update(source="local-ml", offline=True)
                    response = act(prediction)
                    response_text = str(response).strip() if response is not None else ""
                add_log("Simulated cockpit action/result: " + (response_text or "[act() returned None]"))
                with state_lock:
                    state["commands_processed"] += 1
                    state["response"] = response_text

                if response_text:
                    speak(response_text)
                else:
                    speak("The command was processed, but no response text was returned.")

                update(
                    status="ready",
                    message="Command complete. Listening for the next command.",
                    avatar_action=state["avatar_action"] if state["avatar_action"] == "failure" else "success",
                )

            except Exception as exc:
                add_log(f"Voice session error: {type(exc).__name__}: {exc}")
                update(
                    status="error",
                    message=f"Voice error: {exc}",
                    avatar_action="failure",
                )
                speak("Sorry, an error occurred. Check the dashboard logs.")
                time.sleep(1)

    except Exception as exc:
        add_log(f"Initialization failed: {type(exc).__name__}: {exc}")
        update(
            status="error",
            message=f"Initialization failed: {exc}",
            avatar_action="failure",
        )
        speak("The voice cockpit could not start. Check the dashboard logs.")
    finally:
        add_log("Voice worker ended.")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    with state_lock:
        return jsonify(dict(state))


@app.post("/api/stop")
def api_stop():
    stop_event.set()
    update(
        status="stopping",
        message="Stop requested. The current recording may finish before the loop exits.",
        avatar_action="sleep",
    )
    return jsonify({"ok": True, "message": "Stop requested"})


def main():
    parser = argparse.ArgumentParser(description="Hyundai AI Cockpit local dashboard")
    parser.add_argument("--model-dir", default=str(BASE_DIR / "models" / "indic-transcribe-flex"))
    parser.add_argument("--language", default="hi", choices=["hi", "en"])
    parser.add_argument("--seconds", type=int, default=RECORD_SECONDS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    runtime["model_dir"] = args.model_dir
    runtime["language"] = args.language
    runtime["seconds"] = max(1, args.seconds)

    threading.Thread(target=speech_worker, name="tts-worker", daemon=True).start()
    threading.Thread(target=voice_worker, name="voice-worker", daemon=True).start()

    add_log(f"Dashboard starting at http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
