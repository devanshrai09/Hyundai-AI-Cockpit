"""Prediction function + simulated cockpit + CLI demo.
Run:  python demo.py            (interactive)
      python demo.py "AC chala do"   (one-shot)     add --json for raw output
Nothing here controls real hardware: actions only update a simulated state.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

from common import (CLARIFY_T, ENTITY_FOR, LIMITS, MARGIN_T, UNKNOWN, UNKNOWN_T,
                    extract_entities, normalize)

MODEL_PATH = Path(__file__).parent / "models" / "best_model.joblib"
_model = None

# explicit negation / cancel words (checked on normalized text, so "don't" -> "don t")
NEG = re.compile(r"\b(don t|dont|do not|never mind|nevermind|never|cancel|forget it|mat|"
                 r"nahi|nahin|rehne do|rahne do|chhodo|chodo)\b")
VOLUME_INTENTS = {"INCREASE_VOLUME", "DECREASE_VOLUME"}
DIRECTION = {"up", "down", "increase", "decrease", "raise", "lower", "reduce", "louder",
             "quieter", "badha", "badhao", "badhado", "badhana", "badhaiye", "kam", "ghata",
             "ghatao", "zyada", "tez", "dheere"}


def _load():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            sys.exit("Model not found. Run: python generate_data.py && python train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(utterance: str) -> dict:
    """Classify text and apply safeguards. `safe_to_execute` is the only execution permission:
    the classifier's intent alone never authorises an action.
    confidence = calibrated max class probability (predict_proba)."""
    res = {"utterance": utterance, "intent": UNKNOWN, "entities": {}, "confidence": None,
           "requires_clarification": False, "negated": False, "safe_to_execute": False,
           "message": ""}
    norm = normalize(utterance)
    if not norm:
        res["message"] = "Empty input."
        return res
    model = _load()
    proba = model.predict_proba([utterance])[0]
    order = np.argsort(proba)[::-1]
    classes = [str(c) for c in model.classes_]
    intent, conf = classes[order[0]], float(proba[order[0]])
    margin, second = conf - float(proba[order[1]]), classes[order[1]]
    res["confidence"] = round(conf, 3)

    if intent == UNKNOWN or conf < UNKNOWN_T:
        res["message"] = "Unsupported or unclear command."
        return res
    res["intent"] = intent
    ents, err = extract_entities(intent, norm)
    res["entities"] = ents
    toks = set(norm.split())

    if intent != "STOP_SPEAKING" and NEG.search(norm):      # cancel / "don't ..." -> never execute
        res["negated"] = True
        res["message"] = "Negation or cancel detected; no action will be taken."
        return res

    q = None
    if err:
        q = f"Invalid value: {err}. Please say it again."
        res["entities"] = {}
    elif intent == "SET_TEMPERATURE" and not ents:
        q = "What temperature should I set (%d-%d C)?" % LIMITS["temperature"]
    elif intent == "SET_FAN_SPEED" and not ents:
        q = "Which fan speed (%d-%d)?" % LIMITS["fan_speed"]
    elif intent in VOLUME_INTENTS and ents and not (DIRECTION & toks):
        q = f"Do you want to increase or decrease the volume to {ents['volume_level']}?"
    elif conf < CLARIFY_T or margin < MARGIN_T:
        q = f"Did you mean {intent} or {second}?"
    if q:
        res.update(requires_clarification=True, message=q)
    else:
        res["safe_to_execute"] = True
    return res


# ---------------- simulated cockpit ----------------
STATE = {"ac": False, "temp": 24, "fan": 2, "volume": 40, "muted": False, "playing": False, "track": 1}


def act(r: dict, s: dict = STATE) -> str:
    """Simulate the action. Only runs when predict() marked it safe."""
    if r["negated"]:
        return "Okay, I won't do that."
    if r["intent"] == UNKNOWN:
        return "Sorry, I can't help with that yet."
    if not r["safe_to_execute"]:
        return r["message"] or "Could you clarify?"
    i, e = r["intent"], r["entities"]
    if i == "AC_ON":
        s["ac"] = True; return "AC turned on"
    if i == "AC_OFF":
        s["ac"] = False; return "AC turned off"
    if i == "SET_TEMPERATURE":
        s["temp"] = e["temperature"]; return f"Temperature set to {s['temp']} C"
    if i == "INCREASE_TEMPERATURE":
        s["temp"] = min(30, s["temp"] + 1); return f"Temperature increased to {s['temp']} C"
    if i == "DECREASE_TEMPERATURE":
        s["temp"] = max(16, s["temp"] - 1); return f"Temperature decreased to {s['temp']} C"
    if i == "SET_FAN_SPEED":
        s["fan"] = e["fan_speed"]; return f"Fan speed set to {s['fan']}"
    if i in VOLUME_INTENTS:
        d = 10 if i == "INCREASE_VOLUME" else -10
        s["volume"] = e.get("volume_level", max(0, min(100, s["volume"] + d)))
        s["muted"] = False
        return f"Volume {'set to' if e else 'now'} {s['volume']}"
    if i == "MUTE_AUDIO":
        s["muted"] = True; return "Audio muted"
    if i == "PLAY_MEDIA":
        s["playing"] = True; return "Playing music"
    if i == "NEXT_TRACK":
        s["track"] += 1; return f"Skipping to track {s['track']}"
    if i == "STOP_SPEAKING":
        return "Okay, I'll stay quiet."
    return "Done"


def show(text: str, as_json: bool = False):
    r = predict(text)
    if as_json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f"  intent={r['intent']}  entities={r['entities']}  confidence={r['confidence']}  "
              f"clarify={r['requires_clarification']}  negated={r['negated']}  safe={r['safe_to_execute']}")
    print(f"  Cockpit> {act(r)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.text:
        show(a.text, a.json)
    else:
        print("Cockpit NLP demo (offline). Type a command, or 'quit'.")
        while True:
            try:
                t = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if t.lower() in {"quit", "exit", "q"}:
                break
            if t:
                show(t, a.json)
