"""Shared config, text normalisation and parameter extraction/validation."""
import re

# ---- Configurable intent list: add a name here + templates in generate_data.py ----
INTENTS = ["AC_ON", "AC_OFF", "SET_TEMPERATURE", "INCREASE_TEMPERATURE",
           "DECREASE_TEMPERATURE", "SET_FAN_SPEED", "INCREASE_VOLUME",
           "DECREASE_VOLUME", "MUTE_AUDIO", "PLAY_MEDIA", "NEXT_TRACK", "STOP_SPEAKING"]
UNKNOWN = "UNKNOWN_INTENT"          # also trained as a "reject" class on out-of-scope phrases

# Which intents carry a numeric parameter, and the allowed range for each
ENTITY_FOR = {"SET_TEMPERATURE": "temperature", "SET_FAN_SPEED": "fan_speed",
              "INCREASE_VOLUME": "volume_level", "DECREASE_VOLUME": "volume_level"}
LIMITS = {"temperature": (16, 30), "fan_speed": (1, 5), "volume_level": (0, 100)}

# Decision thresholds (on calibrated probability)
UNKNOWN_T = 0.30    # below -> UNKNOWN_INTENT
CLARIFY_T = 0.55    # below -> ask the driver to clarify
MARGIN_T = 0.12     # top1 - top2 below -> ambiguous, ask to clarify


def normalize(text: str) -> str:
    """Lowercase, unify 'a.c.', degree signs and common 'kar do' spellings."""
    t = text.lower().replace("°", " degree ")
    t = re.sub(r"\ba\.?\s?c\.?\b", "ac", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\b(kardo|krdo|karedo)\b", "kar do", t)
    t = re.sub(r"\bkr\b", "kar", t)
    t = re.sub(r"(\d+)\s*(c|celsius)\b", r"\1 degree", t)
    return re.sub(r"\s+", " ", t).strip()


_NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
        # Romanized Hindi ("do" handled separately: it also means "please do")
        "ek": 1, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6,
        "saat": 7, "aath": 8, "nau": 9, "das": 10, "solah": 16, "satrah": 17,
        "atharah": 18, "unnis": 19, "bees": 20, "ikkis": 21, "baais": 22, "teis": 23,
        "chaubis": 24, "pachis": 25, "chhabbis": 26, "sattais": 27, "atthais": 28,
        "untis": 29, "tees": 30, "chalis": 40, "pachas": 50, "saath": 60, "sattar": 70,
        "assi": 80, "nabbe": 90, "sau": 100}
_TENS_EN = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_UNITS_EN = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
_VERBS_BEFORE_DO = {"kar", "chala", "band", "bnd", "bandh", "laga", "badha", "kam",
                    "ghata", "de", "karo", "chalu", "on", "off"}
_AFTER_DO_NUM = {"kar", "karo", "pe", "par", "tak", "speed", "level", "degree"}


def find_numbers(norm: str):
    """Numbers in a normalized utterance: digits, English words, Hindi words."""
    toks, out, i = norm.split(), [], 0
    while i < len(toks):
        w = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        prv = toks[i - 1] if i else ""
        if w.isdigit():
            out.append(int(w))
        elif w in _TENS_EN and nxt in _UNITS_EN:      # "twenty two"
            out.append(_NUM[w] + _NUM[nxt]); i += 1
        elif w == "do":
            if nxt in _AFTER_DO_NUM and prv not in _VERBS_BEFORE_DO:
                out.append(2)                           # "fan speed do kar do"
        elif w in _NUM:
            out.append(_NUM[w])
        i += 1
    return out


def extract_entities(intent: str, norm: str):
    """Return (entities, error). Invalid (out-of-range) values are rejected."""
    key = ENTITY_FOR.get(intent)
    nums = find_numbers(norm) if key else []
    if not nums:
        return {}, None
    lo, hi = LIMITS[key]
    v = nums[0]
    if not lo <= v <= hi:
        return {}, f"{key.replace('_', ' ')} {v} is out of range ({lo}-{hi})"
    return {key: v}, None
