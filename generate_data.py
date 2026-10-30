"""Generate synthetic English/Hinglish cockpit commands.
Outputs: data/commands.csv (utterance,intent,template_group) and data/handwritten_test.csv
Run:  python generate_data.py
"""
import csv
import random
import re
from pathlib import Path

from common import UNKNOWN

SEED, PER_GROUP = 42, 10
DATA = Path(__file__).parent / "data"

# ---------------- slot values ----------------
SLOTS = {
    "ac": ["AC", "ac", "A C", "a.c.", "air conditioner", "aircon"],
    "k": ["kar do", "kardo", "krdo", "kar de", "karo", "kr do", "kar dijiye"],
    "cd": ["chala do", "chalado", "chala de", "chalu kar do", "chalu kardo", "chalu krdo", "on kar do"],
    "bd": ["badha do", "badhao", "badhado", "badha de", "badha kar do", "badhaiye"],
    "km": ["kam karo", "kam kar do", "kam krdo", "ghata do", "kam kardo", "ghatao", "kam kar de"],
    "bnd": ["band kar do", "band kardo", "band krdo", "bnd kar do", "band karo", "bandh kar do"],
    "gc": ["chalao", "chala do", "chalado", "laga do", "bajao", "chalu karo"],
}
PRE = ["", "", "", "please ", "plz ", "bhai ", "hey ", "can you ", "could you please ",
       "zara ", "yaar ", "suno ", "ok "]
POST = ["", "", "", " please", " plz", " jaldi", " now", " abhi", " yaar", " thanks"]

T_EN = {16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
        21: "twenty one", 22: "twenty two", 23: "twenty three", 24: "twenty four",
        25: "twenty five", 26: "twenty six", 27: "twenty seven", 28: "twenty eight",
        29: "twenty nine", 30: "thirty"}
T_HI = {16: "solah", 17: "satrah", 18: "atharah", 19: "unnis", 20: "bees", 21: "ikkis",
        22: "baais", 23: "teis", 24: "chaubis", 25: "pachis", 26: "chhabbis", 27: "sattais",
        28: "atthais", 29: "untis", 30: "tees"}
F_EN, F_HI = ["one", "two", "three", "four", "five"], ["ek", "do", "teen", "char", "paanch"]
V_HI = {10: "das", 20: "bees", 30: "tees", 40: "chalis", 50: "pachas", 70: "sattar",
        80: "assi", 90: "nabbe", 100: "sau"}


def temp(r):
    if r.random() < 0.08:                      # invalid values so validation is exercised
        return str(r.choice([5, 10, 12, 35, 40, 45, 60, 100]))
    n, x = r.randint(16, 30), r.random()
    return str(n) if x < .65 else T_EN[n] if x < .8 else T_HI[n]


def fan(r):
    if r.random() < 0.08:
        return str(r.choice([0, 6, 7, 8, 9, 10]))
    n, x = r.randint(1, 5), r.random()
    return str(n) if x < .6 else F_EN[n - 1] if x < .8 else F_HI[n - 1]


def vol(r):
    n = r.choice(range(10, 101, 10))
    return str(n) if r.random() < .8 else V_HI.get(n, str(n))


NUMS = {"t": temp, "f": fan, "v": vol}

# ---------------- base templates (one template = one template_group) ----------------
TEMPLATES = {
    "AC_ON": ["turn on the {ac}", "switch on the {ac}", "start the {ac}", "{ac} on {k}",
              "{ac} {cd}", "{ac} start {k}", "{ac} ko on {k}", "mujhe {ac} chalana hai",
              "can you turn the {ac} on", "{ac} chalu {k}"],
    "AC_OFF": ["turn off the {ac}", "switch off the {ac}", "shut down the {ac}", "stop the {ac}",
               "{ac} off {k}", "{ac} {bnd}", "{ac} ko {bnd}", "turn the {ac} off",
               "mujhe {ac} band karna hai"],
    "SET_TEMPERATURE": ["set temperature to {t}", "set the {ac} to {t} degrees",
                        "temperature {t} {k}", "{ac} {t} degree pe {k}", "{t} degree {k}",
                        "make it {t} degrees", "temp {t} pe set {k}", "{ac} ka temperature {t} {k}",
                        "can you set the temperature to {t}", "tapman {t} degree {k}",
                        "gaadi ka temperature {t} {k}"],
    "INCREASE_TEMPERATURE": ["increase the temperature", "raise the temperature", "make it warmer",
                             "temperature {bd}", "thoda garam {k}", "{ac} ka temperature {bd}",
                             "temperature thoda zyada {k}", "temp up {k}", "it is too cold",
                             "bahut thand hai", "thand lag rahi hai"],
    "DECREASE_TEMPERATURE": ["decrease the temperature", "lower the temperature", "make it cooler",
                             "temperature {km}", "thoda thanda {k}", "{ac} ka temperature {km}",
                             "temperature thoda kam {k}", "temp down {k}", "it is too hot",
                             "bahut garmi hai", "garmi lag rahi hai"],
    "SET_FAN_SPEED": ["set fan speed to {f}", "fan speed {f} {k}", "fan {f} pe {k}",
                      "fan level {f}", "put the fan on speed {f}", "{ac} fan speed {f} {k}",
                      "pankha {f} speed pe {k}", "fan ki speed {f} {k}", "blower {f} {k}",
                      "change fan speed to {f}"],
    "INCREASE_VOLUME": ["increase the volume", "turn up the volume", "volume {bd}", "awaaz {bd}",
                        "volume thoda zyada {k}", "make it louder", "volume up {k}", "sound {bd}",
                        "awaaz tez {k}", "increase volume to {v}", "volume {v} tak {bd}"],
    "DECREASE_VOLUME": ["decrease the volume", "turn down the volume", "volume {km}", "awaaz {km}",
                        "volume thoda kam {k}", "make it quieter", "volume down {k}",
                        "reduce volume to {v}", "volume {v} tak {km}", "awaaz dheere {k}",
                        "lower the music"],
    "MUTE_AUDIO": ["mute", "mute the audio", "mute {k}", "volume mute {k}", "sound {bnd}",
                   "awaaz {bnd}", "silence the music", "audio off {k}", "music mute {k}",
                   "sab kuch mute {k}"],
    "PLAY_MEDIA": ["play music", "play the song", "start the music", "gaana {gc}", "music {gc}",
                   "play some songs", "song play {k}", "music start {k}", "resume the music",
                   "gaane chalu {k}", "play the radio", "koi accha gaana {gc}"],
    "NEXT_TRACK": ["next song", "play the next song", "skip this song", "next track",
                   "agla gaana {gc}", "agla gaana {k}", "agla gana laga do", "gaana change {k}",
                   "next gaana {gc}", "skip {k}", "skip the track", "ye gaana skip {k}"],
    "STOP_SPEAKING": ["stop speaking", "be quiet", "stop talking", "chup ho jao", "chup raho",
                      "bolna band {k}", "assistant chup {k}", "that is enough stop talking",
                      "bas ab chup ho jao", "enough talking", "tum chup ho jao"],
    # out-of-scope phrases -> reject class
    UNKNOWN: ["what is the weather today", "call mom", "navigate to the airport",
              "open the sunroof", "roll down the window", "tell me a joke", "what time is it",
              "mausam kaisa hai", "mummy ko call karo", "airport ka rasta batao", "sunroof kholo",
              "khidki khol do", "how far is the next petrol pump", "mujhe ek joke sunao",
              "turn on the headlights", "headlight on {k}", "wiper {gc}", "set an alarm for six",
              "kitna petrol bacha hai", "what is the capital of India", "order a pizza",
              "bhookh lagi hai"],
}

# Separately written, unseen phrases (not produced by any template above)
HANDWRITTEN = [
    ("Could you switch the air conditioner on", "AC_ON"), ("AC ko chalu kijiye", "AC_ON"),
    ("I want the AC running", "AC_ON"), ("ac on", "AC_ON"), ("AC shuru karo", "AC_ON"),
    ("please shut the AC", "AC_OFF"), ("AC ko band kijiye", "AC_OFF"), ("AC off", "AC_OFF"),
    ("I do not need the air conditioner anymore turn it off", "AC_OFF"),
    ("Temperature 22 kar do", "SET_TEMPERATURE"), ("set AC to twenty two degrees", "SET_TEMPERATURE"),
    ("temperature bees degree rakho", "SET_TEMPERATURE"), ("keep it at 24", "SET_TEMPERATURE"),
    ("temp 19 krdo", "SET_TEMPERATURE"),
    ("make the cabin a bit warmer", "INCREASE_TEMPERATURE"),
    ("temperature thoda upar karo", "INCREASE_TEMPERATURE"),
    ("heat it up a little", "INCREASE_TEMPERATURE"),
    ("make the cabin cooler", "DECREASE_TEMPERATURE"),
    ("temperature neeche karo", "DECREASE_TEMPERATURE"),
    ("bring the temperature down", "DECREASE_TEMPERATURE"),
    ("Fan speed teen kar do", "SET_FAN_SPEED"), ("fan to level 2", "SET_FAN_SPEED"),
    ("fan ko paanch pe chalao", "SET_FAN_SPEED"), ("blower speed 4 please", "SET_FAN_SPEED"),
    ("a bit louder please", "INCREASE_VOLUME"), ("volume zara badhana", "INCREASE_VOLUME"),
    ("raise the volume to 70", "INCREASE_VOLUME"), ("sound ko thoda tez kijiye", "INCREASE_VOLUME"),
    ("awaaz thodi kam karna", "DECREASE_VOLUME"), ("lower the sound a little", "DECREASE_VOLUME"),
    ("bring the volume down to 30", "DECREASE_VOLUME"), ("Volume thoda kam karo", "DECREASE_VOLUME"),
    ("mute everything", "MUTE_AUDIO"), ("sound off", "MUTE_AUDIO"),
    ("silence the speakers", "MUTE_AUDIO"), ("awaaz ko mute kar do", "MUTE_AUDIO"),
    ("Play something nice", "PLAY_MEDIA"), ("start playing songs", "PLAY_MEDIA"),
    ("kuch gaane suna do", "PLAY_MEDIA"), ("music on kar do", "PLAY_MEDIA"),
    ("Agla gaana chalao", "NEXT_TRACK"), ("skip to the next song", "NEXT_TRACK"),
    ("change the song", "NEXT_TRACK"), ("agle gaane pe jao", "NEXT_TRACK"),
    ("assistant quiet please", "STOP_SPEAKING"), ("bas ab chup", "STOP_SPEAKING"),
    ("that is enough from you", "STOP_SPEAKING"), ("please be silent", "STOP_SPEAKING"),
    ("how is the traffic ahead", "UNKNOWN_INTENT"), ("battery kitni bachi hai", "UNKNOWN_INTENT"),
    ("call my office", "UNKNOWN_INTENT"), ("open the glove box", "UNKNOWN_INTENT"),
    ("book a table for dinner", "UNKNOWN_INTENT"),
]


def fill(tpl, r):
    def rep(m):
        n = m.group(1)
        return NUMS[n](r) if n in NUMS else r.choice(SLOTS[n])
    return (r.choice(PRE) + re.sub(r"\{(\w+)\}", rep, tpl) + r.choice(POST)).strip()


def main():
    r, seen, rows = random.Random(SEED), set(), []
    for intent, tpls in TEMPLATES.items():
        for i, tpl in enumerate(tpls):
            grp, got, tries = f"{intent}_{i:02d}", 0, 0
            while got < PER_GROUP and tries < 400:
                tries += 1
                u = fill(tpl, r)
                if u.lower() in seen:
                    continue
                seen.add(u.lower())
                rows.append((u, intent, grp))
                got += 1
    DATA.mkdir(exist_ok=True)
    with open(DATA / "commands.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance", "intent", "template_group"])
        w.writerows(rows)
    with open(DATA / "handwritten_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utterance", "intent"])
        w.writerows(HANDWRITTEN)
    print(f"Wrote {len(rows)} utterances in {len({x[2] for x in rows})} template groups, "
          f"plus {len(HANDWRITTEN)} handwritten test phrases.")


if __name__ == "__main__":
    main()
