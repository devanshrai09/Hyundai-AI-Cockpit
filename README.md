# Hyundai AI Cockpit — final integrated prototype

## Routing architecture

1. Microphone audio is transcribed by local Indic-Transcribe Flex.
2. `demo.py` classifies known cockpit commands and safely simulates actions.
3. If the classifier returns `UNKNOWN_INTENT` (and the utterance is not negated), `llm_fallback.py` calls the OpenAI API only when an API key is configured and internet connectivity is available; otherwise it calls the local Ollama API at `http://localhost:11434/api/chat`.
4. The response is displayed on the Flask dashboard and spoken using the verified fresh-SAPI5-engine-per-utterance TTS method.

Vehicle actions are simulated; this prototype does not control a real car.

## Windows setup

Run these commands in PowerShell from this folder. Python 3.10/3.11 is recommended for the ASR model's dependency compatibility.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-asr-flex.txt
```

Follow the model repository's own installation instructions for any extra dependencies:
https://huggingface.co/bodhan-ai/indic-transcribe-flex

The model's actual files must be present under `models/indic-transcribe-flex/`. The ZIP contains a placeholder only, not the large model weights.

## Local LLM (Ollama)

Install Ollama, then run:

```powershell
ollama pull llama3.2:3b
ollama serve
```

If Ollama is already running as a Windows service/app, do not start a second server; just use `ollama list` to verify the model exists. `llm_fallback.py` defaults to model `llama3.2:3b` and endpoint `http://localhost:11434`.

Optional: copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want online API fallback when internet is available. Leave it unset to use Ollama only.

## Run

```powershell
python cockpit_dashboard.py --model-dir ".\models\indic-transcribe-flex" --language hi
```

Open http://127.0.0.1:5000. Say `you may switch off` to stop the voice loop.

## Useful checks

```powershell
python -c "import sys; sys.path.insert(0, r'.\models\indic-transcribe-flex'); from indic_transcribe import IndicTranscribe; print('ASR import OK')"
ollama list
```

The dashboard logs show whether a response came from `local-ml`, `offline-ollama`, or `online-api`.
