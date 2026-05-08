# Day 6: Intelligent Voice-Activated Personal Assistant with Contextual Memory

A full-stack AI assistant that combines **speech recognition**, **machine
learning–based intent classification**, **rule-based entity extraction**,
**contextual memory** (SQLite), and **task automation** (calendar, reminders,
weather, news, email).

---

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│  Voice / Text   │───▶│    NLU Pipeline       │───▶│  Task Executor   │
│    Input        │    │  TF-IDF + LogReg      │    │  (scheduling,    │
│ (SpeechRecog /  │    │  Intent Classifier    │    │   weather, news, │
│  keyboard)      │    │  + Regex Entity       │    │   reminders, …)  │
└─────────────────┘    │    Extractor          │    └────────┬─────────┘
                       └──────────┬───────────┘             │
                                  │ intent + entities        │ response
                       ┌──────────▼───────────┐             │
                       │  Contextual Memory   │◀────────────┘
                       │  (SQLite)            │
                       │  • conversations     │
                       │  • events/calendar   │
                       │  • reminders         │
                       │  • preferences       │
                       └──────────────────────┘
```

---

## Features

| Feature | Implementation |
|---|---|
| **Voice Recognition** | `SpeechRecognition` (Google STT) — falls back to text |
| **Text-to-Speech** | `pyttsx3` — falls back to `print` |
| **Intent Recognition** | TF-IDF + Logistic Regression (`scikit-learn`) |
| **Entity Extraction** | Regex patterns (dates, times, names, locations, email) |
| **Calendar Scheduling** | SQLite `events` table via `ContextualMemory` |
| **Reminders** | SQLite `reminders` table |
| **Weather** | [wttr.in](https://wttr.in) JSON API (no API key) |
| **News Headlines** | BBC News RSS feed (no API key) |
| **Email Drafting** | Draft preview; SMTP config via user preferences |
| **Contextual Memory** | Conversation history + upcoming events + reminders |

---

## ML Components

### Intent Classifier (`assistant/nlu.py`)

A scikit-learn **Pipeline** composed of:

1. `TfidfVectorizer` — unigram + bigram features, sublinear TF scaling  
2. `LogisticRegression` — multinomial, `lbfgs` solver

Trained on ~140 hand-crafted examples spanning **8 intent classes**:

| Intent | Example |
|---|---|
| `schedule_meeting` | *"Schedule a meeting with Pratik tomorrow at 10 AM"* |
| `check_schedule` | *"When is my next meeting?"* |
| `set_reminder` | *"Remind me to call mom tomorrow"* |
| `check_reminders` | *"What are my reminders?"* |
| `get_weather` | *"What's the weather in London?"* |
| `get_news` | *"Tell me today's headlines"* |
| `send_email` | *"Send an email to Alice about the project"* |
| `general_query` | *"Hello", "What time is it?", "Tell me a joke"* |

The trained model can be saved to disk with `joblib` (`--model` flag) for
instant reload on subsequent runs.

> **Upgrading to BERT**: replace `IntentClassifier` in `nlu.py` with a
> Hugging Face `pipeline("text-classification", model="distilbert-base-uncased")`
> fine-tuned on the same `INTENT_TRAINING_DATA`.

### Entity Extractor (`assistant/nlu.py`)

Rule-based regex patterns extract:

- **Dates** — `today`, `tomorrow`, weekday names, `next <day>`, `Month DD`  
- **Times** — `HH:MM`, `H AM/PM`, `noon`, `morning`, `afternoon`  
- **Person names** — capitalised words following relational prepositions  
- **Locations** — capitalised words following location prepositions  
- **Email addresses** — RFC-5321-like pattern  
- **`resolved_date`** — first date resolved to a `datetime.date` object

### Contextual Memory (`assistant/memory.py`)

SQLite-backed memory with four tables:

| Table | Purpose |
|---|---|
| `conversations` | Full turn-by-turn history with intent + entity JSON |
| `events` | Scheduled meetings / appointments |
| `reminders` | To-do items with optional due date and time |
| `preferences` | JSON key-value user preferences |

The `get_context_summary()` helper returns recent turns, upcoming events,
and pending reminders so every response is context-aware.

---

## Project Structure

```
day_6_voice_assistant/
├── assistant/
│   ├── __init__.py        # package entry: exports VoiceAssistant
│   ├── assistant.py       # VoiceAssistant orchestrator
│   ├── nlu.py             # IntentClassifier + EntityExtractor + NLUPipeline
│   ├── memory.py          # ContextualMemory (SQLite)
│   ├── tasks.py           # TaskExecutor (all intent handlers)
│   └── speech.py          # SpeechRecognizer + TextToSpeech + VoiceIO
├── main.py                # CLI entry point
├── demo.py                # Scripted demo (no microphone needed)
└── requirements.txt
```

---

## Installation

```bash
# Minimum (text mode, no voice):
pip install scikit-learn numpy joblib

# Full installation:
pip install -r requirements.txt

# macOS users may also need:
brew install portaudio
```

---

## Usage

### Run the demo (no microphone required)

```bash
cd day_6_voice_assistant
python demo.py
```

### Start the interactive assistant (text mode)

```bash
python main.py
```

### Start with voice I/O (microphone + pyttsx3 required)

```bash
python main.py --voice
```

### Additional CLI options

```
--db PATH      SQLite database path   (default: assistant_memory.db)
--model PATH   Intent model path      (default: models/intent_classifier.pkl)
--debug        Enable verbose logging
```

---

## Demo Walkthrough

```
You:       Schedule a meeting with Pratik tomorrow at 10 AM
Assistant: ✅ Meeting with Pratik has been scheduled on Tuesday, April 08, 2026 at 10 am (Event ID: 1).

You:       When is my next meeting?
Assistant: Your next event: Meeting with Pratik on Tuesday, April 08, 2026 at 10 am with Pratik.

You:       Remind me to review the project report on Friday at 3 PM
Assistant: ⏰ Reminder set: 'review the project report' on Friday, April 10, 2026 at 3 pm (Reminder ID: 1).

You:       What's the weather in London?
Assistant: 🌤 Weather in London:
             Condition:   Partly cloudy
             Temperature: 12°C / 54°F
             Humidity:    72%
             Wind Speed:  18 km/h
```

The assistant retains full context between turns — asking *"When is my next
meeting?"* returns the event scheduled earlier in the same session.

---

## Extending the Assistant

### Add a new intent

1. Add training sentences to `INTENT_TRAINING_DATA` in `nlu.py`.  
2. Add a handler method to `TaskExecutor` in `tasks.py`.  
3. Register the handler in `TaskExecutor._handlers`.

### Upgrade speech recognition to Whisper

```python
# In assistant/speech.py
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
text = result["text"]
```

### Connect real email sending (SMTP)

```python
# Store SMTP settings via the assistant
assistant.memory.set_preference("smtp_host", "smtp.gmail.com")
assistant.memory.set_preference("smtp_user", "you@gmail.com")
# Then update tasks.py _send_email to use smtplib
```

### IoT integration

Add an `iot_control` intent and handler that sends MQTT messages to smart
devices via the `paho-mqtt` library.

---

## Key Learning Points

- Building an end-to-end NLP pipeline with scikit-learn  
- Rule-based entity extraction with compiled regex patterns  
- Persistent context management with SQLite  
- Graceful degradation (voice → text, TTS → print)  
- Clean separation of concerns: NLU · Memory · Task execution · I/O
