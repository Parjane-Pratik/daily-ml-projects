"""
Natural Language Understanding (NLU) module.

ML Model
--------
Intent classification uses a scikit-learn Pipeline:
  TfidfVectorizer (unigrams + bigrams) → LogisticRegression (multinomial)

The model is trained at startup on embedded labelled examples and can be
persisted to disk with joblib so subsequent runs load instantly.

Entity Extraction
-----------------
Rule-based regex patterns extract:
  - Dates  (today, tomorrow, weekday names, month+day, numeric)
  - Times  (HH:MM, H AM/PM, noon, morning, …)
  - Person names (capitalised words following prepositions)
  - Locations    (capitalised words following location prepositions)
  - Email addresses
"""

import logging
import os
import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training corpus for intent classification
# ---------------------------------------------------------------------------

INTENT_TRAINING_DATA: Dict[str, List[str]] = {
    "schedule_meeting": [
        "schedule a meeting with John tomorrow at 10 AM",
        "set up a meeting with Pratik on Monday at 3 PM",
        "book a meeting for tomorrow afternoon",
        "I need to schedule a meeting",
        "can you set up a call with Sarah next Friday at 2 PM",
        "schedule appointment with doctor on Thursday at 9 AM",
        "plan a meeting with the team for next week",
        "arrange a conference call with clients on Wednesday",
        "create a calendar event for a standup tomorrow at 9 AM",
        "add a meeting with manager on Friday at 11 AM",
        "I want to schedule a call with Raj at 4 PM today",
        "set a meeting tomorrow morning with Alice",
        "book a slot with Bob on Tuesday at 2 PM",
        "organize a team meeting for Monday",
        "put a meeting in my calendar with Priya on Thursday",
        "schedule standup with the engineering team at 9 AM",
        "set up a one-on-one with my manager next week",
        "calendar invite for project review on Monday at noon",
        "book a demo call with the client on Friday afternoon",
        "arrange a sync with Alice and Bob tomorrow at 11 AM",
    ],
    "check_schedule": [
        "what is my schedule for today",
        "when is my next meeting",
        "show me my calendar",
        "what do I have tomorrow",
        "any meetings today",
        "list all my upcoming meetings",
        "what is on my agenda",
        "do I have any appointments",
        "show my schedule for this week",
        "what meetings are coming up",
        "am I free tomorrow",
        "tell me my next appointment",
        "check my schedule",
        "what events do I have",
        "show upcoming events",
        "what is scheduled for next week",
        "do I have anything on Monday",
        "show me my agenda for today",
        "list upcoming calendar events",
        "what appointments do I have this week",
    ],
    "get_weather": [
        "what is the weather in Mumbai",
        "how is the weather today",
        "tell me the weather forecast for London",
        "is it going to rain tomorrow in Delhi",
        "what is the temperature in New York",
        "weather update for Paris",
        "how hot is it in Dubai",
        "will it snow in Chicago tomorrow",
        "current weather in Tokyo",
        "weather conditions in Sydney",
        "what should I wear today based on weather",
        "give me weather report",
        "climate update for Bangalore",
        "is it cold in Berlin",
        "forecast for San Francisco",
        "check the weather for me",
        "what is the weather like outside",
        "temperature today",
        "will it be sunny tomorrow",
        "what is the humidity in Singapore",
    ],
    "set_reminder": [
        "remind me to take medicine at 8 PM",
        "set a reminder for my meeting at 10 AM",
        "remind me to call mom tomorrow",
        "add a reminder to submit the report by Friday",
        "set an alarm to wake up at 6 AM",
        "don't let me forget the dentist appointment",
        "remind me to exercise at 7 AM",
        "set reminder for bill payment on 15th",
        "notify me about the deadline tomorrow",
        "remind me to buy groceries this evening",
        "create a reminder to review code at 3 PM",
        "remind me about the presentation tomorrow morning",
        "set reminder to call the bank",
        "don't forget to send the email",
        "remind me to water the plants at 8 AM",
        "alert me to check emails at noon",
        "set a notification for 5 PM today",
        "remind me about the interview on Wednesday",
        "create reminder to follow up with the client",
        "remind me to take a break every two hours",
    ],
    "check_reminders": [
        "what are my reminders",
        "show all reminders",
        "list my reminders",
        "what did I set as reminders",
        "do I have any reminders today",
        "check my reminders",
        "what have I been reminded about",
        "show pending reminders",
        "what should I remember",
        "any upcoming reminders",
        "tell me my reminders for today",
        "upcoming reminders list",
        "what reminders are due",
        "show me what I need to do",
        "list all notifications",
        "pending tasks and reminders",
        "show my to-do list",
        "what do I need to remember",
        "list all my alerts",
        "check pending notifications",
    ],
    "send_email": [
        "send an email to John about the project update",
        "compose an email to Sarah",
        "write an email to the team about the meeting",
        "send a message to Raj about tomorrow's presentation",
        "email Alice the report",
        "draft an email to manager about leave",
        "send mail to client about the proposal",
        "write to Bob about the budget review",
        "compose message to HR about joining",
        "send email to info@company.com",
        "mail the invoice to the vendor",
        "write an email to the support team",
        "send a follow-up email to the client",
        "email Priya the meeting notes",
        "compose a thank you email",
        "draft an apology email to the customer",
        "send weekly report to the team",
        "write a resignation letter and email it",
        "compose an introduction email for new team members",
        "send project status update by email",
    ],
    "get_news": [
        "what are the latest news",
        "tell me today's headlines",
        "any tech news",
        "give me news update",
        "what happened in the world today",
        "latest sports news",
        "top stories today",
        "what is in the news",
        "current events",
        "news about technology",
        "business news update",
        "breaking news",
        "show me headlines",
        "what is trending today",
        "news from BBC",
        "read me the morning news",
        "what are the top stories",
        "science and technology news",
        "political news today",
        "global headlines",
    ],
    "general_query": [
        "hello",
        "hi there",
        "how are you",
        "what can you do",
        "help me",
        "what is the time",
        "what is today's date",
        "tell me a joke",
        "good morning",
        "good night",
        "thanks",
        "thank you",
        "goodbye",
        "bye",
        "who are you",
        "what day is it",
        "what year is it",
        "how do you work",
        "what are your capabilities",
        "stop",
    ],
}


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """
    ML-based intent classifier using a TF-IDF + Logistic Regression pipeline.

    The model is trained on INTENT_TRAINING_DATA at instantiation and optionally
    persisted to disk via joblib for fast subsequent loads.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_corpus(self) -> Tuple[List[str], List[str]]:
        """Flatten INTENT_TRAINING_DATA into parallel (texts, labels) lists."""
        texts, labels = [], []
        for intent, examples in INTENT_TRAINING_DATA.items():
            for example in examples:
                texts.append(example.lower())
                labels.append(intent)
        return texts, labels

    def _train(self) -> None:
        """Train the intent pipeline (or load from disk if available)."""
        # Try loading a pre-trained model first
        if self.model_path and os.path.exists(self.model_path):
            try:
                saved = joblib.load(self.model_path)
                self.pipeline = saved["pipeline"]
                self.label_encoder = saved["label_encoder"]
                logger.info("Loaded saved intent classifier from %s", self.model_path)
                return
            except Exception as e:
                logger.warning("Could not load saved model (%s). Retraining.", e)

        texts, labels = self._build_corpus()

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=5000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=1.0,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        y = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(texts, y)
        logger.info("Intent classifier trained on %d examples.", len(texts))

        # Persist to disk if a path was provided
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(
                {"pipeline": self.pipeline, "label_encoder": self.label_encoder},
                self.model_path,
            )
            logger.info("Saved intent classifier to %s", self.model_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the intent for *text*.

        Parameters
        ----------
        text : str

        Returns
        -------
        (intent, confidence) : tuple[str, float]
        """
        proba = self.pipeline.predict_proba([text.lower()])[0]
        predicted_idx = int(np.argmax(proba))
        confidence = float(proba[predicted_idx])
        intent = self.label_encoder.inverse_transform([predicted_idx])[0]
        return intent, confidence


# ---------------------------------------------------------------------------
# Entity extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """
    Rule-based entity extractor using compiled regex patterns.

    Extracts
    --------
    dates      : absolute and relative date expressions
    times      : clock times, period-of-day keywords
    persons    : capitalised names preceded by relational prepositions
    locations  : capitalised place names preceded by location prepositions
    email      : email addresses
    resolved_date : the first date resolved to a ``datetime.date`` object
    """

    _DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    _MONTHS = (
        r"(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )

    DATE_PATTERNS = [
        re.compile(r"\b(today|tomorrow|yesterday)\b", re.IGNORECASE),
        re.compile(rf"\b(next\s+{_DAYS})\b", re.IGNORECASE),
        re.compile(rf"\b(this\s+{_DAYS})\b", re.IGNORECASE),
        re.compile(rf"\b({_DAYS})\b", re.IGNORECASE),
        re.compile(
            rf"\b({_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s+\d{{4}})?)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b"),
    ]

    TIME_PATTERNS = [
        re.compile(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2}\s*(?:AM|PM))\b", re.IGNORECASE),
        re.compile(r"\b(noon|midnight|morning|afternoon|evening|night)\b", re.IGNORECASE),
    ]

    EMAIL_RE = re.compile(
        r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b"
    )
    PERSON_RE = re.compile(
        r"\b(?:with|to|from|for|meet|call|email)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )
    LOCATION_RE = re.compile(
        r"\b(?:in|at|for|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
    )

    _DAY_MAP = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def extract(self, text: str) -> Dict:
        """
        Extract entities from *text*.

        Parameters
        ----------
        text : str

        Returns
        -------
        dict
            Keys: dates, times, persons, locations, email, resolved_date, raw_text
        """
        entities: Dict = {
            "dates": [],
            "times": [],
            "persons": [],
            "locations": [],
            "email": None,
            "resolved_date": None,
            "raw_text": text,
        }

        # Dates (search on lowercased text, deduplicate via dict)
        seen: Dict[str, None] = {}
        for pat in self.DATE_PATTERNS:
            for m in pat.findall(text):
                val = m.strip().lower()
                if val and val not in seen:
                    seen[val] = None
                    entities["dates"].append(val)

        # Times
        seen = {}
        for pat in self.TIME_PATTERNS:
            for m in pat.findall(text):
                val = m.strip().lower()
                if val and val not in seen:
                    seen[val] = None
                    entities["times"].append(val)

        # Email (original-case text)
        email_match = self.EMAIL_RE.search(text)
        if email_match:
            entities["email"] = email_match.group(1)

        # Person names (original-case text)
        entities["persons"] = list(
            dict.fromkeys(self.PERSON_RE.findall(text))
        )

        # Locations (original-case text)
        entities["locations"] = list(
            dict.fromkeys(self.LOCATION_RE.findall(text))
        )

        # Resolve first date expression to a date object
        entities["resolved_date"] = self._resolve_date(entities["dates"])

        return entities

    def _resolve_date(self, date_strings: List[str]) -> Optional[date]:
        """Resolve the first recognisable relative date string to a date."""
        today = date.today()
        for ds in date_strings:
            ds = ds.lower().strip()
            if ds == "today":
                return today
            if ds == "tomorrow":
                return today + timedelta(days=1)
            if ds == "yesterday":
                return today - timedelta(days=1)

            # Weekday references ("monday", "next friday", "this wednesday")
            for day_name, day_num in self._DAY_MAP.items():
                if day_name in ds:
                    days_ahead = (day_num - today.weekday()) % 7 or 7
                    if "next" in ds:
                        days_ahead += 7
                    return today + timedelta(days=days_ahead)
        return None


# ---------------------------------------------------------------------------
# Combined NLU pipeline
# ---------------------------------------------------------------------------


class NLUPipeline:
    """
    End-to-end NLU pipeline: intent classification + entity extraction.

    Usage
    -----
    >>> nlu = NLUPipeline()
    >>> result = nlu.process("Schedule a meeting with Pratik tomorrow at 10 AM")
    >>> result["intent"]
    'schedule_meeting'
    >>> result["confidence"]
    0.97...
    """

    def __init__(self, model_path: Optional[str] = None):
        self.intent_classifier = IntentClassifier(model_path=model_path)
        self.entity_extractor = EntityExtractor()

    def process(self, text: str) -> Dict:
        """
        Run the full NLU pipeline on *text*.

        Parameters
        ----------
        text : str

        Returns
        -------
        dict
            Keys: text, intent, confidence, entities
        """
        intent, confidence = self.intent_classifier.predict(text)
        entities = self.entity_extractor.extract(text)

        return {
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
        }
