"""
Task automation module.

Dispatches work to individual handlers based on detected intent:

  schedule_meeting   → calendar event in SQLite
  check_schedule     → query upcoming events
  set_reminder       → reminder in SQLite
  check_reminders    → query pending reminders
  get_weather        → wttr.in JSON API  (no API key required)
  get_news           → BBC RSS feed      (no API key required)
  send_email         → draft preview  (SMTP config via preferences)
  general_query      → greetings, time/date, help, jokes, …
"""

import json
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Dict, Optional

from .memory import ContextualMemory

logger = logging.getLogger(__name__)


def _fmt_date(d) -> str:
    """Human-readable date string from a date object or ISO string."""
    if d is None:
        return "date not specified"
    if isinstance(d, date):
        return d.strftime("%A, %B %d, %Y")
    try:
        return date.fromisoformat(str(d)).strftime("%A, %B %d, %Y")
    except (ValueError, TypeError):
        return str(d)


class TaskExecutor:
    """
    Execute assistant tasks given a structured NLU result dict.

    Parameters
    ----------
    memory : ContextualMemory
        Shared memory instance used to read/write events and reminders.
    """

    def __init__(self, memory: ContextualMemory):
        self.memory = memory
        self._handlers = {
            "schedule_meeting": self._schedule_meeting,
            "check_schedule": self._check_schedule,
            "set_reminder": self._set_reminder,
            "check_reminders": self._check_reminders,
            "get_weather": self._get_weather,
            "get_news": self._get_news,
            "send_email": self._send_email,
            "general_query": self._general_query,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, nlu_result: Dict) -> str:
        """
        Dispatch and execute the task encoded in *nlu_result*.

        Parameters
        ----------
        nlu_result : dict
            Output from ``NLUPipeline.process()``.

        Returns
        -------
        str
            Human-readable response text.
        """
        intent = nlu_result["intent"]
        entities = nlu_result["entities"]
        text = nlu_result["text"]
        context = self.memory.get_context_summary()

        handler = self._handlers.get(intent, self._fallback)
        return handler(text, entities, context)

    # ------------------------------------------------------------------
    # Individual task handlers
    # ------------------------------------------------------------------

    def _schedule_meeting(self, text: str, entities: Dict, context: Dict) -> str:
        participants = entities.get("persons", [])
        dates = entities.get("dates", [])
        times = entities.get("times", [])
        resolved_date = entities.get("resolved_date")

        title = (
            f"Meeting with {', '.join(participants)}" if participants else "Meeting"
        )
        event_time = times[0] if times else None

        event_id = self.memory.schedule_event(
            title=title,
            participants=participants or None,
            event_date=resolved_date,
            event_time=event_time,
            event_type="meeting",
        )

        parts = [f"✅ {title} has been scheduled"]
        if resolved_date:
            parts.append(f"on {_fmt_date(resolved_date)}")
        elif dates:
            parts.append(f"on {dates[0]}")
        if event_time:
            parts.append(f"at {event_time}")
        parts.append(f"(Event ID: {event_id})")
        return " ".join(parts) + "."

    def _check_schedule(self, text: str, entities: Dict, context: Dict) -> str:
        text_lower = text.lower()

        if "next" in text_lower or "upcoming" in text_lower:
            event = self.memory.get_next_event()
            if event:
                return self._fmt_event(event, prefix="Your next event:")
            return "You have no upcoming events scheduled."

        events = self.memory.get_upcoming_events(days_ahead=7)
        if not events:
            return "You have no upcoming events in the next 7 days."

        lines = ["📅 Your upcoming events:"]
        for i, ev in enumerate(events[:5], 1):
            lines.append(f"  {i}. {self._fmt_event_short(ev)}")
        if len(events) > 5:
            lines.append(f"  … and {len(events) - 5} more event(s).")
        return "\n".join(lines)

    def _set_reminder(self, text: str, entities: Dict, context: Dict) -> str:
        dates = entities.get("dates", [])
        times = entities.get("times", [])
        resolved_date = entities.get("resolved_date")

        # Extract the reminder topic from the command text
        reminder_text = text
        for kw in (
            "remind me to",
            "remind me about",
            "reminder to",
            "reminder for",
            "set reminder",
            "set a reminder",
        ):
            if kw in text.lower():
                idx = text.lower().index(kw) + len(kw)
                reminder_text = text[idx:].strip()
                # Strip trailing date/time fragments
                for fragment in dates + times:
                    reminder_text = re.sub(
                        re.escape(fragment), "", reminder_text, flags=re.IGNORECASE
                    ).strip()
                # Remove dangling prepositions left after stripping date/time
                while True:
                    cleaned = re.sub(
                        r"\s+(?:on|at|by|from|until|before|after)\s*$",
                        "",
                        reminder_text,
                        flags=re.IGNORECASE,
                    ).strip(" ,.")
                    if cleaned == reminder_text:
                        break
                    reminder_text = cleaned
                break

        if not reminder_text:
            reminder_text = "Reminder"

        event_time = times[0] if times else None
        reminder_id = self.memory.add_reminder(
            reminder_text=reminder_text,
            due_date=resolved_date,
            due_time=event_time,
        )

        parts = [f"⏰ Reminder set: '{reminder_text}'"]
        if resolved_date:
            parts.append(f"on {_fmt_date(resolved_date)}")
        elif dates:
            parts.append(f"on {dates[0]}")
        if event_time:
            parts.append(f"at {event_time}")
        parts.append(f"(Reminder ID: {reminder_id})")
        return " ".join(parts) + "."

    def _check_reminders(self, text: str, entities: Dict, context: Dict) -> str:
        reminders = self.memory.get_pending_reminders()
        if not reminders:
            return "You have no pending reminders. 🎉"

        lines = [f"📋 You have {len(reminders)} pending reminder(s):"]
        for i, rem in enumerate(reminders[:10], 1):
            due = ""
            if rem.get("due_date"):
                due += f" on {rem['due_date']}"
            if rem.get("due_time"):
                due += f" at {rem['due_time']}"
            lines.append(f"  {i}. {rem['reminder_text']}{due}")
        return "\n".join(lines)

    def _get_weather(self, text: str, entities: Dict, context: Dict) -> str:
        locations = entities.get("locations", [])
        city = (
            locations[0]
            if locations
            else self.memory.get_preference("default_city", "London")
        )
        return self._fetch_weather(city)

    def _fetch_weather(self, city: str) -> str:
        """Fetch current weather from wttr.in (no API key required)."""
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode())

            cc = data["current_condition"][0]
            desc = cc["weatherDesc"][0]["value"]
            temp_c = cc["temp_C"]
            temp_f = cc["temp_F"]
            humidity = cc["humidity"]
            wind = cc["windspeedKmph"]

            return (
                f"🌤 Weather in {city}:\n"
                f"  Condition:   {desc}\n"
                f"  Temperature: {temp_c}°C / {temp_f}°F\n"
                f"  Humidity:    {humidity}%\n"
                f"  Wind Speed:  {wind} km/h"
            )
        except Exception as e:
            logger.warning("Weather fetch failed: %s", e)
            return (
                f"Sorry, I couldn't fetch weather for {city} right now. "
                "Please check your internet connection."
            )

    def _get_news(self, text: str, entities: Dict, context: Dict) -> str:
        return self._fetch_news()

    def _fetch_news(self) -> str:
        """Fetch top-5 headlines from the BBC News RSS feed (no API key)."""
        url = "https://feeds.bbci.co.uk/news/rss.xml"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                content = resp.read().decode()

            root = ET.fromstring(content)
            channel = root.find("channel")
            items = channel.findall("item")[:5]

            lines = ["📰 Latest Headlines (BBC News):"]
            for i, item in enumerate(items, 1):
                title = item.findtext("title", "No title")
                lines.append(f"  {i}. {title}")
            return "\n".join(lines)

        except Exception as e:
            logger.warning("News fetch failed: %s", e)
            return (
                "Sorry, I couldn't fetch the latest news right now. "
                "Please check your internet connection."
            )

    def _send_email(self, text: str, entities: Dict, context: Dict) -> str:
        persons = entities.get("persons", [])
        email_addr = entities.get("email")
        recipient = email_addr or (persons[0] if persons else "recipient")

        # Extract subject from "about …" / "regarding …"
        subject = ""
        for kw in ("about", "regarding", "re:"):
            if kw in text.lower():
                idx = text.lower().index(kw) + len(kw)
                subject = text[idx:].strip().split(".")[0].strip()
                break
        if not subject:
            subject = "Message from your assistant"

        smtp_host = self.memory.get_preference("smtp_host")
        config_note = (
            f"SMTP host configured: {smtp_host}"
            if smtp_host
            else "Configure SMTP: set_preference('smtp_host', 'smtp.gmail.com')"
        )

        return (
            f"📧 Email draft ready:\n"
            f"  To:      {recipient}\n"
            f"  Subject: {subject}\n\n"
            f"  Note: {config_note}"
        )

    def _general_query(self, text: str, entities: Dict, context: Dict) -> str:
        text_lower = text.lower().strip()
        now = datetime.now()

        # Greetings
        if any(g in text_lower for g in ("hello", "hi ", "hey", "good morning",
                                          "good afternoon", "good evening")):
            hour = now.hour
            if hour < 12:
                salutation = "Good morning"
            elif hour < 17:
                salutation = "Good afternoon"
            else:
                salutation = "Good evening"
            return f"{salutation}! I'm your AI assistant. How can I help you today?"

        # Farewells
        if any(f in text_lower for f in ("bye", "goodbye", "farewell")):
            return "Goodbye! Have a great day! 👋"

        # Time / date
        if "time" in text_lower:
            return f"The current time is {now.strftime('%I:%M %p')}."
        if "date" in text_lower or "day is" in text_lower:
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        # Help
        if "help" in text_lower or "what can you do" in text_lower:
            return self._help_text()

        # Appreciation
        if "thank" in text_lower:
            return "You're welcome! Is there anything else I can help you with?"

        # Joke
        if "joke" in text_lower:
            return (
                "Why don't scientists trust atoms?\n"
                "Because they make up everything! 😄"
            )

        # Context-aware fallback
        last_intent = context.get("last_intent")
        if last_intent and last_intent != "general_query":
            topic = last_intent.replace("_", " ")
            return (
                f"I'm not sure I caught that. "
                f"Last time you were asking about {topic}. "
                "How can I help?"
            )
        return (
            "I'm not sure how to help with that. "
            "Say 'help' to see what I can do."
        )

    def _fallback(self, text: str, entities: Dict, context: Dict) -> str:
        return (
            "I didn't understand that. "
            "You can ask me to schedule meetings, check your calendar, "
            "set reminders, get the weather, or read the news."
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _fmt_event(self, event: Dict, prefix: str = "") -> str:
        parts = ([prefix] if prefix else []) + [event["title"]]
        if event.get("event_date"):
            parts.append(f"on {_fmt_date(event['event_date'])}")
        if event.get("event_time"):
            parts.append(f"at {event['event_time']}")
        participants = event.get("participants")
        if participants and isinstance(participants, list):
            parts.append(f"with {', '.join(participants)}")
        return " ".join(parts) + "."

    def _fmt_event_short(self, event: Dict) -> str:
        title = event["title"]
        details = [
            d
            for d in (event.get("event_date", ""), event.get("event_time", ""))
            if d
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{title}{suffix}"

    @staticmethod
    def _help_text() -> str:
        return (
            "🤖 Here's what I can do:\n"
            "  📅 Schedule meetings  — 'Schedule a meeting with John tomorrow at 10 AM'\n"
            "  🗓  Check schedule    — 'What's my next meeting?' / 'Show my schedule'\n"
            "  ⏰ Set reminders     — 'Remind me to call mom tomorrow'\n"
            "  📋 Check reminders   — 'What are my reminders?'\n"
            "  🌤 Get weather       — 'What's the weather in London?'\n"
            "  📰 Read news         — 'What are the latest headlines?'\n"
            "  📧 Draft emails      — 'Send an email to John about the project'\n"
            "  🕐 Current time      — 'What time is it?'\n"
            "  📆 Today's date      — 'What is today's date?'"
        )
