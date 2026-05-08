"""
Contextual memory module backed by SQLite.

Stores and retrieves:
  - Conversation history  (multi-turn context window)
  - Scheduled events      (meetings, appointments, calls)
  - Reminders             (with optional due date / time)
  - User preferences      (key-value store)

All timestamps are stored as ISO-8601 strings so the database is portable.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_default(obj):
    """JSON serialiser for types not handled by the standard library."""
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


DEFAULT_DB_PATH = "assistant_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    role        TEXT    NOT NULL,   -- 'user' | 'assistant'
    message     TEXT    NOT NULL,
    intent      TEXT,
    entities    TEXT                -- JSON blob
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    event_type   TEXT NOT NULL DEFAULT 'meeting',
    title        TEXT NOT NULL,
    participants TEXT,              -- JSON array
    event_date   TEXT,             -- ISO date  YYYY-MM-DD
    event_time   TEXT,             -- free-form  e.g. "10 AM"
    notes        TEXT,
    status       TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS reminders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    reminder_text TEXT NOT NULL,
    due_date      TEXT,
    due_time      TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS preferences (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,       -- JSON-encoded value
    updated_at TEXT NOT NULL
);
"""


class ContextualMemory:
    """
    SQLite-backed memory for the voice assistant.

    Parameters
    ----------
    db_path : str
        Filesystem path for the SQLite database file.
    context_window : int
        How many recent conversation turns to return by default.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, context_window: int = 10):
        self.db_path = db_path
        self.context_window = context_window
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        """Yield a committed (or rolled-back) SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        logger.info("Database ready: %s", self.db_path)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_conversation_turn(
        self,
        role: str,
        message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict] = None,
    ) -> None:
        """
        Append a conversation turn.

        Parameters
        ----------
        role : str
            ``'user'`` or ``'assistant'``.
        message : str
        intent : str, optional
            Detected intent (user turns only).
        entities : dict, optional
            Extracted entity dict (user turns only).
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (timestamp, role, message, intent, entities)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    role,
                    message,
                    intent,
                    json.dumps(entities, default=_json_default) if entities else None,
                ),
            )

    def get_recent_context(self, n: Optional[int] = None) -> List[Dict]:
        """
        Return the *n* most recent conversation turns in chronological order.

        Parameters
        ----------
        n : int, optional
            Defaults to ``self.context_window``.
        """
        limit = n or self.context_window
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, message, intent, timestamp"
                " FROM conversations"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_last_intent(self) -> Optional[str]:
        """Return the intent from the most recent user turn."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT intent FROM conversations"
                " WHERE role = 'user' AND intent IS NOT NULL"
                " ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["intent"] if row else None

    # ------------------------------------------------------------------
    # Events / Calendar
    # ------------------------------------------------------------------

    def schedule_event(
        self,
        title: str,
        participants: Optional[List[str]] = None,
        event_date=None,
        event_time: Optional[str] = None,
        event_type: str = "meeting",
        notes: Optional[str] = None,
    ) -> int:
        """
        Persist a new event and return its row ID.

        Parameters
        ----------
        title : str
        participants : list of str, optional
        event_date : datetime.date or str, optional
        event_time : str, optional
        event_type : str
        notes : str, optional
        """
        if isinstance(event_date, date):
            event_date = event_date.isoformat()

        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO events"
                " (created_at, event_type, title, participants, event_date, event_time, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    event_type,
                    title,
                    json.dumps(participants) if participants else None,
                    str(event_date) if event_date else None,
                    event_time,
                    notes,
                ),
            )
            return cursor.lastrowid

    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """
        Return active events whose date is today or within *days_ahead* days.

        Events without a date are included so nothing is silently lost.
        """
        today = date.today().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events"
                " WHERE status = 'active'"
                " AND (event_date IS NULL OR event_date >= ?)"
                " ORDER BY event_date ASC, event_time ASC"
                " LIMIT 20",
                (today,),
            ).fetchall()

        events = []
        for row in rows:
            e = dict(row)
            if e["participants"]:
                e["participants"] = json.loads(e["participants"])
            events.append(e)
        return events

    def get_next_event(self) -> Optional[Dict]:
        """Return the single nearest upcoming event, or ``None``."""
        events = self.get_upcoming_events(days_ahead=30)
        return events[0] if events else None

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    def add_reminder(
        self,
        reminder_text: str,
        due_date=None,
        due_time: Optional[str] = None,
    ) -> int:
        """
        Persist a reminder and return its row ID.

        Parameters
        ----------
        reminder_text : str
        due_date : datetime.date or str, optional
        due_time : str, optional
        """
        if isinstance(due_date, date):
            due_date = due_date.isoformat()

        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO reminders (created_at, reminder_text, due_date, due_time)"
                " VALUES (?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    reminder_text,
                    str(due_date) if due_date else None,
                    due_time,
                ),
            )
            return cursor.lastrowid

    def get_pending_reminders(self) -> List[Dict]:
        """Return all pending reminders ordered by due date."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders"
                " WHERE status = 'pending'"
                " ORDER BY due_date ASC, due_time ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_reminder_done(self, reminder_id: int) -> None:
        """Mark a reminder as completed."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'done' WHERE id = ?",
                (reminder_id,),
            )

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Upsert a user preference."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at)"
                " VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat()),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Return a preference value, or *default* if not set."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    def get_all_preferences(self) -> Dict[str, Any]:
        """Return all stored preferences as a dict."""
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # ------------------------------------------------------------------
    # Context summary
    # ------------------------------------------------------------------

    def get_context_summary(self) -> Dict:
        """
        Return a snapshot of the current context suitable for response
        generation.

        Keys
        ----
        recent_conversation : list
        last_intent         : str or None
        upcoming_events     : list
        pending_reminders   : list
        """
        return {
            "recent_conversation": self.get_recent_context(5),
            "last_intent": self.get_last_intent(),
            "upcoming_events": self.get_upcoming_events(days_ahead=3),
            "pending_reminders": self.get_pending_reminders()[:3],
        }

    def clear_history(self) -> None:
        """Delete all conversation history (events and reminders are kept)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM conversations")
        logger.info("Conversation history cleared.")
