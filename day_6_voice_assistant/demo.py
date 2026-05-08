"""
Demo script — Intelligent Voice Assistant showcase.

Runs a scripted conversation that exercises every major feature:
  • Meeting scheduling and calendar recall (contextual memory)
  • Reminders
  • Weather lookup
  • News headlines
  • Email drafting
  • General queries (time, date, help, jokes)

No microphone is required; all input is text-based.
"""

import logging
import os
import sys

# Make the package importable when run directly from this directory
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)

# Scripted conversation that mirrors the problem-statement demo scenario
DEMO_SCRIPT = [
    "Hello! What can you do?",
    "Schedule a meeting with Pratik tomorrow at 10 AM",
    "When is my next meeting?",
    "What's my schedule for this week?",
    "Remind me to review the project report on Friday at 3 PM",
    "What are my reminders?",
    "What's the weather in London?",
    "What are the latest news headlines?",
    "Send an email to Alice about the project update",
    "What time is it?",
    "What is today's date?",
    "Tell me a joke",
    "Thank you",
    "Goodbye",
]


def _separator(char: str = "─", width: int = 70) -> str:
    return char * width


def run_demo() -> None:
    from assistant import VoiceAssistant

    print("\n" + _separator("="))
    print("  INTELLIGENT VOICE ASSISTANT — DEMO")
    print("  Showcasing: NLP · Intent Recognition · Contextual Memory")
    print(_separator("=") + "\n")

    # Temporary database so the demo is self-contained and repeatable
    demo_db = "/tmp/demo_voice_assistant.db"
    if os.path.exists(demo_db):
        os.remove(demo_db)

    assistant = VoiceAssistant(
        db_path=demo_db,
        model_path=None,  # train in memory; no file I/O needed for demo
        use_voice=False,
    )

    for utterance in DEMO_SCRIPT:
        print(_separator())
        print(f"You: {utterance}")
        response = assistant.process_input(utterance)
        print(f"\nAssistant: {response}\n")

    # ------------------------------------------------------------------
    # Demonstrate contextual memory
    # ------------------------------------------------------------------
    print(_separator("="))
    print("📊  CONTEXTUAL MEMORY SNAPSHOT")
    print(_separator("="))

    ctx = assistant.get_context_summary()

    upcoming = ctx["upcoming_events"]
    print(f"\n📅  Upcoming Events ({len(upcoming)}):")
    if upcoming:
        for ev in upcoming:
            date_str = ev.get("event_date") or "TBD"
            time_str = ev.get("event_time") or "TBD"
            print(f"   • {ev['title']}  —  {date_str} at {time_str}")
    else:
        print("   (none)")

    reminders = ctx["pending_reminders"]
    print(f"\n⏰  Pending Reminders ({len(reminders)}):")
    if reminders:
        for rem in reminders:
            due = rem.get("due_date") or "TBD"
            print(f"   • {rem['reminder_text']}  (due: {due})")
    else:
        print("   (none)")

    recent = ctx["recent_conversation"]
    print(f"\n💬  Last {len(recent)} Conversation Turn(s):")
    for turn in recent:
        label = "You      " if turn["role"] == "user" else "Assistant"
        msg = turn["message"]
        truncated = msg[:75] + "…" if len(msg) > 75 else msg
        print(f"   [{label}]: {truncated}")

    print("\n" + _separator("="))
    print("✅  Demo complete! Contextual memory persisted across the session.")
    print(_separator("=") + "\n")

    # Clean up temporary database
    if os.path.exists(demo_db):
        os.remove(demo_db)


if __name__ == "__main__":
    run_demo()
