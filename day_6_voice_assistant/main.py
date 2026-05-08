"""
Entry point for the Intelligent Voice Assistant.

Usage
-----
  python main.py                    # text mode
  python main.py --voice            # voice mode (requires microphone + pyttsx3)
  python main.py --debug            # enable verbose logging
  python main.py --db memory.db     # custom database path
"""

import argparse
import logging
import sys


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intelligent Voice-Activated Personal Assistant"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice I/O (requires microphone + pyttsx3)",
    )
    parser.add_argument(
        "--db",
        default="assistant_memory.db",
        metavar="PATH",
        help="SQLite database path (default: assistant_memory.db)",
    )
    parser.add_argument(
        "--model",
        default="models/intent_classifier.pkl",
        metavar="PATH",
        help="Intent classifier model path (default: models/intent_classifier.pkl)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.debug)

    # Local import so the module can be imported without side-effects
    from assistant import VoiceAssistant

    assistant = VoiceAssistant(
        db_path=args.db,
        model_path=args.model,
        use_voice=args.voice,
    )
    assistant.run()


if __name__ == "__main__":
    main()
