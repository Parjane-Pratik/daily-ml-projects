"""
Main assistant orchestrator.

Architecture
------------
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Voice / Text  │───▶│  NLU Pipeline    │───▶│ Task Executor   │
│     Input       │    │ Intent + Entities│    │ (calendar,      │
└─────────────────┘    └──────────────────┘    │  weather, …)    │
                                │               └────────┬────────┘
                       ┌────────▼────────┐               │
                       │   Contextual    │◀──────────────┘
                       │ Memory (SQLite) │
                       └─────────────────┘

Usage
-----
>>> from assistant import VoiceAssistant
>>> va = VoiceAssistant()
>>> va.process_input("Schedule a meeting with Pratik tomorrow at 10 AM")
'✅ Meeting with Pratik has been scheduled on …'
>>> va.run()   # interactive loop
"""

import logging
from typing import Optional

from .memory import ContextualMemory
from .nlu import NLUPipeline
from .tasks import TaskExecutor

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Intelligent Voice-Activated Personal Assistant with Contextual Memory.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    model_path : str, optional
        Path to persist/load the trained intent classifier.
    use_voice : bool
        Enable microphone input and TTS output.
    context_window : int
        Number of recent turns kept in the active context.
    """

    EXIT_COMMANDS = frozenset({"exit", "quit", "bye", "goodbye", "stop"})

    def __init__(
        self,
        db_path: str = "assistant_memory.db",
        model_path: Optional[str] = None,
        use_voice: bool = False,
        context_window: int = 10,
    ):
        self.memory = ContextualMemory(db_path=db_path, context_window=context_window)
        self.nlu = NLUPipeline(model_path=model_path)
        self.executor = TaskExecutor(memory=self.memory)
        self.use_voice = use_voice
        self._io = None

        if use_voice:
            from .speech import VoiceIO

            self._io = VoiceIO()

        logger.info("VoiceAssistant initialised (voice=%s).", use_voice)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_input(self, user_input: str) -> str:
        """
        Process a single user utterance and return a response.

        Pipeline
        --------
        1. Run NLU  → intent + entities
        2. Store user turn in contextual memory
        3. Execute the appropriate task
        4. Store assistant response in contextual memory

        Parameters
        ----------
        user_input : str

        Returns
        -------
        str
            The assistant's response.
        """
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you please repeat?"

        nlu_result = self.nlu.process(user_input)
        intent = nlu_result["intent"]
        confidence = nlu_result["confidence"]

        logger.info(
            "Intent: %s (%.2f) | Entities: %s",
            intent,
            confidence,
            nlu_result["entities"],
        )

        # Persist user turn
        self.memory.add_conversation_turn(
            role="user",
            message=user_input,
            intent=intent,
            entities=nlu_result["entities"],
        )

        # Execute task
        response = self.executor.execute(nlu_result)

        # Persist assistant turn
        self.memory.add_conversation_turn(role="assistant", message=response)

        return response

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the interactive assistant loop.

        Accepts text or voice input depending on ``use_voice``.
        Type/say ``exit`` or ``quit`` to stop.
        """
        print("\n" + "=" * 60)
        print("🤖  Intelligent Voice Assistant — Ready!")
        print("=" * 60)
        if self.use_voice and self._io:
            print("🎤  Voice mode enabled. Speak your command.")
        else:
            print("⌨️   Text mode. Type your command (or 'exit' to quit).")
        print("=" * 60 + "\n")

        self._output(self.process_input("hello"))

        while True:
            try:
                user_input = self._get_input()
                if not user_input:
                    continue
                if user_input.lower().strip() in self.EXIT_COMMANDS:
                    self._output("Goodbye! Have a wonderful day! 👋")
                    break
                self._output(self.process_input(user_input))

            except KeyboardInterrupt:
                print("\n\nAssistant: Goodbye! 👋")
                break
            except Exception as e:
                logger.error("Unexpected error: %s", e, exc_info=True)
                self._output("I encountered an error. Please try again.")

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _get_input(self) -> str:
        if self.use_voice and self._io:
            return self._io.get_input()
        return input("You: ").strip()

    def _output(self, text: str) -> None:
        if self.use_voice and self._io:
            self._io.respond(text)
        else:
            print(f"\nAssistant: {text}\n")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_context_summary(self) -> dict:
        """Return a snapshot of the current assistant context."""
        return self.memory.get_context_summary()
