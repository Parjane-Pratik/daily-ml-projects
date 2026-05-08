"""
Speech recognition and text-to-speech module.

Supports voice input via microphone (SpeechRecognition) with automatic
fallback to keyboard text input when a microphone is unavailable.
Text-to-speech output uses pyttsx3 with print fallback.
"""

import logging

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """Handles voice-to-text conversion using the SpeechRecognition library."""

    def __init__(self, language="en-US"):
        self.language = language
        self._recognizer = None
        self._microphone = None
        self._available = self._initialize()

    def _initialize(self):
        """Initialize speech recognition. Returns True if available."""
        try:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
            self._microphone = sr.Microphone()
            logger.info("Speech recognition initialized successfully.")
            return True
        except ImportError:
            logger.warning(
                "speech_recognition not installed. Using text input fallback."
            )
            return False
        except Exception as e:
            logger.warning(
                f"Could not initialize microphone: {e}. Using text input fallback."
            )
            return False

    @property
    def is_available(self):
        return self._available

    def listen(self, timeout=5, phrase_time_limit=10):
        """
        Listen for voice input and return transcribed text.

        Parameters
        ----------
        timeout : int
            Seconds to wait for speech to start.
        phrase_time_limit : int
            Maximum seconds for a single phrase.

        Returns
        -------
        str or None
            Transcribed text, or None if recognition failed.
        """
        if not self._available:
            return None

        import speech_recognition as sr

        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening...")
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

            text = self._recognizer.recognize_google(audio, language=self.language)
            logger.info(f"Recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            logger.debug("No speech detected within timeout.")
            return None
        except sr.UnknownValueError:
            logger.debug("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None


class TextToSpeech:
    """Handles text-to-speech output using pyttsx3."""

    def __init__(self, rate=180, volume=1.0):
        self._engine = None
        self._available = self._initialize(rate, volume)

    def _initialize(self, rate, volume):
        """Initialize TTS engine. Returns True if available."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
            return True
        except ImportError:
            logger.warning("pyttsx3 not installed. Using print output only.")
            return False
        except Exception as e:
            logger.warning(f"Could not initialize TTS: {e}. Using print output only.")
            return False

    @property
    def is_available(self):
        return self._available

    def speak(self, text):
        """
        Convert text to speech and print to console.

        Parameters
        ----------
        text : str
            Text to synthesize.
        """
        print(f"Assistant: {text}")
        if self._available:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")


class VoiceIO:
    """Combined voice input/output handler with transparent text fallback."""

    def __init__(self):
        self.recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()

    def get_input(self, prompt="You: "):
        """
        Get user input via voice, falling back to keyboard text.

        Returns
        -------
        str
            User's input text.
        """
        if self.recognizer.is_available:
            text = self.recognizer.listen()
            if text:
                print(f"You: {text}")
                return text
            print("(Voice recognition failed, falling back to text input)")

        return input(prompt).strip()

    def respond(self, text):
        """Output the assistant's response via TTS and/or print."""
        self.tts.speak(text)
