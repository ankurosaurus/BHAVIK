"""
BHAVIK Continuous Transcription Engine
Manages letter accumulation, tokenization, word boundaries, editing, and state.
"""

from typing import Any, Dict, List


class TranscriptionEngine:
    """Manages continuous text construction from accepted fingerspelling gestures."""

    def __init__(self) -> None:
        self.text: str = ""
        self.history: List[str] = [""]
        self.last_action: str = "INIT"

    def add_letter(self, letter: str) -> str:
        """Appends a valid recognized letter."""
        if not letter or len(letter) != 1:
            return self.text

        letter = letter.upper()
        self.text += letter
        self.history.append(self.text)
        self.last_action = f"ADD_{letter}"
        return self.text

    def add_space(self) -> str:
        """Inserts a space to begin a new word (prevents multiple consecutive spaces)."""
        if self.text and not self.text.endswith(" "):
            self.text += " "
            self.history.append(self.text)
            self.last_action = "SPACE"
        return self.text

    def backspace(self) -> str:
        """Deletes the last character."""
        if self.text:
            self.text = self.text[:-1]
            self.history.append(self.text)
            self.last_action = "BACKSPACE"
        return self.text

    def clear(self) -> str:
        """Clears the transcription buffer."""
        self.text = ""
        self.history.append("")
        self.last_action = "CLEAR"
        return self.text

    def undo(self) -> str:
        """Reverts to the previous transcription state if available."""
        if len(self.history) > 1:
            self.history.pop()
            self.text = self.history[-1]
            self.last_action = "UNDO"
        return self.text

    def get_text(self) -> str:
        """Returns the full continuous transcription text."""
        return self.text

    def get_current_word(self) -> str:
        """Returns the word currently being constructed (characters after the last space)."""
        if not self.text or self.text.endswith(" "):
            return ""
        return self.text.split(" ")[-1]

    def get_words(self) -> List[str]:
        """Returns all completed words in the transcription."""
        return [w for w in self.text.split(" ") if w]

    def get_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the current transcription state."""
        words = self.get_words()
        return {
            "text": self.text,
            "current_word": self.get_current_word(),
            "character_count": len(self.text),
            "word_count": len(words),
            "last_action": self.last_action,
        }
