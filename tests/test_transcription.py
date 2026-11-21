"""
Unit tests for BHAVIK Transcription Engine
"""

import pytest
from backend.src.transcription import TranscriptionEngine


@pytest.fixture
def engine():
    return TranscriptionEngine()


def test_initial_state(engine):
    state = engine.get_state()
    assert state["text"] == ""
    assert state["current_word"] == ""
    assert state["word_count"] == 0
    assert state["character_count"] == 0


def test_bhavik_spelling_sequence(engine):
    letters = ["B", "H", "A", "V", "I", "K"]
    for char in letters:
        engine.add_letter(char)

    assert engine.get_text() == "BHAVIK"
    assert engine.get_current_word() == "BHAVIK"
    assert engine.get_state()["word_count"] == 1


def test_space_and_multiple_words(engine):
    for char in "HELLO":
        engine.add_letter(char)
    engine.add_space()
    for char in "WORLD":
        engine.add_letter(char)

    assert engine.get_text() == "HELLO WORLD"
    assert engine.get_current_word() == "WORLD"
    assert engine.get_words() == ["HELLO", "WORLD"]
    assert engine.get_state()["word_count"] == 2


def test_consecutive_space_prevention(engine):
    engine.add_letter("A")
    engine.add_space()
    engine.add_space()
    engine.add_space()
    assert engine.get_text() == "A "


def test_backspace(engine):
    for char in "TEST":
        engine.add_letter(char)
    engine.backspace()
    assert engine.get_text() == "TES"
    assert engine.get_current_word() == "TES"


def test_clear(engine):
    for char in "BHAVIK":
        engine.add_letter(char)
    engine.clear()
    assert engine.get_text() == ""
    assert engine.get_current_word() == ""
    assert engine.get_state()["word_count"] == 0


def test_undo(engine):
    engine.add_letter("A")
    engine.add_letter("B")
    engine.undo()
    assert engine.get_text() == "A"
