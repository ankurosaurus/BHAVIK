/**
 * BHAVIK Transcription Controller
 * Manages transcript display, clipboard actions, and browser Speech Synthesis (TTS).
 */

class TranscriptionController {
  constructor(displayElement, options = {}) {
    this.display = displayElement;
    this.text = '';
    this.currentWord = '';
    this.onActionRequest = options.onActionRequest || null;

    this.speechSynth = window.speechSynthesis || null;
    this.initKeyboardShortcuts();
  }

  updateState(state) {
    if (!state) return;
    this.text = state.text || '';
    this.currentWord = state.current_word || '';

    // Update DOM display with blinking cursor
    this.render();
  }

  addLetter(letter) {
    if (!letter) return;
    this.text += letter.toUpperCase();
    this.updateCurrentWord();
    this.render();
  }

  addSpace() {
    if (this.text && !this.text.endsWith(' ')) {
      this.text += ' ';
      this.updateCurrentWord();
      this.render();
    }
  }

  backspace() {
    if (this.text) {
      this.text = this.text.slice(0, -1);
      this.updateCurrentWord();
      this.render();
    }
  }

  clear() {
    this.text = '';
    this.currentWord = '';
    this.render();
  }

  updateCurrentWord() {
    if (!this.text || this.text.endsWith(' ')) {
      this.currentWord = '';
    } else {
      const parts = this.text.split(' ');
      this.currentWord = parts[parts.length - 1];
    }
  }

  getState() {
    const words = this.text.split(' ').filter(Boolean);
    return {
      text: this.text,
      current_word: this.currentWord,
      character_count: this.text.length,
      word_count: words.length
    };
  }

  render() {
    if (!this.display) return;
    this.display.textContent = this.text;

    // Scroll to bottom as text grows
    this.display.scrollTop = this.display.scrollHeight;
  }

  async copyToClipboard(buttonElement = null) {
    if (!this.text) return;

    try {
      await navigator.clipboard.writeText(this.text);
      if (buttonElement) {
        const originalText = buttonElement.innerHTML;
        buttonElement.innerHTML = '<span>✓ Copied!</span>';
        setTimeout(() => {
          buttonElement.innerHTML = originalText;
        }, 2000);
      }
    } catch (err) {
      console.warn('Clipboard API error, fallback to execCommand:', err);
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = this.text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      if (buttonElement) {
        buttonElement.textContent = '✓ Copied!';
        setTimeout(() => { buttonElement.textContent = 'Copy'; }, 2000);
      }
    }
  }

  speak(buttonElement = null) {
    if (!this.speechSynth || !this.text) return;

    // Cancel any previous utterances
    this.speechSynth.cancel();

    const utterance = new SpeechSynthesisUtterance(this.text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    if (buttonElement) {
      buttonElement.classList.add('speaking');
      utterance.onend = () => buttonElement.classList.remove('speaking');
      utterance.onerror = () => buttonElement.classList.remove('speaking');
    }

    this.speechSynth.speak(utterance);
  }

  initKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
      // Ignore if user is inside an input or textarea
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        if (this.onActionRequest) this.onActionRequest('space');
      } else if (e.code === 'Backspace') {
        e.preventDefault();
        if (this.onActionRequest) this.onActionRequest('backspace');
      }
    });
  }
}
