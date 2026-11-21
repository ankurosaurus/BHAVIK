/**
 * BHAVIK Main Application Script
 * Orchestrates camera streaming, WebSocket communication, UI data-binding,
 * settings management, and modal interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const videoElem = document.getElementById('camera-video');
  const canvasElem = document.getElementById('landmark-canvas');
  const emptyState = document.getElementById('camera-empty-state');
  const startCameraBtn = document.getElementById('btn-start-camera');
  const stopCameraBtn = document.getElementById('btn-stop-camera');
  const switchCameraBtn = document.getElementById('btn-switch-camera');
  const mirrorToggleBtn = document.getElementById('btn-toggle-mirror');

  // HUD Elements
  const hudHandStatus = document.getElementById('hud-hand-status');
  const hudFps = document.getElementById('hud-fps');
  const activeLetterElem = document.getElementById('active-predicted-letter');
  const letterBox = document.getElementById('letter-display-box');
  const confidenceFill = document.getElementById('confidence-progress-fill');
  const confidenceText = document.getElementById('confidence-val-text');
  const stabilityFill = document.getElementById('stability-progress-fill');
  const stabilityText = document.getElementById('stability-val-text');
  const lastAcceptedElem = document.getElementById('last-accepted-val');
  const recognitionStatusElem = document.getElementById('recognition-status-val');
  const serverStatusDot = document.getElementById('server-status-dot');
  const serverStatusText = document.getElementById('server-status-text');

  // Transcription Elements
  const transcriptDisplay = document.getElementById('live-transcript');
  const btnSpace = document.getElementById('btn-space');
  const btnBackspace = document.getElementById('btn-backspace');
  const btnClear = document.getElementById('btn-clear');
  const btnCopy = document.getElementById('btn-copy');
  const btnSpeak = document.getElementById('btn-speak');
  const wordCountElem = document.getElementById('word-count-badge');

  // Modals & Banners
  const permissionBanner = document.getElementById('permission-banner');
  const btnBannerDismiss = document.getElementById('btn-banner-dismiss');
  const btnSettings = document.getElementById('btn-open-settings');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const settingsModal = document.getElementById('settings-modal');
  const btnAslGuide = document.getElementById('btn-open-guide');
  const btnCloseGuide = document.getElementById('btn-close-guide');
  const guideModal = document.getElementById('guide-modal');

  // Settings Controls
  const sliderConfidence = document.getElementById('slider-confidence');
  const sliderConfidenceVal = document.getElementById('val-confidence');
  const sliderBuffer = document.getElementById('slider-buffer');
  const sliderBufferVal = document.getElementById('val-buffer');
  const sliderCooldown = document.getElementById('slider-cooldown');
  const sliderCooldownVal = document.getElementById('val-cooldown');
  const btnApplySettings = document.getElementById('btn-apply-settings');

  // Initialize Subsystems
  const camera = new CameraController(videoElem, canvasElem, {
    targetFps: 18,
    width: 640,
    height: 480
  });

  const socket = new RecognitionSocket('/ws/recognize');

  const inBrowserEngine = new InBrowserEngine({
    confidenceThreshold: 0.55,
    bufferSize: 7,
    stabilityRatio: 0.70,
    cooldownSeconds: 0.9,
  });

  const transcription = new TranscriptionController(transcriptDisplay, {
    onActionRequest: (actionName) => {
      if (actionName === 'space') transcription.addSpace();
      if (actionName === 'backspace') transcription.backspace();
      updateTranscriptionMeta(transcription.getState());
      if (socket.isConnected) socket.sendAction(actionName);
    }
  });

  // Link Camera to WebSocket
  camera.onFrameCallback = (jpegDataUrl) => {
    socket.sendFrame(jpegDataUrl);
  };

  camera.onErrorCallback = (errInfo) => {
    showCameraError(errInfo);
  };

  // Link WebSocket to UI & Transcription
  socket.onMessageCallback = (data) => {
    handleRecognitionData(data);
  };

  socket.onStatusChangeCallback = (status) => {
    updateServerStatus(status);
  };

  // Connect WebSocket initially
  socket.connect();

  // ==========================================================================
  // Recognition Result Handler
  // ==========================================================================
  function handleRecognitionData(data) {
    if (data.action_ack) {
      if (data.transcription) {
        transcription.updateState(data.transcription);
        updateTranscriptionMeta(data.transcription);
      }
      return;
    }

    // 1. Hand Skeleton Overlay
    if (data.hand_detected && data.landmarks) {
      camera.drawLandmarks(data.landmarks);
      hudHandStatus.textContent = 'Hand: TRACKING';
      hudHandStatus.className = 'hud-pill detected';
    } else {
      camera.clearCanvas();
      hudHandStatus.textContent = 'Hand: NONE';
      hudHandStatus.className = 'hud-pill no-hand';
    }

    // 2. Server FPS
    if (data.server_fps) {
      hudFps.textContent = `${data.server_fps} FPS`;
    }

    // 3. Predicted Letter
    const currentLetter = data.predicted_letter || '—';
    activeLetterElem.textContent = currentLetter;

    // 4. Confidence Gauge
    const confPct = Math.round((data.confidence || 0) * 100);
    confidenceFill.style.width = `${confPct}%`;
    confidenceText.textContent = `${confPct}%`;

    // 5. Stability Gauge
    const stabPct = Math.round((data.stabilization_progress || 0) * 100);
    stabilityFill.style.width = `${stabPct}%`;
    stabilityText.textContent = `${stabPct}%`;

    // 6. Last Accepted Letter & Status
    if (data.accepted_letter) {
      lastAcceptedElem.textContent = data.accepted_letter;
      // Trigger subtle pulse glow on acceptance
      letterBox.classList.add('accepted');
      setTimeout(() => letterBox.classList.remove('accepted'), 400);
    }

    recognitionStatusElem.textContent = formatStatus(data.status);

    // 7. Update Transcription State
    if (data.transcription) {
      transcription.updateState(data.transcription);
      updateTranscriptionMeta(data.transcription);
    }
  }

  function formatStatus(status) {
    switch (status) {
      case 'ACCEPTED': return 'Accepted ✓';
      case 'STABILIZING': return 'Stabilizing...';
      case 'COOLDOWN': return 'Cooldown';
      case 'HELD': return 'Holding Gesture';
      case 'LOW_CONFIDENCE': return 'Low Confidence';
      case 'NO_HAND': return 'Searching Hand...';
      default: return 'Ready';
    }
  }

  function updateTranscriptionMeta(state) {
    if (wordCountElem && state.word_count !== undefined) {
      wordCountElem.textContent = `${state.word_count} word${state.word_count === 1 ? '' : 's'} (${state.character_count || 0} chars)`;
    }
  }

  function updateServerStatus(status) {
    if (status === 'ONLINE') {
      serverStatusDot.className = 'status-dot online';
      serverStatusText.textContent = 'AI Engine Connected';
    } else if (status === 'CONNECTING') {
      serverStatusDot.className = 'status-dot';
      serverStatusText.textContent = 'Connecting to AI...';
    } else {
      if (typeof inBrowserEngine !== 'undefined' && inBrowserEngine.isReady()) {
        serverStatusDot.className = 'status-dot online';
        serverStatusText.textContent = 'AI Engine: Client Ready';
      } else {
        serverStatusDot.className = 'status-dot';
        serverStatusText.textContent = 'AI Disconnected';
      }
    }
  }

  // ==========================================================================
  // Camera User Actions
  // ==========================================================================
  startCameraBtn.addEventListener('click', async () => {
    try {
      startCameraBtn.disabled = true;
      startCameraBtn.textContent = 'Starting...';
      await camera.startCamera();

      // Initialize in-browser MediaPipe inference for zero-latency client recognition
      if (typeof inBrowserEngine !== 'undefined' && inBrowserEngine.isReady()) {
        camera.initInBrowserMediaPipe((results) => {
          let landmarks = null;
          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            landmarks = results.multiHandLandmarks[0].map((pt) => ({
              x: pt.x,
              y: pt.y,
              z: pt.z,
            }));
          }

          const res = inBrowserEngine.processLandmarks(landmarks);
          if (res.accepted_letter) {
            transcription.addLetter(res.accepted_letter);
            updateTranscriptionMeta(transcription.getState());
          }
          handleRecognitionData(res);
        });

        serverStatusDot.className = 'status-dot online';
        serverStatusText.textContent = 'AI Engine: Active';
      }

      emptyState.style.display = 'none';
      startCameraBtn.style.display = 'none';
      stopCameraBtn.style.display = 'inline-flex';
      switchCameraBtn.disabled = false;
      mirrorToggleBtn.disabled = false;
    } catch (err) {
      console.error('Camera startup error:', err);
    } finally {
      startCameraBtn.disabled = false;
      startCameraBtn.textContent = 'Start Camera';
    }
  });

  stopCameraBtn.addEventListener('click', () => {
    camera.stopCamera();
    emptyState.style.display = 'flex';
    stopCameraBtn.style.display = 'none';
    startCameraBtn.style.display = 'inline-flex';
    switchCameraBtn.disabled = true;
    mirrorToggleBtn.disabled = true;
    activeLetterElem.textContent = '—';
    confidenceFill.style.width = '0%';
    confidenceText.textContent = '0%';
    stabilityFill.style.width = '0%';
    stabilityText.textContent = '0%';
    recognitionStatusElem.textContent = 'Camera Stopped';
    hudHandStatus.textContent = 'Camera OFF';
    hudHandStatus.className = 'hud-pill';
  });

  switchCameraBtn.addEventListener('click', async () => {
    try {
      await camera.switchCamera();
    } catch (err) {
      console.warn('Unable to switch camera:', err);
    }
  });

  mirrorToggleBtn.addEventListener('click', () => {
    videoElem.classList.toggle('mirrored');
    canvasElem.classList.toggle('mirrored');
  });

  function showCameraError(errInfo) {
    alert(`Camera Notice: ${errInfo.message}`);
  }

  // ==========================================================================
  // Transcription Controls
  // ==========================================================================
  btnSpace.addEventListener('click', () => {
    transcription.addSpace();
    updateTranscriptionMeta(transcription.getState());
    if (socket.isConnected) socket.sendAction('space');
  });

  btnBackspace.addEventListener('click', () => {
    transcription.backspace();
    updateTranscriptionMeta(transcription.getState());
    if (socket.isConnected) socket.sendAction('backspace');
  });

  btnClear.addEventListener('click', () => {
    if (confirm('Clear the current transcription?')) {
      transcription.clear();
      updateTranscriptionMeta(transcription.getState());
      if (socket.isConnected) socket.sendAction('clear');
    }
  });

  btnCopy.addEventListener('click', () => {
    transcription.copyToClipboard(btnCopy);
  });

  btnSpeak.addEventListener('click', () => {
    transcription.speak(btnSpeak);
  });

  if (btnBannerDismiss) {
    btnBannerDismiss.addEventListener('click', () => {
      permissionBanner.classList.add('hidden');
    });
  }

  // ==========================================================================
  // Settings Modal Logic
  // ==========================================================================
  btnSettings.addEventListener('click', () => settingsModal.classList.add('active'));
  btnCloseSettings.addEventListener('click', () => settingsModal.classList.remove('active'));

  sliderConfidence.addEventListener('input', (e) => {
    sliderConfidenceVal.textContent = `${Math.round(e.target.value * 100)}%`;
  });
  sliderBuffer.addEventListener('input', (e) => {
    sliderBufferVal.textContent = `${e.target.value} frames`;
  });
  sliderCooldown.addEventListener('input', (e) => {
    sliderCooldownVal.textContent = `${e.target.value}s`;
  });

  btnApplySettings.addEventListener('click', () => {
    const settings = {
      confidence_threshold: parseFloat(sliderConfidence.value),
      buffer_size: parseInt(sliderBuffer.value, 10),
      cooldown_seconds: parseFloat(sliderCooldown.value),
    };
    socket.sendAction('configure', settings);
    settingsModal.classList.remove('active');
  });

  // ==========================================================================
  // ASL Guide Modal Logic
  // ==========================================================================
  btnAslGuide.addEventListener('click', () => guideModal.classList.add('active'));
  btnCloseGuide.addEventListener('click', () => guideModal.classList.remove('active'));

  // Close modals on clicking overlay backdrop
  window.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.classList.remove('active');
    if (e.target === guideModal) guideModal.classList.remove('active');
  });

  // Register PWA Service Worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./service-worker.js').catch((err) => {
      console.warn('Service Worker registration skipped:', err);
    });
  }
});
