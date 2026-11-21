/**
 * BHAVIK Camera Controller
 * Manages getUserMedia stream, camera enumeration, facingMode toggling,
 * canvas-based frame capture, and frame rate throttling for mobile and desktop.
 */

class CameraController {
  constructor(videoElement, canvasElement, options = {}) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');

    this.targetFps = options.targetFps || 18;
    this.width = options.width || 640;
    this.height = options.height || 480;

    this.stream = null;
    this.facingMode = 'user'; // 'user' (front) or 'environment' (rear)
    this.availableDevices = [];
    this.activeDeviceId = null;
    this.isStreaming = false;

    this.captureTimer = null;
    this.onFrameCallback = null;
    this.onErrorCallback = null;

    // Mode: 'server' (stream to FastAPI WebSocket backend) vs 'client' (in-browser MediaPipe)
    this.mode = 'server';
    this.useInBrowserMediaPipe = false;
    this.isProcessingMediaPipe = false;
    this.mpHands = null;

    // Offscreen canvas for JPEG encoding
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCanvas.width = this.width;
    this.offscreenCanvas.height = this.height;
    this.offscreenCtx = this.offscreenCanvas.getContext('2d');
  }

  async checkSupport() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async enumerateCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.availableDevices = devices.filter((d) => d.kind === 'videoinput');
      return this.availableDevices;
    } catch (err) {
      console.warn('Unable to enumerate camera devices:', err);
      return [];
    }
  }

  async startCamera(facingMode = this.facingMode, deviceId = null) {
    if (this.isStreaming) {
      this.stopCamera();
    }

    const constraints = {
      audio: false,
      video: {
        width: { ideal: this.width },
        height: { ideal: this.height },
        frameRate: { ideal: this.targetFps, max: 30 }
      }
    };

    if (deviceId) {
      constraints.video.deviceId = { exact: deviceId };
    } else {
      constraints.video.facingMode = facingMode;
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();

      this.facingMode = facingMode;
      this.activeDeviceId = deviceId;
      this.isStreaming = true;

      // Adjust preview mirroring (mirror user front camera, unmirror rear camera)
      if (this.facingMode === 'user') {
        this.video.classList.add('mirrored');
        this.canvas.classList.add('mirrored');
      } else {
        this.video.classList.remove('mirrored');
        this.canvas.classList.remove('mirrored');
      }

      // Sync canvas dimensions to actual video stream resolution
      const updateDimensions = () => {
        const vw = this.video.videoWidth || this.width;
        const vh = this.video.videoHeight || this.height;
        if (vw > 0 && vh > 0) {
          this.canvas.width = vw;
          this.canvas.height = vh;
          this.offscreenCanvas.width = vw;
          this.offscreenCanvas.height = vh;
        }
      };
      updateDimensions();
      this.video.addEventListener('loadeddata', updateDimensions, { once: true });
      this.video.addEventListener('resize', updateDimensions);

      this.startFrameCaptureLoop();
      await this.enumerateCameras();
      return true;
    } catch (error) {
      this.isStreaming = false;
      if (this.onErrorCallback) {
        this.onErrorCallback(this.formatCameraError(error));
      }
      throw error;
    }
  }

  stopCamera() {
    this.stopFrameCaptureLoop();
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;
    this.isStreaming = false;
    this.clearCanvas();
  }

  async switchCamera() {
    const nextFacingMode = this.facingMode === 'user' ? 'environment' : 'user';
    return this.startCamera(nextFacingMode);
  }

  startFrameCaptureLoop() {
    const intervalMs = 1000 / this.targetFps;
    let lastTime = 0;

    const loop = (timestamp) => {
      if (!this.isStreaming) return;

      if (timestamp - lastTime >= intervalMs) {
        lastTime = timestamp;
        this.captureAndEmitFrame();
      }

      this.captureTimer = requestAnimationFrame(loop);
    };

    this.captureTimer = requestAnimationFrame(loop);
  }

  stopFrameCaptureLoop() {
    if (this.captureTimer) {
      cancelAnimationFrame(this.captureTimer);
      this.captureTimer = null;
    }
  }

  setMode(mode) {
    this.mode = mode; // 'server' or 'client'
  }

  async initInBrowserMediaPipe(onResultsCallback) {
    if (this.mpHands) {
      this.useInBrowserMediaPipe = true;
      return true;
    }

    if (typeof window.Hands !== 'function') {
      console.warn('MediaPipe Hands is not loaded in window.Hands');
      this.useInBrowserMediaPipe = false;
      return false;
    }

    try {
      this.mpHands = new window.Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      this.mpHands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0, // Lite model: smooth and responsive on web/mobile CPU
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      this.mpHands.onResults((results) => {
        if (onResultsCallback) {
          onResultsCallback(results);
        }
      });

      // Warm up MediaPipe WASM graph
      await this.mpHands.initialize();
      this.useInBrowserMediaPipe = true;
      return true;
    } catch (err) {
      console.warn('Could not initialize in-browser MediaPipe:', err);
      this.useInBrowserMediaPipe = false;
      return false;
    }
  }

  captureAndEmitFrame() {
    if (!this.video || this.video.readyState < 2) return;

    // 1. Server Mode: send JPEG frame to FastAPI WebSocket backend
    if (this.mode === 'server') {
      if (!this.onFrameCallback) return;

      const w = this.offscreenCanvas.width;
      const h = this.offscreenCanvas.height;
      if (w <= 0 || h <= 0) return;

      this.offscreenCtx.drawImage(this.video, 0, 0, w, h);
      const jpegDataUrl = this.offscreenCanvas.toDataURL('image/jpeg', 0.65);
      this.onFrameCallback(jpegDataUrl);
      return;
    }

    // 2. Client Mode: evaluate with in-browser MediaPipe with strict concurrency locking
    if (this.mode === 'client' && this.useInBrowserMediaPipe && this.mpHands) {
      if (this.isProcessingMediaPipe) {
        return; // Skip frame if previous WASM frame is still in flight (prevents pipeline lockup!)
      }

      this.isProcessingMediaPipe = true;
      this.mpHands.send({ image: this.video })
        .catch((err) => {
          console.warn('In-browser MediaPipe inference error:', err);
        })
        .finally(() => {
          this.isProcessingMediaPipe = false;
        });
      return;
    }

    // 3. Fallback: stream frame if callback exists
    if (this.onFrameCallback) {
      const w = this.offscreenCanvas.width;
      const h = this.offscreenCanvas.height;
      if (w > 0 && h > 0) {
        this.offscreenCtx.drawImage(this.video, 0, 0, w, h);
        const jpegDataUrl = this.offscreenCanvas.toDataURL('image/jpeg', 0.65);
        this.onFrameCallback(jpegDataUrl);
      }
    }
  }

  drawLandmarks(landmarks) {
    this.clearCanvas();
    if (!landmarks || landmarks.length !== 21) return;

    const w = this.canvas.width;
    const h = this.canvas.height;

    // Hand skeleton connection pairs (MediaPipe Hands topology)
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [9, 13], [13, 14], [14, 15], [15, 16],// Ring
      [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [0, 17]                               // Palm base
    ];

    // Draw connection lines
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = '#00d2b4';
    this.ctx.lineCap = 'round';

    for (const [startIdx, endIdx] of connections) {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];
      this.ctx.beginPath();
      this.ctx.moveTo(p1.x * w, p1.y * h);
      this.ctx.lineTo(p2.x * w, p2.y * h);
      this.ctx.stroke();
    }

    // Draw landmark joint points
    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      const isTip = [4, 8, 12, 16, 20].includes(i);

      this.ctx.beginPath();
      this.ctx.arc(p.x * w, p.y * h, isTip ? 6 : 4, 0, 2 * Math.PI);
      this.ctx.fillStyle = isTip ? '#3b82f6' : '#ffffff';
      this.ctx.fill();
      this.ctx.lineWidth = 1.5;
      this.ctx.strokeStyle = '#0a0e17';
      this.ctx.stroke();
    }
  }

  clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  formatCameraError(err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      return {
        type: 'PERMISSION_DENIED',
        message: 'Camera permission was denied. Please grant permission in your browser address bar settings to recognize sign gestures.'
      };
    }
    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      return {
        type: 'NO_CAMERA',
        message: 'No supported video camera was detected on this device.'
      };
    }
    if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      return {
        type: 'CAMERA_BUSY',
        message: 'Camera is currently in use by another application or browser tab.'
      };
    }
    return {
      type: 'UNKNOWN',
      message: err.message || 'An unexpected error occurred while accessing the camera.'
    };
  }
}
