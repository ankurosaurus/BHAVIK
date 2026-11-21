/**
 * BHAVIK In-Browser Client-Side Machine Learning Engine
 * Provides 100% serverless, zero-latency in-browser gesture inference
 * for GitHub Pages and offline Progressive Web App deployments.
 */

class InBrowserEngine {
  constructor(options = {}) {
    this.modelData = window.ASL_MODEL_DATA || null;
    this.confidenceThreshold = options.confidenceThreshold || 0.55;
    this.bufferSize = options.bufferSize || 7;
    this.stabilityRatio = options.stabilityRatio || 0.70;
    this.cooldownSeconds = options.cooldownSeconds || 0.9;

    this.buffer = [];
    this.lastAcceptedLetter = null;
    this.lastAcceptedTime = 0;
    this.heldLetter = null;
  }

  isReady() {
    return Boolean(this.modelData && this.modelData.trees && this.modelData.trees.length > 0);
  }

  /**
   * Normalizes 21 3D landmarks into 63-dimensional translation- and scale-invariant features.
   */
  extractFeatures(landmarks) {
    if (!landmarks || landmarks.length !== 21) return null;

    const wrist = landmarks[0];
    const rel = new Float32Array(63);
    let maxDistSq = 0.0;

    for (let i = 0; i < 21; i++) {
      const dx = landmarks[i].x - wrist.x;
      const dy = landmarks[i].y - wrist.y;
      const dz = (landmarks[i].z || 0.0) - (wrist.z || 0.0);

      rel[i * 3] = dx;
      rel[i * 3 + 1] = dy;
      rel[i * 3 + 2] = dz;

      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq > maxDistSq) {
        maxDistSq = distSq;
      }
    }

    const scale = Math.sqrt(maxDistSq) || 1.0;
    const invScale = 1.0 / scale;

    const features = new Float32Array(63);
    for (let i = 0; i < 63; i++) {
      features[i] = rel[i] * invScale;
    }

    return features;
  }

  /**
   * Evaluates the in-browser Random Forest model on 63 features.
   */
  predict(features) {
    if (!this.isReady() || !features) return null;

    const trees = this.modelData.trees;
    const classes = this.modelData.classes;
    const numClasses = classes.length;
    const votes = new Int32Array(numClasses);

    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t];
      const cl = tree.cl;
      const cr = tree.cr;
      const f = tree.f;
      const th = tree.th;
      const v = tree.v;

      let node = 0;
      while (cl[node] !== -1) {
        const featIdx = f[node];
        if (features[featIdx] <= th[node]) {
          node = cl[node];
        } else {
          node = cr[node];
        }
      }

      const classIdx = v[node];
      votes[classIdx]++;
    }

    let topVotes = -1;
    let topClassIdx = 0;
    for (let c = 0; c < numClasses; c++) {
      if (votes[c] > topVotes) {
        topVotes = votes[c];
        topClassIdx = c;
      }
    }

    const confidence = topVotes / trees.length;
    return {
      letter: classes[topClassIdx],
      confidence: confidence
    };
  }

  /**
   * Processes landmark frame through prediction and temporal smoothing.
   */
  processLandmarks(landmarks, now = Date.now() / 1000) {
    if (!landmarks || landmarks.length !== 21) {
      this.buffer = [];
      this.heldLetter = null;
      return {
        hand_detected: false,
        landmarks: null,
        predicted_letter: null,
        confidence: 0,
        is_stable: false,
        stabilization_progress: 0,
        accepted_letter: null,
        status: "NO_HAND"
      };
    }

    const features = this.extractFeatures(landmarks);
    const pred = this.predict(features);

    if (!pred || pred.confidence < this.confidenceThreshold) {
      this.buffer.push(null);
      if (this.buffer.length > this.bufferSize) this.buffer.shift();

      return {
        hand_detected: true,
        landmarks: landmarks,
        predicted_letter: pred ? pred.letter : null,
        confidence: pred ? pred.confidence : 0,
        is_stable: false,
        stabilization_progress: 0,
        accepted_letter: null,
        status: "LOW_CONFIDENCE"
      };
    }

    this.buffer.push(pred.letter);
    if (this.buffer.length > this.bufferSize) this.buffer.shift();

    const count = this.buffer.filter((item) => item === pred.letter).length;
    const requiredCount = Math.ceil(this.bufferSize * this.stabilityRatio);
    const progress = Math.min(1.0, count / requiredCount);
    const isStable = count >= requiredCount;

    let acceptedLetter = null;
    let status = "STABILIZING";

    if (isStable) {
      const timeSinceLast = now - this.lastAcceptedTime;
      const inCooldown = (pred.letter === this.lastAcceptedLetter) && (timeSinceLast < this.cooldownSeconds);
      const isHeld = (pred.letter === this.heldLetter);

      if (inCooldown) {
        status = "COOLDOWN";
      } else if (isHeld) {
        status = "HELD";
      } else {
        acceptedLetter = pred.letter;
        this.lastAcceptedLetter = pred.letter;
        this.lastAcceptedTime = now;
        this.heldLetter = pred.letter;
        status = "ACCEPTED";
      }
    }

    return {
      hand_detected: true,
      landmarks: landmarks,
      predicted_letter: pred.letter,
      confidence: pred.confidence,
      is_stable: isStable,
      stabilization_progress: progress,
      accepted_letter: acceptedLetter,
      status: status
    };
  }

  configure(settings = {}) {
    if (settings.confidence_threshold !== undefined) {
      this.confidenceThreshold = settings.confidence_threshold;
    }
    if (settings.buffer_size !== undefined) {
      this.bufferSize = settings.buffer_size;
    }
    if (settings.cooldown_seconds !== undefined) {
      this.cooldownSeconds = settings.cooldown_seconds;
    }
  }
}
