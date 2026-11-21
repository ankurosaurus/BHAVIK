# BHAVIK Presentation Guide & Viva Voce Handbook

---

## 1. Ten-Slide Academic PowerPoint Outline

### Slide 1: Title & Introduction
- **Title**: BHAVIK – Bridging Hands and Visual Intelligence through Knowledge
- **Subtitle**: Real-Time ASL Fingerspelling Recognition and Continuous Transcription System
- **Presenter**: Second-Year B.Tech Computer Science & Engineering Student
- **Tagline**: *"Your Hands. Your Voice. Your Words."*

### Slide 2: Problem Statement & Motivation
- **Context**: Over 70 million Deaf and hard-of-hearing individuals worldwide face daily communication barriers.
- **The Gap**: Qualified human interpreters are scarce and expensive; existing research prototypes require costly sensor gloves or high-end desktop GPUs.
- **The Challenge**: Creating a system that works on both ordinary smartphone browsers and laptop webcams without paid cloud APIs.

### Slide 3: Scope & Critical Distinctions
- **Scope**: Focuses on real-time American Sign Language (ASL) alphabet fingerspelling.
- **ASL vs. ISL**: ASL is a one-handed fingerspelling system; Indian Sign Language (ISL) primarily uses two hands and distinct grammar.
- **Dynamic Letter Caveat**: Letters **J** and **Z** involve continuous spatial motion; to maintain scientific rigor, they are explicitly excluded from this static spatial classifier.

### Slide 4: System Architecture
- **Client Tier**: Responsive HTML5, modern CSS3, ES6 JavaScript, MediaDevices API, Canvas API.
- **Transport**: Full-duplex WebSockets (`/ws/recognize`) streaming 640×480 JPEG frames at 15–20 FPS.
- **Server Tier**: FastAPI on Uvicorn, OpenCV frame decoding, Google MediaPipe 21-joint extraction, 63-D feature engineering, Random Forest Classifier.

### Slide 5: Computer Vision with Google MediaPipe
- Why MediaPipe instead of raw CNNs? MediaPipe extracts **21 3D landmarks** ($X, Y, Z$) on CPU in ~15 ms.
- Eliminates environmental noise: background clutter, changing room lighting, and skin tone bias are discarded, leaving only pure hand joint geometry.

### Slide 6: Invariant Feature Engineering (63 Features)
- **Problem**: Hands move around the screen and change distance from the camera lens.
- **Solution 1 (Translation Invariance)**: Subtract wrist coordinate ($P_0$) from all joints: $P_i' = P_i - P_0$.
- **Solution 2 (Scale Invariance)**: Divide by maximum Euclidean distance: $\hat{P}_i = P_i' / S$.
- **Result**: Exactly $21 \times 3 = 63$ invariant numerical features.

### Slide 7: Machine Learning Model (Random Forest)
- **Configuration**: 150 randomized decision trees, max depth of 20, balanced class weights.
- **Why Random Forest over SVM/Neural Nets?**: Non-linear decision boundaries, fast multi-class posterior probabilities, zero GPU requirement, sub-millisecond CPU inference (0.85 ms).

### Slide 8: Temporal Prediction Stabilization (Smoother)
- **Why it matters**: A single unstable video frame must never inject a letter into transcription.
- **Mechanisms**:
  1. *Rolling FIFO Queue*: 10-frame sliding window with 80% majority voting agreement.
  2. *Cooldown Period*: 1.2s delay preventing duplicate letters from held gestures.
  3. *Hand Removal Reset*: Buffer flushes when hand exits the frame.

### Slide 9: Multi-Device Mobile Implementation & HTTPS
- **Responsive Mobile First**: Large touch targets ($\ge 48\text{px}$), safe-area insets, front/rear camera toggling, CSS preview mirroring.
- **HTTPS Mandate**: Why mobile camera fails on raw local IP (`192.168.x.x`) due to W3C Secure Contexts; solution using Cloudflare Tunnel or ngrok.

### Slide 10: Results, Demonstration & Future Scope
- **Metrics**: **96.17% Test Accuracy**, **96.19% Macro Precision**, **96.17% Macro F1-Score**.
- **Live Demo**: Spelled *"BHAVIK"*, inserted spaces, copied to clipboard, and vocalized via Text-to-Speech (TTS).
- **Future Scope**: 1D-CNN/GRU for dynamic J/Z gestures and two-handed Indian Sign Language (ISL).

---

## 2. Five-Minute Presentation Script (Elevator Pitch)

> *"Good morning, respected professors and evaluators. Today, I am proud to present **BHAVIK: Bridging Hands and Visual Intelligence through Knowledge**.*
>
> *Over 500,000 individuals in North America communicate using American Sign Language. However, for a Deaf person visiting a bank, a clinic, or a college counter, the person on the other side almost never understands sign language. Traditional solutions either use \$500 sensory gloves or heavy deep learning models requiring expensive cloud GPUs.*
>
> *Our goal with BHAVIK was clear: Build a real-time, zero-cost sign language fingerspelling transcriber that works on any standard smartphone browser as smoothly as it does on a laptop webcam.*
>
> *Here is how BHAVIK works under the hood:*
> *First, the user opens the web app and grants camera permission. The browser streams video frames over a low-latency WebSocket connection to our FastAPI backend.*
> *Second, Google MediaPipe tracks the 21 3D joint landmarks of the hand. This isolates the skeleton and eliminates background clutter and lighting issues.*
> *Third, our feature extractor converts these landmarks into a 63-dimensional vector that is mathematically invariant to hand position and camera distance.*
> *Fourth, a 150-tree Random Forest classifier predicts the letter in less than 1 millisecond. To prevent accidental typos, our temporal smoother ensures the gesture is held stable for at least 8 to 10 frames before committing it to text.*
>
> *The user can form words, insert spaces, backspace, copy the final transcription, or click Speak to hear it vocalized.*
> *On our independent validation set, BHAVIK achieves a genuine **96.17% accuracy** across 24 ASL letters. Thank you, and I look forward to your questions."*

---

## 3. Ten-Minute Comprehensive Presentation Script

*(Use the 5-minute structure above for the first half, then expand on the engineering details below):*

> *"To make this system practical for real-world academic and industrial training, we solved three major engineering bottlenecks:*
>
> *1. **The Mobile HTTPS Bottleneck**: Many student projects fail on phones because developers assume typing their laptop’s IP address into a phone browser will work. Under modern browser security specifications, camera APIs are blocked on unencrypted HTTP. We solved this by architecting a dual deployment path: local development on localhost, and a secure Cloudflare HTTPS development tunnel that enables front and rear camera switching on Android and iOS.*
>
> *2. **Cross-Hand and Scale Invariance**: A child’s hand is smaller than an adult’s hand, and standing two feet away shrinks the image. If you feed raw pixel coordinates into a classifier, it fails. We solved this with wrist-relative translation ($P_i - P_0$) and palm-span normalization ($P_i' / \text{max\_dist}$), guaranteeing that distance from the camera does not alter the feature vector.*
>
> *3. **Temporal Jitter vs. Repeated Letters**: If a user signs 'B', camera noise might briefly flicker to 'D' for a single frame. Without stabilization, you get gibberish. Furthermore, if you hold 'B', you don’t want the system writing 'BBBBBBBB'. Our dual-state stabilizer uses majority voting combined with a cooldown hysteresis timer to ensure smooth, natural transcription.*
>
> *Importantly, we maintain academic integrity: BHAVIK recognizes static fingerspelling letters. Letters J and Z require temporal motion strokes, which we clearly separate in our system architecture rather than faking static predictions. Thank you."*

---

## 4. Comprehensive Viva Voce Questions & Answers

### Q1: Why did you use MediaPipe instead of training a Convolutional Neural Network (CNN) directly on hand images?
**Answer**: Direct image-based CNNs learn superficial shortcuts—such as background wallpaper, room shadows, and skin pigmentation—which leads to poor real-world generalization. Furthermore, CNN inference is computationally heavy for mobile devices. MediaPipe decouples the problem: it isolates pure anatomical 3D keypoint geometry in ~15 ms on CPU, making our pipeline lightweight, privacy-preserving, and immune to background clutter.

### Q2: Why did you choose Random Forest over Deep Neural Networks or SVM?
**Answer**: The input to our model is a structured, normalized 63-dimensional numerical vector. For tabular spatial coordinates, Random Forests provide:
1. Excellent non-linear decision boundary partitioning without needing GPU acceleration.
2. Direct calibrated class probability estimation via `predict_proba`.
3. Inherent resistance to overfitting through bootstrap aggregation (bagging) and random feature subspaces.
4. Extremely low latency: 0.85 ms per prediction, enabling seamless 20+ FPS streaming.

### Q3: Why are letters 'J' and 'Z' excluded from the recognition classes?
**Answer**: In American Sign Language, 24 letters are static hand configurations, but **J** and **Z** inherently involve motion trajectories: 'J' requires tracing a hook trajectory with the pinky, and 'Z' requires tracing a zig-zag path with the index finger. Classifying them as single static spatial snapshots is methodologically incorrect. They require temporal sequence models such as 1D-CNNs, GRUs, or Hidden Markov Models, which we have outlined in our Future Scope.

### Q4: What is the mathematical basis of your feature normalization?
**Answer**: We implement two transformations:
1. *Translation Invariance*: Shifting all 21 3D points relative to the wrist (landmark 0): $\mathbf{P}_i' = \mathbf{P}_i - \mathbf{P}_0$. This ensures the coordinates are identical regardless of where the hand is located in the camera frame.
2. *Scale Invariance*: Dividing each relative coordinate by the maximum Euclidean distance from the wrist to any fingertip: $\hat{\mathbf{P}}_i = \mathbf{P}_i' / S$. This ensures the coordinates are identical whether the user is 1 foot or 5 feet from the lens.

### Q5: Why can't a smartphone open the camera just by navigating to the laptop's Wi-Fi IP address (e.g., `http://192.168.1.5:8000`)?
**Answer**: The W3C Media Capture specification enforces that `getUserMedia()` can only be accessed from a **Secure Context** (`https://` or `http://localhost`). Browsers enforce this to protect user privacy against eavesdropping on unencrypted Wi-Fi networks. To test on mobile devices, we use an HTTPS tunnel (such as Cloudflare Tunnel or ngrok) or Chrome USB Port Forwarding.

### Q6: How does your prediction stabilizer prevent duplicate letters when a user holds a gesture?
**Answer**: Our `PredictionSmoother` implements a state machine with three checks:
1. It maintains a 10-frame FIFO queue requiring an 80% majority vote.
2. Once letter $L$ is accepted, $L$ is recorded as the `held_letter`. As long as the user continues holding $L$, further insertions are suppressed.
3. If the user removes their hand or signs a different letter, the buffer and hold state are reset, allowing $L$ to be signed again after a 1.2-second cooldown window.

### Q7: What is the difference between ASL and Indian Sign Language (ISL)?
**Answer**: Sign languages are natural languages with independent vocabularies and grammars. ASL is predominantly a one-handed fingerspelling system developed in the United States and Canada. In contrast, Indian Sign Language (ISL) is native to the Indian subcontinent and heavily utilizes two-handed fingerspelling. BHAVIK specifically models the 24 static letters of ASL.

### Q8: Does BHAVIK claim to replace a certified sign language interpreter?
**Answer**: No. Fingerspelling is only one component of sign language, typically used for proper nouns, names, and technical terms. Complete sign language communication relies on facial expressions, body orientation, spatial references, and dynamic two-handed signs. BHAVIK is an assistive transcription and educational tool, not an interpreter replacement.
