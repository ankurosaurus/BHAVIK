# BHAVIK Mobile & Cross-Device Testing Guide

## 1. The Mobile HTTPS Requirement Explained

### Why Mobile Browsers Block Camera on Local IP Addresses
When testing on a desktop computer, accessing `http://localhost:8000` is classified by modern browsers as a **Trustworthy Origin (Secure Context)**.

However, when you attempt to connect a smartphone (Android Chrome or iOS Safari) to your computer's local Wi-Fi IP address (e.g. `http://192.168.1.15:8000`), the browser **strictly disables `navigator.mediaDevices.getUserMedia`**.

**W3C Specification Requirement**:
The W3C Media Capture and Streams API specification mandates that video and audio hardware can only be accessed within a **Secure Context** (`https://` or `http://localhost`). Browsers enforce this security policy to prevent man-in-the-middle attackers on public or shared Wi-Fi networks from silently intercepting camera feeds.

---

## 2. Practical Mobile Testing Workflows

### Option A: Cloudflare Tunnel (Recommended, Zero Cost, No Account Required)
Cloudflare Tunnel provides an instant, secure `https://*.trycloudflare.com` tunnel directly to your local FastAPI server.

1. Download `cloudflared` from [Cloudflare Releases](https://github.com/cloudflare/cloudflared/releases).
2. Start the BHAVIK backend server locally on port 8000:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
3. In a second terminal, launch the tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
4. Cloudflare outputs a secure HTTPS URL (e.g., `https://bhavik-demo.trycloudflare.com`).
5. Open this URL in Chrome on Android or Safari on iPhone. Both the camera and WebSocket (`wss://`) will connect seamlessly.

---

### Option B: ngrok Secure Tunnel
1. Download and authenticate `ngrok`.
2. Start the tunnel:
   ```bash
   ngrok http 8000
   ```
3. Copy the generated `https://` forwarding URL to your mobile browser.

---

### Option C: Chrome USB Port Forwarding (Android Only)
If internet access or tunnels are unavailable, you can test directly via USB cable:
1. Connect Android smartphone to computer with USB debugging enabled.
2. Open Google Chrome on your computer and navigate to: `chrome://inspect/#devices`.
3. Click **Port forwarding...**
4. Add rule: Port `8000` forwarding to `localhost:8000`.
5. On the Android phone, open Chrome and navigate to `http://localhost:8000`. Because Android treats `localhost` as a secure context, camera access will work without SSL.

---

## 3. Comprehensive Multi-Device Quality Assurance Checklist

| # | Test Item | Verification Procedure | Expected Outcome | Pass/Fail Criteria |
|---|-----------|------------------------|------------------|-------------------|
| 1 | **Permission Dialog** | Tap *Start Camera* on mobile device. | Browser renders native camera permission dialog explaining usage. | Must not access camera silently. |
| 2 | **Permission Denial** | Tap *Block* or *Deny* in camera dialog. | App displays friendly banner: "Camera permission was denied...". | No crash; button re-enables gracefully. |
| 3 | **Front Camera** | Accept permission with front-facing camera. | Live video appears; preview is mirrored naturally for signing. | User sees their hand naturally (mirror effect). |
| 4 | **Rear Camera** | Tap *Switch Camera (Front/Rear)*. | Feed switches to rear sensor; mirroring automatically disables. | Rear camera shows unmirrored peer perspective. |
| 5 | **Orientation Change** | Rotate phone between Portrait & Landscape. | Viewport maintains aspect ratio; UI adapts vertically or side-by-side. | No horizontal page scrolling or distorted stretching. |
| 6 | **Hand Landmark Detection** | Place hand within camera view. | Teal 21-joint skeleton renders directly on joints; "Hand: TRACKING" pill turns green. | Stable landmark overlay aligned with hand. |
| 7 | **Out-of-Frame Handling** | Remove hand from camera view. | Canvas clears; status displays "Searching Hand..."; letter stops updating. | No crash or memory leak. |
| 8 | **Prediction Stabilization** | Form letter 'B' and hold steady for 10 frames. | Stability gauge rises to 100%; 'B' is accepted and appends to transcript. | Single-frame jitters do not append false letters. |
| 9 | **Held Gesture Gate** | Keep holding 'B' steadily for 5 seconds. | Status displays "Holding Gesture"; 'B' is NOT repeatedly added. | Prevents runaway strings like "BBBBBBBB". |
| 10 | **Distinct Gesture Transition** | Transition hand from 'B' to 'H'. | Stabilizer detects transition, resets progress, and accepts 'H'. | Continuous transcription becomes "BH". |
| 11 | **Touch Target Dimensions** | Measure touch buttons (Space, Backspace, etc.). | Minimum touch dimensions $\ge 48\text{px} \times 48\text{px}$. | Easy one-handed tapping on small smartphones. |
| 12 | **Safe-Area Insets** | Test on iPhone with home indicator notch. | Content respects `env(safe-area-inset-bottom)` with bottom padding. | Navigation buttons not obscured by system bar. |
| 13 | **Clipboard Copy** | Tap *Copy Text*. | Text is copied to device clipboard; button pulses "✓ Copied!". | Clipboard matches transcript box content. |
| 14 | **Text-to-Speech (TTS)** | Tap *🔊 Speak*. | Device vocalizes transcribed sentence using Web Speech API. | Clear audio output without lag. |
| 15 | **Network Reconnection** | Toggle Airplane Mode ON for 5s then OFF. | Status transitions to "AI Disconnected" then automatically reconnects to "AI Connected". | Session resumes automatically without page reload. |

---

## 4. Tested Screen Resolution Matrix

- **Small Smartphone**: 360 × 640 px (Android / small screen layout)
- **Large Smartphone**: 414 × 896 px (iPhone XR/11/12/13/14 Pro Max)
- **Tablet (Portrait)**: 768 × 1024 px (iPad / Android Tablet)
- **Tablet (Landscape)**: 1024 × 768 px
- **Laptop / Desktop**: 1366 × 768 px and 1920 × 1080 px (Dual-column desktop workspace)
