/**
 * BHAVIK WebSocket Client
 * Manages full-duplex communication with the recognition server,
 * automatic reconnection, and dynamic protocol negotiation (ws:// vs wss://).
 */

class RecognitionSocket {
  constructor(endpoint = '/ws/recognize') {
    this.endpoint = endpoint;
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.reconnectDelay = 1500;
    this.reconnectTimer = null;
    this.isDeliberateClose = false;
    this.isSending = false;

    this.onMessageCallback = null;
    this.onStatusChangeCallback = null;
  }

  getWebSocketUrl() {
    if (window.location.protocol === 'file:') {
      return `ws://localhost:8000${this.endpoint}`;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host = window.location.host;
    if (!host || host === '') {
      host = 'localhost:8000';
    } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      if (window.location.port !== '8000' && window.location.port !== '8443') {
        const port = window.location.protocol === 'https:' ? '8443' : '8000';
        host = `${window.location.hostname}:${port}`;
      }
    }
    return `${protocol}//${host}${this.endpoint}`;
  }

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isDeliberateClose = false;
    const url = this.getWebSocketUrl();
    this.updateStatus('CONNECTING');

    try {
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.updateStatus('ONLINE');
      };

      this.socket.onmessage = (event) => {
        if (this.sendTimeout) clearTimeout(this.sendTimeout);
        this.isSending = false;
        try {
          const payload = JSON.parse(event.data);
          if (this.onMessageCallback) {
            this.onMessageCallback(payload);
          }
        } catch (err) {
          console.error('Error parsing WebSocket response:', err);
        }
      };

      this.socket.onclose = () => {
        this.isConnected = false;
        this.isSending = false;
        this.updateStatus('DISCONNECTED');
        if (!this.isDeliberateClose) {
          this.scheduleReconnect();
        }
      };

      this.socket.onerror = (error) => {
        console.warn('WebSocket connection error:', error);
        this.socket.close();
      };
    } catch (err) {
      console.error('Failed to initialize WebSocket:', err);
      this.scheduleReconnect();
    }
  }

  disconnect() {
    this.isDeliberateClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.isConnected = false;
    this.updateStatus('DISCONNECTED');
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.updateStatus('OFFLINE');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(10000, this.reconnectDelay * Math.pow(1.3, this.reconnectAttempts));
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  sendFrame(base64Image) {
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    // Drop frame if previous frame is still in flight
    if (this.isSending) {
      return false;
    }

    this.isSending = true;
    if (this.sendTimeout) clearTimeout(this.sendTimeout);
    this.sendTimeout = setTimeout(() => {
      this.isSending = false;
    }, 250); // automatic safety fallback

    try {
      this.socket.send(JSON.stringify({
        image: base64Image,
        client_timestamp: Date.now()
      }));
      return true;
    } catch (err) {
      this.isSending = false;
      return false;
    }
  }

  sendAction(actionName, settings = null) {
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    const payload = { action: actionName };
    if (settings) {
      payload.settings = settings;
    }
    this.socket.send(JSON.stringify(payload));
    return true;
  }

  updateStatus(status) {
    if (this.onStatusChangeCallback) {
      this.onStatusChangeCallback(status);
    }
  }
}
