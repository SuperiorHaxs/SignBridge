/**
 * WebSocket Service for real-time communication with the backend
 */

import { io, Socket } from 'socket.io-client';

// Configuration - update this to your server's IP address
const SERVER_URL = 'http://192.168.1.100:5000';  // Replace with your computer's IP

export interface GlossEvent {
  gloss: string;
  confidence: number;
  top_k_predictions?: Array<{ gloss: string; confidence: number }>;
}

export interface SentenceEvent {
  sentence: string;
}

export interface SpeechCaptionEvent {
  text: string;
  is_final: boolean;
  speaker: string;
}

export interface TTSRequestEvent {
  text: string;
  speaker: string;
}

export interface ConversationMessageEvent {
  text: string;
  speaker: 'asl_user' | 'hearing_user';
  type: 'asl_caption' | 'speech';
  timestamp: number;
}

type GlossCallback = (data: GlossEvent) => void;
type SentenceCallback = (data: SentenceEvent) => void;
type ConnectionCallback = (connected: boolean) => void;
type SpeechCaptionCallback = (data: SpeechCaptionEvent) => void;
type TTSRequestCallback = (data: TTSRequestEvent) => void;
type ConversationMessageCallback = (data: ConversationMessageEvent) => void;

class SocketService {
  private socket: Socket | null = null;
  private serverUrl: string = SERVER_URL;
  private glossCallbacks: GlossCallback[] = [];
  private sentenceCallbacks: SentenceCallback[] = [];
  private connectionCallbacks: ConnectionCallback[] = [];
  private speechCaptionCallbacks: SpeechCaptionCallback[] = [];
  private ttsRequestCallbacks: TTSRequestCallback[] = [];
  private conversationMessageCallbacks: ConversationMessageCallback[] = [];

  /**
   * Update server URL
   */
  setServerUrl(url: string) {
    this.serverUrl = url;
    // Reconnect if already connected
    if (this.socket?.connected) {
      this.disconnect();
      this.connect();
    }
  }

  /**
   * Connect to the WebSocket server
   */
  connect(): Promise<boolean> {
    return new Promise((resolve) => {
      if (this.socket?.connected) {
        resolve(true);
        return;
      }

      this.socket = io(this.serverUrl, {
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      this.socket.on('connect', () => {
        console.log('[Socket] Connected to server');
        this.notifyConnectionChange(true);
        resolve(true);
      });

      this.socket.on('disconnect', () => {
        console.log('[Socket] Disconnected from server');
        this.notifyConnectionChange(false);
      });

      this.socket.on('connect_error', (error) => {
        console.error('[Socket] Connection error:', error);
        this.notifyConnectionChange(false);
        resolve(false);
      });

      // Listen for gloss detection events
      this.socket.on('gloss_detected', (data: GlossEvent) => {
        console.log('[Socket] Gloss detected:', data.gloss);
        this.glossCallbacks.forEach(cb => cb(data));
      });

      // Listen for sentence updates
      this.socket.on('sentence_update', (data: SentenceEvent) => {
        console.log('[Socket] Sentence update:', data.sentence);
        this.sentenceCallbacks.forEach(cb => cb(data));
      });

      // Listen for session reset
      this.socket.on('session_reset', () => {
        console.log('[Socket] Session reset');
      });

      // Listen for speech captions from hearing user
      this.socket.on('speech_caption', (data: SpeechCaptionEvent) => {
        console.log('[Socket] Speech caption:', data.text);
        this.speechCaptionCallbacks.forEach(cb => cb(data));
      });

      // Listen for TTS requests
      this.socket.on('tts_request', (data: TTSRequestEvent) => {
        console.log('[Socket] TTS request:', data.text);
        this.ttsRequestCallbacks.forEach(cb => cb(data));
      });

      // Listen for conversation messages
      this.socket.on('conversation_message', (data: ConversationMessageEvent) => {
        console.log('[Socket] Conversation message:', data.text);
        this.conversationMessageCallbacks.forEach(cb => cb(data));
      });

      // Timeout after 5 seconds
      setTimeout(() => {
        if (!this.socket?.connected) {
          resolve(false);
        }
      }, 5000);
    });
  }

  /**
   * Disconnect from the server
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  /**
   * Send a video frame via WebSocket
   */
  sendFrame(frameBase64: string) {
    if (this.socket?.connected) {
      this.socket.emit('frame', { frame: frameBase64 });
    }
  }

  /**
   * Signal end of sign (trigger prediction)
   */
  endSign() {
    if (this.socket?.connected) {
      this.socket.emit('end_sign');
    }
  }

  /**
   * Register callback for gloss detection
   */
  onGlossDetected(callback: GlossCallback) {
    this.glossCallbacks.push(callback);
    return () => {
      this.glossCallbacks = this.glossCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register callback for sentence updates
   */
  onSentenceUpdate(callback: SentenceCallback) {
    this.sentenceCallbacks.push(callback);
    return () => {
      this.sentenceCallbacks = this.sentenceCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register callback for connection changes
   */
  onConnectionChange(callback: ConnectionCallback) {
    this.connectionCallbacks.push(callback);
    return () => {
      this.connectionCallbacks = this.connectionCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register callback for speech captions
   */
  onSpeechCaption(callback: SpeechCaptionCallback) {
    this.speechCaptionCallbacks.push(callback);
    return () => {
      this.speechCaptionCallbacks = this.speechCaptionCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register callback for TTS requests
   */
  onTTSRequest(callback: TTSRequestCallback) {
    this.ttsRequestCallbacks.push(callback);
    return () => {
      this.ttsRequestCallbacks = this.ttsRequestCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register callback for conversation messages
   */
  onConversationMessage(callback: ConversationMessageCallback) {
    this.conversationMessageCallbacks.push(callback);
    return () => {
      this.conversationMessageCallbacks = this.conversationMessageCallbacks.filter(cb => cb !== callback);
    };
  }

  private notifyConnectionChange(connected: boolean) {
    this.connectionCallbacks.forEach(cb => cb(connected));
  }
}

export const socketService = new SocketService();
export default socketService;
