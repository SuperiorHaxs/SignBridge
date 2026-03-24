/**
 * API Service for communicating with the Mobile ASL Backend
 */

// Configuration - update this to your server's IP address
const SERVER_URL = 'http://192.168.1.100:5000';  // Replace with your computer's IP

export interface GlossData {
  gloss: string;
  confidence: number;
  top_k_predictions?: Array<{ gloss: string; confidence: number }>;
}

export interface PredictionResult {
  success: boolean;
  gloss?: string;
  confidence?: number;
  top_k?: Array<{ gloss: string; confidence: number }>;
  current_caption?: string;
  error?: string;
}

export interface SessionStatus {
  gloss_count: number;
  current_caption: string;
  sentence_count: number;
  frame_buffer_size: number;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = SERVER_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Update the server URL (for when user enters IP)
   */
  setServerUrl(url: string) {
    this.baseUrl = url;
  }

  /**
   * Health check - verify server is running
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Send a video frame for pose extraction
   */
  async processFrame(frameBase64: string): Promise<{ success: boolean; pose_detected: boolean; frame_count?: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/process-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameBase64 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Process frame error:', error);
      return { success: false, pose_detected: false };
    }
  }

  /**
   * Trigger sign prediction from accumulated poses
   */
  async predictSign(): Promise<PredictionResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/predict-sign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Predict sign error:', error);
      return { success: false, error: String(error) };
    }
  }

  /**
   * Manually add a gloss (for testing)
   */
  async addGloss(gloss: string, confidence: number = 1.0): Promise<{ success: boolean; current_caption?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/add-gloss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gloss, confidence }),
      });

      return await response.json();
    } catch (error) {
      console.error('Add gloss error:', error);
      return { success: false };
    }
  }

  /**
   * Force sentence construction
   */
  async constructSentence(): Promise<{ success: boolean; sentence?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/construct-sentence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      return await response.json();
    } catch (error) {
      console.error('Construct sentence error:', error);
      return { success: false };
    }
  }

  /**
   * Reset session
   */
  async resetSession(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      return response.ok;
    } catch (error) {
      console.error('Reset session error:', error);
      return false;
    }
  }

  /**
   * Get session status
   */
  async getStatus(): Promise<SessionStatus | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error('Get status error:', error);
      return null;
    }
  }

  /**
   * Send speech-to-text result to broadcast to ASL user
   */
  async sendSpeechText(text: string, isFinal: boolean = true): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/speech-to-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, is_final: isFinal }),
      });
      return response.ok;
    } catch (error) {
      console.error('Send speech text error:', error);
      return false;
    }
  }

  /**
   * Request text-to-speech for ASL caption
   */
  async requestTTS(text: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/text-to-speech`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      return response.ok;
    } catch (error) {
      console.error('TTS request error:', error);
      return false;
    }
  }

  /**
   * Add conversation message
   */
  async addConversationMessage(
    text: string,
    speaker: 'asl_user' | 'hearing_user',
    type: 'asl_caption' | 'speech'
  ): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/conversation/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speaker, type }),
      });
      return response.ok;
    } catch (error) {
      console.error('Add conversation message error:', error);
      return false;
    }
  }
}

export const apiService = new ApiService();
export default apiService;
