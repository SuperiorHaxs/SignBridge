import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Platform,
  useWindowDimensions,
} from 'react-native';
import apiService from '../services/api';
import socketService, {
  GlossEvent,
  SentenceEvent,
  SpeechCaptionEvent,
  ConversationMessageEvent,
} from '../services/socket';

// Only import expo-camera for native platforms
let CameraView: any = null;
if (Platform.OS !== 'web') {
  const ExpoCamera = require('expo-camera');
  CameraView = ExpoCamera.CameraView;
}

interface Message {
  id: number;
  type: 'asl' | 'speech';
  speaker: 'asl_user' | 'hearing_user';
  text: string;
  timestamp: Date;
}

// ============================================
// Web Camera Component (for browser support)
// ============================================
function WebCamera({
  onFrame,
  isRecording,
}: {
  onFrame: (base64: string) => void;
  isRecording: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const captureIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    async function setupCamera() {
      console.log('[WebCamera] Setting up camera...');
      try {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          console.error('[WebCamera] getUserMedia not supported');
          setHasPermission(false);
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
        });

        console.log('[WebCamera] Got camera stream');

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            console.log('[WebCamera] Video metadata loaded');
            videoRef.current?.play();
          };
          setHasPermission(true);
        }
      } catch (err: any) {
        console.error('[WebCamera] Camera access error:', err.name, err.message);
        setHasPermission(false);
      }
    }
    setupCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        console.log('[WebCamera] Camera stopped');
      }
    };
  }, []);

  useEffect(() => {
    if (isRecording && hasPermission) {
      captureIntervalRef.current = window.setInterval(() => {
        captureFrame();
      }, 200); // 5 FPS
    } else {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    }

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
      }
    };
  }, [isRecording, hasPermission]);

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
    const base64 = dataUrl.split(',')[1];
    onFrame(base64);
  };

  if (hasPermission === null) {
    return (
      <Text style={styles.cameraStatusText}>Requesting camera...</Text>
    );
  }
  if (hasPermission === false) {
    return (
      <Text style={styles.cameraStatusText}>Camera access denied</Text>
    );
  }

  return (
    <View style={styles.webCameraWrapper}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: 'scaleX(-1)',
        }}
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {isRecording && (
        <View style={styles.recordingBadge}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingBadgeText}>REC</Text>
        </View>
      )}
    </View>
  );
}

// ============================================
// Web Speech Recognition Hook
// ============================================
function useSpeechRecognition() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (Platform.OS !== 'web') return;

    // Check for browser support
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event: any) => {
      let interim = '';
      let final = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript + ' ';
        } else {
          interim += result[0].transcript;
        }
      }

      if (final) {
        setTranscript((prev) => prev + final);
        // Send final transcript to backend
        apiService.sendSpeechText(final.trim(), true);
      }
      setInterimTranscript(interim);
    };

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      if (event.error === 'not-allowed') {
        setIsListening(false);
      }
    };

    recognition.onend = () => {
      // Restart if still supposed to be listening
      if (isListening && recognitionRef.current) {
        try {
          recognition.start();
        } catch (e) {
          console.log('Recognition restart failed:', e);
        }
      }
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setTranscript('');
      setInterimTranscript('');
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {
        console.error('Failed to start recognition:', e);
      }
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      // Send any remaining transcript
      if (transcript.trim()) {
        apiService.sendSpeechText(transcript.trim(), true);
      }
    }
  };

  const clearTranscript = () => {
    setTranscript('');
    setInterimTranscript('');
  };

  return {
    isListening,
    transcript,
    interimTranscript,
    startListening,
    stopListening,
    clearTranscript,
    isSupported: Platform.OS === 'web' && !!recognitionRef.current,
  };
}

// ============================================
// Text-to-Speech Hook
// ============================================
function useTextToSpeech() {
  const [isReady, setIsReady] = useState(false);

  // Initialize TTS on first user interaction
  useEffect(() => {
    if (Platform.OS !== 'web') return;

    const initTTS = () => {
      if ('speechSynthesis' in window) {
        // Trigger voice loading
        window.speechSynthesis.getVoices();
        setIsReady(true);
      }
    };

    // Try to initialize immediately
    initTTS();

    // Also initialize on user interaction (required by some browsers)
    const handleInteraction = () => {
      initTTS();
      // Speak empty string to "unlock" TTS in some browsers
      if ('speechSynthesis' in window) {
        const unlock = new SpeechSynthesisUtterance('');
        unlock.volume = 0;
        window.speechSynthesis.speak(unlock);
      }
    };

    document.addEventListener('click', handleInteraction, { once: true });
    document.addEventListener('touchstart', handleInteraction, { once: true });

    return () => {
      document.removeEventListener('click', handleInteraction);
      document.removeEventListener('touchstart', handleInteraction);
    };
  }, []);

  const speak = (text: string) => {
    if (Platform.OS !== 'web') return;

    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;

      // Try to get a good English voice
      const voices = window.speechSynthesis.getVoices();
      const englishVoice = voices.find(v => v.lang.startsWith('en'));
      if (englishVoice) {
        utterance.voice = englishVoice;
      }

      utterance.onerror = (e) => {
        console.error('[TTS] Error:', e);
      };

      utterance.onstart = () => {
        console.log('[TTS] Speaking:', text);
      };

      window.speechSynthesis.speak(utterance);
    } else {
      console.warn('[TTS] Speech synthesis not supported');
    }
  };

  return { speak, isReady };
}

// ============================================
// Main Conversation Screen
// ============================================
export default function ConversationScreen() {
  // Get dynamic window dimensions
  const { width, height } = useWindowDimensions();
  const isWideScreen = width > 700; // Side-by-side on wider screens
  const isSmallHeight = height < 600;

  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [aslCaption, setAslCaption] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [autoSpeak, setAutoSpeak] = useState(true);

  // Refs
  const messageIdRef = useRef(0);
  const scrollViewRef = useRef<ScrollView>(null);
  const cameraRef = useRef<any>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Hooks
  const speech = useSpeechRecognition();
  const tts = useTextToSpeech();

  // Subscribe to socket events
  useEffect(() => {
    // ASL user's sentence updates
    const unsubSentence = socketService.onSentenceUpdate(
      (data: SentenceEvent) => {
        setAslCaption(data.sentence);
      }
    );

    // Gloss detection
    const unsubGloss = socketService.onGlossDetected((data: GlossEvent) => {
      console.log('[Conversation] Gloss:', data.gloss);
    });

    // Speech captions from hearing user (for remote sync)
    const unsubSpeech = socketService.onSpeechCaption(
      (data: SpeechCaptionEvent) => {
        console.log('[Conversation] Speech caption:', data.text);
      }
    );

    // Conversation messages (broadcast from server)
    const unsubConvo = socketService.onConversationMessage(
      (data: ConversationMessageEvent) => {
        addMessage(
          data.type === 'asl_caption' ? 'asl' : 'speech',
          data.speaker,
          data.text
        );
      }
    );

    return () => {
      unsubSentence();
      unsubGloss();
      unsubSpeech();
      unsubConvo();
      stopCapturing();
    };
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [messages]);

  // ============================================
  // Message Handling
  // ============================================
  const addMessage = (
    type: 'asl' | 'speech',
    speaker: 'asl_user' | 'hearing_user',
    text: string
  ) => {
    if (!text.trim()) return;

    const newMessage: Message = {
      id: messageIdRef.current++,
      type,
      speaker,
      text: text.trim(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
  };

  // ============================================
  // ASL Signer Controls
  // ============================================
  const handleWebFrame = async (base64: string) => {
    try {
      const result = await apiService.processFrame(base64);
      if (result.frame_count) {
        setFrameCount(result.frame_count);
      }
    } catch (error) {
      console.error('Frame process error:', error);
    }
  };

  const startCapturing = async () => {
    setIsRecording(true);
    setFrameCount(0);

    // Native camera capture
    if (Platform.OS !== 'web' && cameraRef.current) {
      captureIntervalRef.current = setInterval(async () => {
        try {
          if (cameraRef.current) {
            const photo = await cameraRef.current.takePictureAsync({
              base64: true,
              quality: 0.5,
              skipProcessing: true,
            });
            if (photo?.base64) {
              const result = await apiService.processFrame(photo.base64);
              if (result.frame_count) {
                setFrameCount(result.frame_count);
              }
            }
          }
        } catch (error) {
          console.error('Frame capture error:', error);
        }
      }, 200);
    }
  };

  const stopCapturing = async () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }

    setIsRecording(false);

    // Trigger prediction if enough frames
    if (frameCount >= 10) {
      const result = await apiService.predictSign();
      if (result.success && result.current_caption) {
        setAslCaption(result.current_caption);
      }
    }

    setFrameCount(0);
  };

  const sendAslCaption = async () => {
    if (!aslCaption.trim()) return;

    // Add to local messages
    addMessage('asl', 'asl_user', aslCaption);

    // Broadcast to other devices via server
    await apiService.addConversationMessage(aslCaption, 'asl_user', 'asl_caption');

    // Speak it if auto-speak is enabled
    if (autoSpeak) {
      tts.speak(aslCaption);
      await apiService.requestTTS(aslCaption);
    }

    // Reset
    setAslCaption('');
    await apiService.resetSession();
  };

  // ============================================
  // Hearing User Controls
  // ============================================
  const sendSpeechMessage = async () => {
    const text = speech.transcript.trim();
    if (!text) return;

    // Add to local messages
    addMessage('speech', 'hearing_user', text);

    // Broadcast to other devices
    await apiService.addConversationMessage(text, 'hearing_user', 'speech');

    // Clear transcript
    speech.clearTranscript();
  };

  const clearConversation = async () => {
    setMessages([]);
    setAslCaption('');
    speech.clearTranscript();
    await apiService.resetSession();
  };

  // ============================================
  // Render
  // ============================================
  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Two-Way Conversation</Text>
        <TouchableOpacity
          style={styles.autoSpeakToggle}
          onPress={() => setAutoSpeak(!autoSpeak)}
        >
          <Text style={styles.autoSpeakText}>
            Auto-Speak: {autoSpeak ? 'ON' : 'OFF'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Split Screen Layout */}
      <View style={[styles.splitContainer, { flexDirection: isWideScreen ? 'row' : 'column' }]}>
        {/* ASL User Section (Top/Left) */}
        <View style={styles.aslSection}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>ASL Signer</Text>
            <Text style={styles.sectionSubtitle}>Sign into camera</Text>
          </View>

          {/* Camera Preview */}
          <View style={[styles.cameraContainer, { height: isSmallHeight ? 80 : 120 }]}>
            {Platform.OS === 'web' ? (
              <WebCamera onFrame={handleWebFrame} isRecording={isRecording} />
            ) : (
              CameraView && (
                <CameraView
                  ref={cameraRef}
                  style={styles.camera}
                  facing="front"
                />
              )
            )}
            {isRecording && (
              <View style={styles.frameCountBadge}>
                <Text style={styles.frameCountText}>{frameCount} frames</Text>
              </View>
            )}
          </View>

          {/* ASL Caption Preview */}
          <View style={styles.captionPreview}>
            {aslCaption ? (
              <View style={styles.captionRow}>
                <Text style={styles.captionText} numberOfLines={2}>
                  {aslCaption}
                </Text>
                <TouchableOpacity
                  style={styles.sendCaptionButton}
                  onPress={sendAslCaption}
                >
                  <Text style={styles.sendCaptionText}>Send</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <Text style={styles.captionPlaceholder}>
                Caption will appear here
              </Text>
            )}
          </View>

          {/* Record Button */}
          <TouchableOpacity
            style={[
              styles.recordButton,
              isRecording && styles.recordButtonActive,
            ]}
            onPressIn={startCapturing}
            onPressOut={stopCapturing}
          >
            <Text style={styles.recordButtonText}>
              {isRecording ? 'Signing...' : 'Hold to Sign'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Divider */}
        <View style={[styles.divider, isWideScreen ? styles.dividerVertical : styles.dividerHorizontal]} />

        {/* Hearing User Section (Bottom/Right) */}
        <View style={styles.hearingSection}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Hearing User</Text>
            <Text style={styles.sectionSubtitle}>Speak into microphone</Text>
          </View>

          {/* Speech Display */}
          <View style={styles.speechDisplay}>
            {speech.isListening ? (
              <View style={styles.listeningIndicator}>
                <View style={styles.listeningDot} />
                <Text style={styles.listeningText}>Listening...</Text>
              </View>
            ) : null}
            <Text style={styles.speechText}>
              {speech.transcript}
              <Text style={styles.interimText}>{speech.interimTranscript}</Text>
            </Text>
            {!speech.transcript && !speech.interimTranscript && (
              <Text style={styles.speechPlaceholder}>
                {speech.isSupported
                  ? 'Speech will appear here'
                  : 'Speech recognition not supported in this browser'}
              </Text>
            )}
          </View>

          {/* Speech Controls */}
          <View style={styles.speechControls}>
            <TouchableOpacity
              style={[
                styles.micButton,
                speech.isListening && styles.micButtonActive,
              ]}
              onPress={speech.isListening ? speech.stopListening : speech.startListening}
              disabled={!speech.isSupported}
            >
              <Text style={styles.micButtonText}>
                {speech.isListening ? 'Stop' : 'Start Speaking'}
              </Text>
            </TouchableOpacity>

            {speech.transcript && (
              <TouchableOpacity
                style={styles.sendSpeechButton}
                onPress={sendSpeechMessage}
              >
                <Text style={styles.sendSpeechText}>Send</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </View>

      {/* Conversation History */}
      <View style={[styles.historySection, { height: isSmallHeight ? 120 : 180 }]}>
        <View style={styles.historyHeader}>
          <Text style={styles.historyTitle}>Conversation</Text>
          <TouchableOpacity onPress={clearConversation}>
            <Text style={styles.clearText}>Clear</Text>
          </TouchableOpacity>
        </View>

        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesScroll}
          contentContainerStyle={styles.messagesContent}
        >
          {messages.length === 0 ? (
            <Text style={styles.emptyText}>
              Messages will appear here
            </Text>
          ) : (
            messages.map((msg) => (
              <View
                key={msg.id}
                style={[
                  styles.messageBubble,
                  msg.speaker === 'asl_user'
                    ? styles.aslBubble
                    : styles.speechBubble,
                ]}
              >
                <Text style={styles.messageLabel}>
                  {msg.speaker === 'asl_user' ? 'ASL User' : 'Hearing User'}
                </Text>
                <Text style={styles.messageText}>{msg.text}</Text>
                <Text style={styles.messageTime}>
                  {msg.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </Text>
              </View>
            ))
          )}
        </ScrollView>
      </View>
    </View>
  );
}

// ============================================
// Styles
// ============================================
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f23',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#1a1a3e',
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a5e',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  autoSpeakToggle: {
    backgroundColor: '#3498db',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  autoSpeakText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  splitContainer: {
    flex: 1,
  },
  aslSection: {
    flex: 1,
    padding: 10,
    backgroundColor: '#1a1a2e',
  },
  hearingSection: {
    flex: 1,
    padding: 10,
    backgroundColor: '#162032',
  },
  divider: {
    backgroundColor: '#3498db',
  },
  dividerHorizontal: {
    height: 2,
    width: '100%',
  },
  dividerVertical: {
    width: 2,
    height: '100%',
  },
  sectionHeader: {
    marginBottom: 8,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sectionSubtitle: {
    color: '#7f8c8d',
    fontSize: 12,
  },
  cameraContainer: {
    backgroundColor: '#000',
    borderRadius: 8,
    overflow: 'hidden',
    marginBottom: 8,
  },
  camera: {
    flex: 1,
  },
  webCameraWrapper: {
    flex: 1,
    position: 'relative',
  },
  cameraStatusText: {
    color: '#7f8c8d',
    textAlign: 'center',
    padding: 20,
  },
  recordingBadge: {
    position: 'absolute',
    top: 8,
    left: 8,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(231, 76, 60, 0.9)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  recordingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#fff',
    marginRight: 4,
  },
  recordingBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  frameCountBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(52, 152, 219, 0.9)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  frameCountText: {
    color: '#fff',
    fontSize: 10,
  },
  captionPreview: {
    backgroundColor: '#2c3e50',
    borderRadius: 8,
    padding: 10,
    minHeight: 50,
    marginBottom: 8,
  },
  captionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  captionText: {
    color: '#fff',
    fontSize: 14,
    flex: 1,
    marginRight: 10,
  },
  captionPlaceholder: {
    color: '#7f8c8d',
    fontSize: 12,
    textAlign: 'center',
  },
  sendCaptionButton: {
    backgroundColor: '#27ae60',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 15,
  },
  sendCaptionText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  recordButton: {
    backgroundColor: '#e74c3c',
    paddingVertical: 12,
    borderRadius: 20,
    alignItems: 'center',
  },
  recordButtonActive: {
    backgroundColor: '#f39c12',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  speechDisplay: {
    backgroundColor: '#2c3e50',
    borderRadius: 8,
    padding: 10,
    minHeight: 80,
    marginBottom: 8,
  },
  listeningIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  listeningDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#27ae60',
    marginRight: 6,
  },
  listeningText: {
    color: '#27ae60',
    fontSize: 12,
    fontWeight: '600',
  },
  speechText: {
    color: '#fff',
    fontSize: 14,
  },
  interimText: {
    color: '#95a5a6',
  },
  speechPlaceholder: {
    color: '#7f8c8d',
    fontSize: 12,
    textAlign: 'center',
  },
  speechControls: {
    flexDirection: 'row',
    gap: 10,
  },
  micButton: {
    flex: 1,
    backgroundColor: '#27ae60',
    paddingVertical: 12,
    borderRadius: 20,
    alignItems: 'center',
  },
  micButtonActive: {
    backgroundColor: '#e74c3c',
  },
  micButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  sendSpeechButton: {
    backgroundColor: '#3498db',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 20,
    alignItems: 'center',
  },
  sendSpeechText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  historySection: {
    backgroundColor: '#0a0a1a',
    borderTopWidth: 1,
    borderTopColor: '#2a2a5e',
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#1a1a3e',
  },
  historyTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  clearText: {
    color: '#e74c3c',
    fontSize: 12,
  },
  messagesScroll: {
    flex: 1,
  },
  messagesContent: {
    padding: 10,
  },
  emptyText: {
    color: '#7f8c8d',
    textAlign: 'center',
    fontSize: 12,
  },
  messageBubble: {
    maxWidth: '75%',
    padding: 10,
    borderRadius: 12,
    marginBottom: 8,
  },
  aslBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#3498db',
  },
  speechBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#27ae60',
  },
  messageLabel: {
    color: 'rgba(255,255,255,0.6)',
    fontSize: 10,
    marginBottom: 2,
  },
  messageText: {
    color: '#fff',
    fontSize: 14,
  },
  messageTime: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 9,
    marginTop: 4,
    textAlign: 'right',
  },
});
