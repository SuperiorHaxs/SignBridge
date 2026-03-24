import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Platform,
} from 'react-native';
import apiService from '../services/api';
import socketService, { ConversationMessageEvent } from '../services/socket';

interface HistoryItem {
  id: number;
  text: string;
  timestamp: Date;
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

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported');
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
        // Broadcast to ASL user
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
// Text-to-Speech Function
// ============================================
function speakText(text: string) {
  if (Platform.OS !== 'web') return;

  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    // Try to get English voice
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en'));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    utterance.onerror = (e) => console.error('[TTS] Error:', e);
    utterance.onstart = () => console.log('[TTS] Speaking:', text);

    window.speechSynthesis.speak(utterance);
  }
}

// Initialize TTS on page load (call this on first click)
function initTTS() {
  if (Platform.OS !== 'web') return;
  if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
    // Speak empty string to unlock TTS
    const unlock = new SpeechSynthesisUtterance('');
    unlock.volume = 0;
    window.speechSynthesis.speak(unlock);
  }
}

// ============================================
// Main Listener Screen
// ============================================
export default function ListenerScreen() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [aslCaptions, setAslCaptions] = useState<string[]>([]);
  const historyIdRef = useRef(0);
  const scrollViewRef = useRef<ScrollView>(null);

  const speech = useSpeechRecognition();

  // Initialize TTS on first user interaction
  useEffect(() => {
    const handleFirstInteraction = () => {
      initTTS();
    };
    document.addEventListener('click', handleFirstInteraction, { once: true });
    document.addEventListener('touchstart', handleFirstInteraction, { once: true });
    return () => {
      document.removeEventListener('click', handleFirstInteraction);
      document.removeEventListener('touchstart', handleFirstInteraction);
    };
  }, []);

  // Subscribe to socket events
  useEffect(() => {
    // Listen for ASL captions from signer
    const unsubConvo = socketService.onConversationMessage(
      (data: ConversationMessageEvent) => {
        if (data.speaker === 'asl_user') {
          setAslCaptions((prev) => [...prev.slice(-4), data.text]);
          // Auto-speak ASL captions
          speakText(data.text);
        }
      }
    );

    return () => {
      unsubConvo();
    };
  }, []);

  // Auto-scroll history
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  }, [history]);

  const sendMessage = async () => {
    const text = speech.transcript.trim();
    if (!text) return;

    // Add to history
    const newItem: HistoryItem = {
      id: historyIdRef.current++,
      text,
      timestamp: new Date(),
    };
    setHistory((prev) => [...prev, newItem]);

    // Broadcast to conversation
    await apiService.addConversationMessage(text, 'hearing_user', 'speech');

    // Clear transcript
    speech.clearTranscript();
  };

  const clearHistory = () => {
    setHistory([]);
    setAslCaptions([]);
    speech.clearTranscript();
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Hearing User Mode</Text>
        <Text style={styles.headerSubtitle}>
          Speak into microphone - your words will be shown to the ASL user
        </Text>
      </View>

      {/* ASL Captions from Signer (if any) */}
      {aslCaptions.length > 0 && (
        <View style={styles.aslCaptionContainer}>
          <Text style={styles.aslCaptionLabel}>ASL User Says:</Text>
          <Text style={styles.aslCaptionText}>
            {aslCaptions[aslCaptions.length - 1]}
          </Text>
        </View>
      )}

      {/* Current Speech */}
      <View style={styles.speechContainer}>
        <Text style={styles.label}>Your Speech:</Text>
        <View style={styles.speechBox}>
          {speech.isListening && (
            <View style={styles.listeningIndicator}>
              <View style={styles.listeningDot} />
              <Text style={styles.listeningText}>Listening...</Text>
            </View>
          )}
          <Text style={styles.speechText}>
            {speech.transcript}
            <Text style={styles.interimText}>{speech.interimTranscript}</Text>
          </Text>
          {!speech.transcript && !speech.interimTranscript && !speech.isListening && (
            <Text style={styles.placeholder}>
              {speech.isSupported
                ? 'Tap "Start Speaking" to begin'
                : 'Speech recognition not supported in this browser. Use Chrome for best results.'}
            </Text>
          )}
        </View>

        {/* Send Button */}
        {speech.transcript && (
          <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
            <Text style={styles.sendButtonText}>Send Message</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* History */}
      <View style={styles.historyContainer}>
        <View style={styles.historyHeader}>
          <Text style={styles.label}>Message History</Text>
          <TouchableOpacity onPress={clearHistory}>
            <Text style={styles.clearText}>Clear</Text>
          </TouchableOpacity>
        </View>
        <ScrollView ref={scrollViewRef} style={styles.historyScroll}>
          {history.length === 0 ? (
            <Text style={styles.emptyHistory}>No messages yet</Text>
          ) : (
            history.map((item) => (
              <TouchableOpacity
                key={item.id}
                style={styles.historyItem}
                onPress={() => speakText(item.text)}
              >
                <Text style={styles.historyText}>{item.text}</Text>
                <Text style={styles.historyMeta}>
                  {item.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}{' '}
                  - Tap to replay
                </Text>
              </TouchableOpacity>
            ))
          )}
        </ScrollView>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[
            styles.micButton,
            speech.isListening ? styles.micButtonActive : styles.micButtonInactive,
            !speech.isSupported && styles.micButtonDisabled,
          ]}
          onPress={speech.isListening ? speech.stopListening : speech.startListening}
          disabled={!speech.isSupported}
        >
          <Text style={styles.micButtonText}>
            {speech.isListening ? 'Stop Listening' : 'Start Speaking'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Status Bar */}
      <View style={styles.statusBar}>
        <View
          style={[
            styles.statusDot,
            speech.isSupported ? styles.statusOnline : styles.statusOffline,
          ]}
        />
        <Text style={styles.statusText}>
          {speech.isSupported
            ? speech.isListening
              ? 'Microphone active'
              : 'Ready to listen'
            : 'Speech recognition unavailable'}
        </Text>
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
    backgroundColor: '#162032',
  },
  header: {
    backgroundColor: '#1a3a5c',
    padding: 20,
    alignItems: 'center',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: '#95a5a6',
    fontSize: 13,
    marginTop: 5,
    textAlign: 'center',
  },
  aslCaptionContainer: {
    backgroundColor: '#3498db',
    padding: 15,
    marginHorizontal: 15,
    marginTop: 15,
    borderRadius: 12,
  },
  aslCaptionLabel: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    marginBottom: 5,
  },
  aslCaptionText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  speechContainer: {
    padding: 15,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ecf0f1',
    marginBottom: 10,
  },
  speechBox: {
    backgroundColor: '#2c3e50',
    borderRadius: 12,
    padding: 15,
    minHeight: 100,
  },
  listeningIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  listeningDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#27ae60',
    marginRight: 8,
  },
  listeningText: {
    color: '#27ae60',
    fontSize: 14,
    fontWeight: '600',
  },
  speechText: {
    color: '#fff',
    fontSize: 16,
    lineHeight: 24,
  },
  interimText: {
    color: '#95a5a6',
  },
  placeholder: {
    color: '#7f8c8d',
    fontSize: 14,
    textAlign: 'center',
  },
  sendButton: {
    backgroundColor: '#27ae60',
    paddingVertical: 12,
    borderRadius: 25,
    alignItems: 'center',
    marginTop: 10,
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  historyContainer: {
    flex: 1,
    padding: 15,
    paddingTop: 0,
  },
  historyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  clearText: {
    color: '#e74c3c',
    fontSize: 14,
  },
  historyScroll: {
    flex: 1,
    backgroundColor: '#1a2a3a',
    borderRadius: 12,
    padding: 10,
  },
  emptyHistory: {
    color: '#7f8c8d',
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: 20,
  },
  historyItem: {
    backgroundColor: '#27ae60',
    padding: 12,
    borderRadius: 10,
    marginBottom: 8,
  },
  historyText: {
    fontSize: 15,
    color: '#fff',
  },
  historyMeta: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.6)',
    marginTop: 5,
  },
  controls: {
    padding: 15,
    backgroundColor: '#1a3a5c',
  },
  micButton: {
    paddingVertical: 18,
    borderRadius: 30,
    alignItems: 'center',
  },
  micButtonInactive: {
    backgroundColor: '#27ae60',
  },
  micButtonActive: {
    backgroundColor: '#e74c3c',
  },
  micButtonDisabled: {
    backgroundColor: '#7f8c8d',
    opacity: 0.6,
  },
  micButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: '#0f1a2a',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusOnline: {
    backgroundColor: '#27ae60',
  },
  statusOffline: {
    backgroundColor: '#e74c3c',
  },
  statusText: {
    color: '#7f8c8d',
    fontSize: 12,
  },
});
