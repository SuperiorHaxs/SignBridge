import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Platform,
} from 'react-native';
import CaptionDisplay from '../components/CaptionDisplay';
import apiService from '../services/api';
import socketService, { GlossEvent, SentenceEvent } from '../services/socket';

// Only import expo-camera for native platforms
let CameraView: any = null;
let useCameraPermissions: any = null;

if (Platform.OS !== 'web') {
  const ExpoCamera = require('expo-camera');
  CameraView = ExpoCamera.CameraView;
  useCameraPermissions = ExpoCamera.useCameraPermissions;
}

// Web Camera Component
function WebCamera({ onFrame, isRecording }: { onFrame: (base64: string) => void; isRecording: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const captureIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    // Request camera access
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 640, height: 480 }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasPermission(true);
        }
      } catch (err) {
        console.error('Camera access denied:', err);
        setHasPermission(false);
      }
    }
    setupCamera();

    return () => {
      // Cleanup stream
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (isRecording && hasPermission) {
      // Start capturing frames
      captureIntervalRef.current = window.setInterval(() => {
        captureFrame();
      }, 200); // 5 FPS
    } else {
      // Stop capturing
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

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const dataUrl = canvas.toDataURL('image/jpeg', 0.5);
    const base64 = dataUrl.split(',')[1];
    onFrame(base64);
  };

  if (hasPermission === null) {
    return <Text style={styles.webCameraText}>Requesting camera permission...</Text>;
  }

  if (hasPermission === false) {
    return <Text style={styles.webCameraText}>Camera access denied. Please allow camera access.</Text>;
  }

  return (
    <View style={styles.webCameraContainer}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          transform: 'scaleX(-1)', // Mirror for selfie view
        }}
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {isRecording && (
        <View style={styles.recordingIndicator}>
          <View style={styles.recordingDot} />
          <Text style={styles.recordingText}>Recording...</Text>
        </View>
      )}
    </View>
  );
}

export default function SignerScreen() {
  const [isRecording, setIsRecording] = useState(false);
  const [currentCaption, setCurrentCaption] = useState('');
  const [glosses, setGlosses] = useState<string[]>([]);
  const [frameCount, setFrameCount] = useState(0);

  // Native camera (only used on mobile)
  const cameraRef = useRef<any>(null);
  const captureInterval = useRef<NodeJS.Timeout | null>(null);

  // For native platforms, use permission hook
  const [permission, setPermission] = useState<{ granted: boolean } | null>(
    Platform.OS === 'web' ? { granted: true } : null
  );

  useEffect(() => {
    if (Platform.OS !== 'web' && useCameraPermissions) {
      // This is a simplified permission check for native
      const checkPermission = async () => {
        try {
          const ExpoCamera = require('expo-camera');
          const [perm, requestPerm] = ExpoCamera.useCameraPermissions();
          setPermission(perm);
        } catch (e) {
          console.error('Permission check error:', e);
        }
      };
    }
  }, []);

  useEffect(() => {
    // Subscribe to WebSocket events
    const unsubGloss = socketService.onGlossDetected((data: GlossEvent) => {
      setGlosses(prev => [...prev, data.gloss]);
    });

    const unsubSentence = socketService.onSentenceUpdate((data: SentenceEvent) => {
      setCurrentCaption(data.sentence);
    });

    return () => {
      unsubGloss();
      unsubSentence();
      stopCapturing();
    };
  }, []);

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
      captureInterval.current = setInterval(async () => {
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
      }, 200); // 5 FPS
    }
  };

  const stopCapturing = async () => {
    if (captureInterval.current) {
      clearInterval(captureInterval.current);
      captureInterval.current = null;
    }

    setIsRecording(false);

    // Trigger prediction
    if (frameCount >= 10) {
      const result = await apiService.predictSign();
      if (result.success && result.gloss) {
        setGlosses(prev => [...prev, result.gloss!]);
        if (result.current_caption) {
          setCurrentCaption(result.current_caption);
        }
      }
    }

    setFrameCount(0);
  };

  const clearSession = async () => {
    await apiService.resetSession();
    setCurrentCaption('');
    setGlosses([]);
    setFrameCount(0);
  };

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <View style={styles.cameraContainer}>
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

        {/* Recording indicator for native */}
        {Platform.OS !== 'web' && isRecording && (
          <View style={styles.recordingIndicator}>
            <View style={styles.recordingDot} />
            <Text style={styles.recordingText}>Recording... ({frameCount} frames)</Text>
          </View>
        )}

        {/* Frame count for web */}
        {Platform.OS === 'web' && isRecording && (
          <View style={styles.frameCountBadge}>
            <Text style={styles.frameCountText}>{frameCount} frames</Text>
          </View>
        )}
      </View>

      {/* Detected Glosses */}
      <View style={styles.glossContainer}>
        <Text style={styles.glossLabel}>Detected Signs:</Text>
        <Text style={styles.glossText}>
          {glosses.length > 0 ? glosses.join(' → ') : 'No signs detected yet'}
        </Text>
      </View>

      {/* Caption Display */}
      <CaptionDisplay caption={currentCaption} />

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[
            styles.recordButton,
            isRecording ? styles.recordingButton : styles.notRecordingButton,
          ]}
          onPressIn={startCapturing}
          onPressOut={stopCapturing}
        >
          <Text style={styles.recordButtonText}>
            {isRecording ? 'Signing...' : 'Hold to Sign'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.clearButton} onPress={clearSession}>
          <Text style={styles.clearButtonText}>Clear</Text>
        </TouchableOpacity>
      </View>

      {/* Instructions */}
      <View style={styles.instructions}>
        <Text style={styles.instructionText}>
          Hold the button while signing, release when done
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  webCameraContainer: {
    flex: 1,
    position: 'relative',
    backgroundColor: '#000',
  },
  webCameraText: {
    color: '#fff',
    textAlign: 'center',
    padding: 20,
  },
  recordingIndicator: {
    position: 'absolute',
    top: 20,
    left: 20,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(231, 76, 60, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#fff',
    marginRight: 8,
  },
  recordingText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  frameCountBadge: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: 'rgba(52, 152, 219, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  frameCountText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  glossContainer: {
    backgroundColor: '#2c3e50',
    padding: 15,
  },
  glossLabel: {
    color: '#95a5a6',
    fontSize: 12,
    marginBottom: 5,
  },
  glossText: {
    color: '#fff',
    fontSize: 16,
  },
  controls: {
    flexDirection: 'row',
    padding: 20,
    backgroundColor: '#16213e',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButton: {
    flex: 1,
    paddingVertical: 20,
    borderRadius: 30,
    alignItems: 'center',
    marginRight: 10,
  },
  notRecordingButton: {
    backgroundColor: '#e74c3c',
  },
  recordingButton: {
    backgroundColor: '#f39c12',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  clearButton: {
    paddingVertical: 20,
    paddingHorizontal: 25,
    backgroundColor: '#95a5a6',
    borderRadius: 30,
  },
  clearButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  instructions: {
    padding: 15,
    backgroundColor: '#0f3460',
  },
  instructionText: {
    color: '#95a5a6',
    fontSize: 12,
    textAlign: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#3498db',
    padding: 15,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});
