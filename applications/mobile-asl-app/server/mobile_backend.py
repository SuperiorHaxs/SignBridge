"""
Mobile ASL Backend Server
Combines CV (pose detection + sign prediction) and LLM (Gemini sentence construction)
into a single HTTP/WebSocket server for mobile app communication.
"""

import os
import sys
import time
import json
import base64
import threading
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "openhands-modernized" / "src"))

# Import model inference utilities
try:
    from util.openhands_modernized_inference import load_model_from_checkpoint, predict_pose_file
    from openhands_modernized import MediaPipeSubset, PoseTransforms
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import model utilities: {e}")
    MODEL_AVAILABLE = False

# Import MediaPipe for pose extraction
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("WARNING: MediaPipe not installed")
    MEDIAPIPE_AVAILABLE = False

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("WARNING: google-generativeai not installed")
    GEMINI_AVAILABLE = False

# Flask app setup
app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBuQp46-lgxrC_-R6vDoX3CsyooCWtkAPk')
DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "models" / "openhands-modernized" / "production-models" / "wlasl_100_class_model")
CAPTION_BUFFER_SIZE = 3  # Glosses before Gemini translation

# Global state
model = None
tokenizer = None
mp_holistic = None
session_data = {
    'glosses': [],
    'running_caption': '',
    'all_sentences': [],
    'frame_buffer': [],  # Buffer for motion detection
    'is_signing': False,
    'pose_sequence': [],  # Accumulated poses during signing
    'last_prediction_time': 0
}
session_lock = threading.Lock()


def initialize_model(checkpoint_path=None):
    """Load the ASL prediction model"""
    global model, tokenizer

    if not MODEL_AVAILABLE:
        print("[BACKEND] Model utilities not available")
        return False

    checkpoint = checkpoint_path or DEFAULT_CHECKPOINT

    if not os.path.exists(checkpoint):
        print(f"[BACKEND] Checkpoint not found: {checkpoint}")
        return False

    try:
        print(f"[BACKEND] Loading model from: {checkpoint}")
        model, tokenizer, _ = load_model_from_checkpoint(checkpoint)
        print(f"[BACKEND] Model loaded successfully! Vocabulary size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"[BACKEND] Failed to load model: {e}")
        return False


def initialize_mediapipe():
    """Initialize MediaPipe Holistic for pose extraction"""
    global mp_holistic

    if not MEDIAPIPE_AVAILABLE:
        print("[BACKEND] MediaPipe not available")
        return False

    try:
        mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[BACKEND] MediaPipe Holistic initialized")
        return True
    except Exception as e:
        print(f"[BACKEND] Failed to initialize MediaPipe: {e}")
        return False


def extract_pose_from_frame(frame):
    """Extract pose landmarks from a single frame using MediaPipe"""
    if mp_holistic is None:
        return None

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = mp_holistic.process(rgb_frame)

    # Check if we have pose and at least one hand
    if results.pose_landmarks is None:
        return None

    # Extract landmarks into numpy array
    # Format: 576 points (33 pose + 468 face + 21 left hand + 21 right hand + 33 pose_world)
    # We only need: 33 pose + 21 left hand + 21 right hand = 75 points

    landmarks = np.zeros((576, 2), dtype=np.float32)

    # Pose landmarks (0-32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x, lm.y]

    # Left hand landmarks (501-521)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            landmarks[501 + i] = [lm.x, lm.y]

    # Right hand landmarks (522-542)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            landmarks[522 + i] = [lm.x, lm.y]

    return landmarks


def extract_pose_subset(full_landmarks):
    """Extract 75-point subset (pose + hands) from full MediaPipe output"""
    # Pose: indices 0-32 (33 points)
    # Left hand: indices 501-521 (21 points)
    # Right hand: indices 522-542 (21 points)

    pose = full_landmarks[0:33]
    left_hand = full_landmarks[501:522]
    right_hand = full_landmarks[522:543]

    return np.concatenate([pose, left_hand, right_hand])


def predict_from_poses(pose_sequence):
    """Make prediction from a sequence of pose frames"""
    global model, tokenizer

    if model is None or tokenizer is None:
        return None

    if len(pose_sequence) < 10:  # Minimum frames needed
        return None

    try:
        # Create temporary pickle file for prediction
        import pickle
        import tempfile

        # Stack poses into array (frames, 75, 2)
        poses_array = np.stack(pose_sequence)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump({'keypoints': poses_array}, f)
            temp_path = f.name

        # Run prediction
        result = predict_pose_file(temp_path, model=model, tokenizer=tokenizer)

        # Clean up
        os.unlink(temp_path)

        return result

    except Exception as e:
        print(f"[BACKEND] Prediction error: {e}")
        return None


def construct_sentence_with_gemini(glosses, running_caption=None, is_reconstruction=False):
    """Construct natural sentence from glosses using Gemini"""

    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        # Fallback: join glosses
        return " ".join([g.get('gloss', str(g)) if isinstance(g, dict) else str(g) for g in glosses])

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Build gloss details
        gloss_details = []
        for i, g in enumerate(glosses, 1):
            if isinstance(g, dict):
                detail = f"Position {i}:\n"
                top_k = g.get('top_k_predictions', [])
                if top_k:
                    for j, pred in enumerate(top_k[:3], 1):
                        detail += f"  Option {j}: '{pred['gloss']}' ({pred['confidence']:.1%})\n"
                else:
                    detail += f"  '{g.get('gloss', 'UNKNOWN')}'\n"
                gloss_details.append(detail)
            else:
                gloss_details.append(f"Position {i}: '{g}'")

        if is_reconstruction and running_caption:
            prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar, creating live captions.

Previous Caption: "{running_caption}"

ALL ASL Glosses (chronological order):
{''.join(gloss_details)}

Instructions - RECONSTRUCTION MODE:
1. COMPLETELY REWRITE the caption from scratch using ALL glosses
2. For each position with 3 options, choose the most coherent combination
3. Use natural pronouns and combine related ideas
4. Optimize for grammatical correctness and natural flow

Return only the reconstructed caption.
"""
        else:
            context = f'\nCurrent Running Caption: "{running_caption}"\n' if running_caption else ""
            prompt = f"""
You are an expert in ASL and English grammar, creating live captions.
{context}
New ASL Glosses to add:
{''.join(gloss_details)}

Instructions:
1. EXTEND the current caption with these new words
2. For positions with multiple options, choose the most coherent
3. Add articles, prepositions, and conjunctions as needed
4. Maintain proper grammar and natural flow

Return the COMPLETE caption (existing + new).
"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print(f"[BACKEND] Gemini error: {e}")
        return " ".join([g.get('gloss', str(g)) if isinstance(g, dict) else str(g) for g in glosses])


# ==================== HTTP ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'mediapipe_ready': mp_holistic is not None,
        'gemini_available': GEMINI_AVAILABLE
    })


@app.route('/api/process-frame', methods=['POST'])
def process_frame():
    """
    Process a single video frame for pose extraction and sign detection.

    Expected: POST with JSON body containing base64-encoded image
    {
        "frame": "base64_encoded_image_data",
        "timestamp": 12345  # optional
    }
    """
    try:
        data = request.get_json()
        frame_b64 = data.get('frame')

        if not frame_b64:
            return jsonify({'error': 'No frame provided'}), 400

        # Decode base64 image
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Extract pose
        full_landmarks = extract_pose_from_frame(frame)

        if full_landmarks is None:
            return jsonify({
                'success': True,
                'pose_detected': False,
                'message': 'No pose detected in frame'
            })

        # Extract 75-point subset
        pose_75 = extract_pose_subset(full_landmarks)

        # Add to pose sequence
        with session_lock:
            session_data['pose_sequence'].append(pose_75)

            # Keep last 256 frames max (model limit)
            if len(session_data['pose_sequence']) > 256:
                session_data['pose_sequence'] = session_data['pose_sequence'][-256:]

        return jsonify({
            'success': True,
            'pose_detected': True,
            'frame_count': len(session_data['pose_sequence'])
        })

    except Exception as e:
        print(f"[BACKEND] Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-sign', methods=['POST'])
def predict_sign():
    """
    Predict sign from accumulated pose sequence.
    Call this when user finishes a sign (e.g., hand motion stops).
    """
    try:
        with session_lock:
            poses = session_data['pose_sequence'].copy()
            session_data['pose_sequence'] = []  # Clear for next sign

        if len(poses) < 10:
            return jsonify({
                'success': False,
                'error': 'Not enough frames for prediction',
                'frame_count': len(poses)
            })

        # Make prediction
        result = predict_from_poses(poses)

        if result is None:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            })

        # Add gloss to session
        gloss_data = {
            'gloss': result['gloss'],
            'confidence': result['confidence'],
            'top_k_predictions': result.get('top_k_predictions', [])
        }

        with session_lock:
            session_data['glosses'].append(gloss_data)

        # Emit via WebSocket
        socketio.emit('gloss_detected', gloss_data)

        # Check if we should construct sentence
        if len(session_data['glosses']) >= CAPTION_BUFFER_SIZE:
            # Trigger sentence construction
            sentence = construct_sentence_with_gemini(
                session_data['glosses'],
                running_caption=session_data['running_caption'],
                is_reconstruction=True
            )

            with session_lock:
                session_data['running_caption'] = sentence
                session_data['all_sentences'].append(sentence)

            # Emit sentence update
            socketio.emit('sentence_update', {'sentence': sentence})

        return jsonify({
            'success': True,
            'gloss': result['gloss'],
            'confidence': result['confidence'],
            'top_k': result.get('top_k_predictions', [])[:5],
            'current_caption': session_data['running_caption']
        })

    except Exception as e:
        print(f"[BACKEND] Error predicting sign: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/add-gloss', methods=['POST'])
def add_gloss():
    """
    Manually add a gloss (for testing or when using external detection).

    Expected: POST with JSON body
    {
        "gloss": "hello",
        "confidence": 0.95
    }
    """
    try:
        data = request.get_json()
        gloss = data.get('gloss')
        confidence = data.get('confidence', 1.0)

        if not gloss:
            return jsonify({'error': 'No gloss provided'}), 400

        gloss_data = {
            'gloss': gloss,
            'confidence': confidence,
            'top_k_predictions': [{'gloss': gloss, 'confidence': confidence}]
        }

        with session_lock:
            session_data['glosses'].append(gloss_data)

        # Emit via WebSocket
        socketio.emit('gloss_detected', gloss_data)

        # Check if we should construct sentence
        if len(session_data['glosses']) >= CAPTION_BUFFER_SIZE:
            sentence = construct_sentence_with_gemini(
                session_data['glosses'],
                running_caption=session_data['running_caption'],
                is_reconstruction=True
            )

            with session_lock:
                session_data['running_caption'] = sentence

            socketio.emit('sentence_update', {'sentence': sentence})

        return jsonify({
            'success': True,
            'gloss_count': len(session_data['glosses']),
            'current_caption': session_data['running_caption']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/construct-sentence', methods=['POST'])
def construct_sentence():
    """
    Force sentence construction from current glosses.
    """
    try:
        with session_lock:
            glosses = session_data['glosses'].copy()

        if not glosses:
            return jsonify({'error': 'No glosses to construct sentence from'}), 400

        sentence = construct_sentence_with_gemini(
            glosses,
            running_caption=session_data['running_caption'],
            is_reconstruction=True
        )

        with session_lock:
            session_data['running_caption'] = sentence
            session_data['all_sentences'].append(sentence)

        socketio.emit('sentence_update', {'sentence': sentence})

        return jsonify({
            'success': True,
            'sentence': sentence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session state for new conversation"""
    with session_lock:
        session_data['glosses'] = []
        session_data['running_caption'] = ''
        session_data['all_sentences'] = []
        session_data['pose_sequence'] = []

    socketio.emit('session_reset', {})

    return jsonify({'success': True, 'message': 'Session reset'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current session status"""
    with session_lock:
        return jsonify({
            'gloss_count': len(session_data['glosses']),
            'current_caption': session_data['running_caption'],
            'sentence_count': len(session_data['all_sentences']),
            'frame_buffer_size': len(session_data['pose_sequence'])
        })


@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    Process speech text from hearing user and broadcast to ASL user.
    The mobile app handles actual speech recognition - this just broadcasts.

    Expected: POST with JSON body
    {
        "text": "Hello, how are you?",
        "is_final": true
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        is_final = data.get('is_final', True)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Broadcast to all clients (ASL user will see this)
        socketio.emit('speech_caption', {
            'text': text,
            'is_final': is_final,
            'speaker': 'hearing_user'
        })

        return jsonify({
            'success': True,
            'text': text,
            'is_final': is_final
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech_request():
    """
    Signal that ASL caption should be spoken (TTS handled on client).

    Expected: POST with JSON body
    {
        "text": "I want to go to the store"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Broadcast TTS request to hearing user's device
        socketio.emit('tts_request', {
            'text': text,
            'speaker': 'asl_user'
        })

        return jsonify({
            'success': True,
            'text': text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/message', methods=['POST'])
def add_conversation_message():
    """
    Add a message to the conversation from either party.

    Expected: POST with JSON body
    {
        "text": "Hello",
        "speaker": "asl_user" or "hearing_user",
        "type": "asl_caption" or "speech"
    }
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        speaker = data.get('speaker', 'unknown')
        msg_type = data.get('type', 'text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        message = {
            'text': text,
            'speaker': speaker,
            'type': msg_type,
            'timestamp': time.time()
        }

        # Broadcast to all clients
        socketio.emit('conversation_message', message)

        return jsonify({
            'success': True,
            'message': message
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    print(f"[BACKEND] Client connected")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[BACKEND] Client disconnected")


@socketio.on('frame')
def handle_frame(data):
    """Handle frame sent via WebSocket (alternative to HTTP)"""
    try:
        frame_b64 = data.get('frame')
        if not frame_b64:
            return

        # Decode and process
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        full_landmarks = extract_pose_from_frame(frame)

        if full_landmarks is not None:
            pose_75 = extract_pose_subset(full_landmarks)

            with session_lock:
                session_data['pose_sequence'].append(pose_75)
                if len(session_data['pose_sequence']) > 256:
                    session_data['pose_sequence'] = session_data['pose_sequence'][-256:]

            emit('pose_detected', {'frame_count': len(session_data['pose_sequence'])})

    except Exception as e:
        print(f"[BACKEND] WebSocket frame error: {e}")


@socketio.on('end_sign')
def handle_end_sign():
    """Signal that user finished signing - trigger prediction"""
    with session_lock:
        poses = session_data['pose_sequence'].copy()
        session_data['pose_sequence'] = []

    if len(poses) < 10:
        emit('prediction_error', {'error': 'Not enough frames'})
        return

    result = predict_from_poses(poses)

    if result:
        gloss_data = {
            'gloss': result['gloss'],
            'confidence': result['confidence'],
            'top_k_predictions': result.get('top_k_predictions', [])
        }

        with session_lock:
            session_data['glosses'].append(gloss_data)

        emit('gloss_detected', gloss_data, broadcast=True)

        # Check if we should construct sentence
        if len(session_data['glosses']) >= CAPTION_BUFFER_SIZE:
            sentence = construct_sentence_with_gemini(
                session_data['glosses'],
                running_caption=session_data['running_caption'],
                is_reconstruction=True
            )

            with session_lock:
                session_data['running_caption'] = sentence

            emit('sentence_update', {'sentence': sentence}, broadcast=True)


# ==================== MAIN ====================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Mobile ASL Backend Server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    args = parser.parse_args()

    print("=" * 70)
    print("MOBILE ASL BACKEND SERVER")
    print("=" * 70)

    # Initialize components
    print("\n[1/3] Initializing MediaPipe...")
    if not initialize_mediapipe():
        print("WARNING: MediaPipe initialization failed - pose detection disabled")

    print("\n[2/3] Loading ASL model...")
    if not initialize_model(args.checkpoint):
        print("WARNING: Model loading failed - predictions disabled")

    print("\n[3/3] Configuring Gemini...")
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        print(f"Gemini API key: {'Configured' if GEMINI_API_KEY else 'NOT SET'}")
    else:
        print("WARNING: Gemini not available - using gloss fallback")

    print("\n" + "=" * 70)
    print(f"Server starting on http://{args.host}:{args.port}")
    print("=" * 70)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /api/process-frame   - Process video frame")
    print("  POST /api/predict-sign    - Predict sign from poses")
    print("  POST /api/add-gloss       - Manually add gloss")
    print("  POST /api/construct-sentence - Force sentence construction")
    print("  POST /api/reset           - Reset session")
    print("  GET  /api/status          - Get session status")
    print("\nWebSocket events:")
    print("  frame      - Send video frame")
    print("  end_sign   - Signal end of sign")
    print("=" * 70 + "\n")

    # Run server
    socketio.run(app, host=args.host, port=args.port, debug=False)
