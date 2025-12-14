# ============================================================
# ASL REAL-TIME DETECTION SYSTEM (300 WORDS)
# Standalone version - extracted from ASL_Training_System.ipynb
# ============================================================

# ========================================
# CONFIGURATION - SET THIS BEFORE RUNNING
# ========================================
# Set to True to use Gemini AI, False to run without it
USE_GEMINI = False  # Change to True when you have your API key

print("="*70)
print("ASL REAL-TIME DETECTION SYSTEM (300 WORDS)")
print("="*70)

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os

# MediaPipe setup with optimized settings
mp_holistic = mp.solutions.holistic

gemini_model = None

if USE_GEMINI:
    print("\n🔧 Gemini AI Mode: ENABLED")
    # Try to import and set up Gemini
    try:
        import google.generativeai as genai
        print("✓ google-generativeai already installed")
    except ImportError:
        print("Installing google-generativeai...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai", "-q"])
        import google.generativeai as genai
        print("✓ google-generativeai installed successfully")

    # ========================================
    # GEMINI API SETUP
    # ========================================
    def setup_gemini(api_key):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            print("✓ Gemini API initialized successfully")
            return model
        except Exception as e:
            print(f"✗ Error initializing Gemini: {e}")
            return None

    def form_sentence_with_gemini(gemini_model, word_predictions):
        """
        Use Gemini to form a coherent sentence from top-3 predictions for each word.

        Args:
            gemini_model: The Gemini model instance
            word_predictions: List of tuples, each containing (top3_words, top3_confidences)

        Returns:
            str: Coherent English sentence formed by Gemini
        """
        if not word_predictions:
            return "No signs detected"

        # Format the prediction data for Gemini
        prompt = """You are an ASL (American Sign Language) interpreter assistant. I will give you the top 3 predictions for each signed word in a sequence, along with their confidence scores. Your job is to form the most coherent and grammatically correct English sentence from these predictions.

Here are the predictions for each signed word:

"""

        for i, (top3_words, top3_confs) in enumerate(word_predictions, 1):
            prompt += f"\nWord {i} predictions:\n"
            for j, (word, conf) in enumerate(zip(top3_words, top3_confs), 1):
                prompt += f"  {j}. {word} (confidence: {conf:.3f})\n"

        prompt += """
Based on these predictions, form the most coherent and grammatically correct English sentence. Consider:
1. Context between words
2. Confidence scores (higher is more likely correct)
3. Common ASL grammar patterns
4. Natural English sentence structure

Respond with ONLY the final sentence, nothing else. Make it natural and fluent."""

        try:
            response = gemini_model.generate_content(prompt)
            sentence = response.text.strip()
            print(f"\nGemini formed sentence: {sentence}")
            return sentence
        except Exception as e:
            print(f"Gemini error: {e}")
            # Fallback to most confident predictions
            fallback = ' '.join([words[0] for words, _ in word_predictions])
            return fallback

    # Get Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠ No GEMINI_API_KEY found in environment.")
        print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
        print("\n💡 TIP: Set the environment variable or edit the script to add your key")
        print("Switching to Simple Mode for now...")
        USE_GEMINI = False
        gemini_model = None
    else:
        gemini_model = setup_gemini(api_key)
        if gemini_model is None:
            print("⚠ Gemini initialization failed - switching to Simple Mode")
            USE_GEMINI = False

if USE_GEMINI and gemini_model:
    print("\n✅ Running in ADVANCED MODE with Gemini AI")
else:
    print("\n🔧 Gemini AI Mode: DISABLED")
    print("✅ Running in SIMPLE MODE (no Gemini)")

# ========================================
# CONFIGURATION - 300 WORDS
# ========================================
actions = np.array([
    # Core vocabulary (1-50)
    'book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin',
    'deaf', 'fine', 'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black',
    'cool', 'finish', 'hot', 'like', 'many', 'mother', 'now', 'orange', 'table', 'thanksgiving',
    'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'family', 'fish', 'graduate',
    'hat', 'hearing', 'kiss', 'language', 'later', 'man', 'shirt', 'study', 'tall', 'white',

    # Common verbs (51-100)
    'eat', 'sleep', 'play', 'work', 'read', 'write', 'watch', 'listen', 'speak', 'run',
    'jump', 'sit', 'stand', 'dance', 'sing', 'cook', 'clean', 'wash', 'drive', 'ride',
    'swim', 'fly', 'buy', 'sell', 'give', 'take', 'make', 'break', 'open', 'close',
    'start', 'stop', 'come', 'leave', 'arrive', 'wait', 'hurry', 'slow', 'fast', 'think',
    'know', 'understand', 'remember', 'forget', 'learn', 'teach', 'ask', 'answer', 'tell', 'say',

    # Common nouns (101-150)
    'house', 'school', 'hospital', 'store', 'restaurant', 'library', 'church', 'office', 'hotel', 'airport',
    'car', 'bus', 'train', 'plane', 'bike', 'boat', 'food', 'water', 'milk', 'bread',
    'meat', 'chicken', 'egg', 'cheese', 'fruit', 'apple', 'banana', 'vegetable', 'tomato', 'potato',
    'coffee', 'tea', 'juice', 'soda', 'beer', 'wine', 'ice', 'fire', 'sun', 'moon',
    'star', 'rain', 'snow', 'wind', 'cloud', 'tree', 'flower', 'grass', 'mountain', 'river',

    # People & relationships (151-200)
    'father', 'sister', 'brother', 'son', 'daughter', 'baby', 'child', 'parent', 'grandparent', 'grandfather',
    'grandmother', 'uncle', 'aunt', 'friend', 'teacher', 'student', 'doctor', 'nurse', 'police', 'firefighter',
    'boy', 'girl', 'husband', 'wife', 'boyfriend', 'girlfriend', 'neighbor', 'stranger', 'boss', 'worker',
    'actor', 'singer', 'dancer', 'artist', 'musician', 'writer', 'chef', 'farmer', 'driver', 'pilot',
    'lawyer', 'engineer', 'scientist', 'soldier', 'president', 'king', 'queen', 'prince', 'princess', 'hero',

    # Adjectives & descriptors (201-250)
    'good', 'bad', 'happy', 'sad', 'angry', 'tired', 'sick', 'healthy', 'strong', 'weak',
    'big', 'small', 'long', 'short', 'heavy', 'light', 'hard', 'soft', 'rough', 'smooth',
    'clean', 'dirty', 'new', 'old', 'young', 'beautiful', 'ugly', 'smart', 'stupid', 'funny',
    'serious', 'lazy', 'busy', 'rich', 'poor', 'expensive', 'cheap', 'free', 'full', 'empty',
    'right', 'wrong', 'true', 'false', 'easy', 'difficult', 'simple', 'complex', 'loud', 'quiet',

    # Time, numbers & misc (251-300)
    'today', 'tomorrow', 'yesterday', 'morning', 'afternoon', 'evening', 'night', 'day', 'week', 'month',
    'spring', 'summer', 'fall', 'winter', 'hour', 'minute', 'second', 'early', 'late', 'always',
    'never', 'sometimes', 'often', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'hundred', 'thousand', 'first', 'last', 'next', 'again', 'more',
    'less', 'same', 'different', 'another', 'other', 'here', 'there', 'where', 'when', 'why'
])

sequence_length = 30
threshold = 0.6  # Lower threshold since we're collecting top-3

# ========================================
# INPUT SOURCE SELECTION
# ========================================
INPUT_SOURCE = 0  # 0 for webcam, or path to video file

# ========================================
# HELPER FUNCTIONS
# ========================================

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz_top3(res, actions, input_frame):
    """Visualize TOP-3 prediction probabilities"""
    output_frame = input_frame.copy()

    # Get TOP-3 predictions
    sorted_indices = np.argsort(res)[::-1][:3]

    for i, idx in enumerate(sorted_indices):
        prob = res[idx]
        action = actions[idx]

        # Color based on rank
        if i == 0:  # Top prediction
            color = (0, 255, 0) if prob > 0.7 else (0, 255, 255)
        elif i == 1:  # 2nd
            color = (100, 200, 255)
        else:  # 3rd
            color = (150, 150, 200)

        # Draw probability bar
        cv2.rectangle(output_frame, (10, 180 + i * 35), (int(prob * 300) + 10, 210 + i * 35), color, -1)

        # Draw text with rank
        text = f"#{i+1}: {action} ({prob:.3f})"
        cv2.putText(output_frame, text, (15, 203 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return output_frame

# ========================================
# LOAD MODEL
# ========================================
print("\nLoading trained 300-word model...")
try:
    model = load_model('best_asl_300_model.h5')
    print("✓ Best 300-word model loaded successfully!")
except:
    try:
        model = load_model('action_300words.h5')
        print("✓ 300-word model loaded from action_300words.h5")
    except:
        try:
            model = load_model('improved_model.h5')
            print("✓ Model loaded from improved_model.h5")
        except:
            print("✗ ERROR: No trained model found!")
            print("Please ensure your trained model is in the current directory.")
            print("Expected filenames: best_asl_300_model.h5, action_300words.h5, or improved_model.h5")
            exit(1)

# ========================================
# REAL-TIME DETECTION WITH TOP-3 COLLECTION
# ========================================

# Initialize variables
sequence = []
collected_predictions = []  # Stores (top3_words, top3_confidences) for each sign
predictions_buffer = []
gemini_sentence = ""  # Stores Gemini-formed sentence
last_prediction_time = time.time()
min_prediction_interval = 2.0  # Seconds between predictions
frame_count = 0

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Perform ASL signs clearly in front of camera")
print("2. Hold each sign steady for 2-3 seconds")
if USE_GEMINI and gemini_model:
    print("3. Press SPACEBAR to finish signing and form sentence with Gemini")
else:
    print("3. Press SPACEBAR to finish and see collected signs")
print("4. Press 'c' to clear collected signs")
print("5. Press 'q' to quit")
print("="*70 + "\n")

print(f"Recognizing {len(actions)} words with TOP-3 predictions")
print("Starting detection...\n")

cap = cv2.VideoCapture(INPUT_SOURCE)

# Optimize webcam settings
if INPUT_SOURCE == 0:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
else:
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame
            image, results = mediapipe_detection(frame, holistic)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            # Display info overlay
            cv2.rectangle(image, (0, 0), (640, 170), (0, 0, 0), -1)

            cv2.putText(image, f'Sequence: {len(sequence)}/30', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(image, f'Signs Collected: {len(collected_predictions)}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            mode_text = 'MODE: Advanced (Gemini)' if (USE_GEMINI and gemini_model) else 'MODE: Simple'
            cv2.putText(image, mode_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

            cv2.putText(image, 'TOP-3 PREDICTIONS:', (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Make prediction when sequence is full
            if len(sequence) == sequence_length and frame_count % 3 == 0:
                try:
                    # Get predictions
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                    # Get TOP-3 predictions
                    top3_indices = np.argsort(res)[::-1][:3]
                    top3_words = [actions[i] for i in top3_indices]
                    top3_confs = [res[i] for i in top3_indices]

                    predicted_action = top3_words[0]
                    confidence = top3_confs[0]

                    # Store prediction
                    predictions_buffer.append((top3_indices[0], top3_words, top3_confs))

                    # Stabilized prediction logic
                    if len(predictions_buffer) >= 10:
                        recent_pred_indices = [p[0] for p in predictions_buffer[-10:]]
                        most_common = max(set(recent_pred_indices), key=recent_pred_indices.count)
                        consistency = recent_pred_indices.count(most_common) / len(recent_pred_indices)

                        current_time = time.time()

                        # Add to collected predictions if confident and consistent
                        if (confidence > threshold and
                            consistency >= 0.6 and
                            current_time - last_prediction_time > min_prediction_interval):

                            # Check if not duplicate
                            if (len(collected_predictions) == 0 or
                                top3_words[0] != collected_predictions[-1][0][0]):

                                collected_predictions.append((top3_words, top3_confs))
                                print(f"\nCollected sign #{len(collected_predictions)}: {top3_words[0]}")
                                print(f"  Top 3: {', '.join([f'{w}({c:.2f})' for w, c in zip(top3_words, top3_confs)])}")
                                last_prediction_time = current_time

                    # Visualize TOP-3 probabilities
                    image = prob_viz_top3(res, actions, image)

                except Exception as e:
                    cv2.putText(image, 'Prediction Error', (10, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Display collected signs
            if collected_predictions:
                signs_text = ' + '.join([words[0] for words, _ in collected_predictions[-5:]])
                cv2.putText(image, f'Collected: {signs_text}', (10, image.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Display Gemini sentence or simple sentence
            if gemini_sentence:
                max_width = 70
                if len(gemini_sentence) > max_width:
                    wrapped = gemini_sentence[:max_width] + "..."
                else:
                    wrapped = gemini_sentence

                label = 'Gemini: ' if (USE_GEMINI and gemini_model) else 'Signs: '
                cv2.putText(image, f'{label}{wrapped}', (10, image.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Instructions
            instructions = "SPACE: Form Sentence | C: Clear | Q: Quit" if (USE_GEMINI and gemini_model) else "SPACE: Finish | C: Clear | Q: Quit"
            cv2.putText(image, instructions,
                       (10, image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Show frame
            window_title = 'ASL Detection with Gemini (300 Words)' if (USE_GEMINI and gemini_model) else 'ASL Detection - Simple Mode (300 Words)'
            cv2.imshow(window_title, image)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                # Clear collected predictions
                collected_predictions = []
                predictions_buffer = []
                gemini_sentence = ""
                print("\nCleared all signs and sentence")

            elif key == ord(' '):  # SPACEBAR - form sentence
                if collected_predictions:
                    if USE_GEMINI and gemini_model:
                        print("\n" + "="*70)
                        print("Sending to Gemini for sentence formation...")
                        print("="*70)

                        gemini_sentence = form_sentence_with_gemini(gemini_model, collected_predictions)

                        print("="*70)
                        print(f"FINAL SENTENCE: {gemini_sentence}")
                        print("="*70 + "\n")
                    else:
                        # Simple mode: just concatenate the top predictions
                        gemini_sentence = ' '.join([words[0] for words, _ in collected_predictions])
                        print("\n" + "="*70)
                        print(f"COLLECTED SIGNS: {gemini_sentence}")
                        print("="*70 + "\n")
                else:
                    print("\nNo signs collected yet! Perform some signs first.")

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Detection Mode: {'Advanced (Gemini AI)' if (USE_GEMINI and gemini_model) else 'Simple (No Gemini)'}")
    print(f"Total signs collected: {len(collected_predictions)}")
    if collected_predictions:
        print(f"\nSigns in order:")
        for i, (words, confs) in enumerate(collected_predictions, 1):
            print(f"  {i}. {words[0]} - Top 3: {', '.join([f'{w}({c:.2f})' for w, c in zip(words, confs)])}")

    if gemini_sentence:
        if USE_GEMINI and gemini_model:
            print(f"\nFinal Gemini Sentence: {gemini_sentence}")
        else:
            print(f"\nCollected Signs: {gemini_sentence}")

    print("="*70)
    print("Detection complete!")

    if USE_GEMINI and gemini_model:
        print("\nYour ASL signs have been translated into fluent English using Gemini AI!")
    else:
        print("\nYour ASL signs have been detected and recorded!")
        print("\n💡 TIP: To enable Gemini AI, set USE_GEMINI = True at the top of this script")
