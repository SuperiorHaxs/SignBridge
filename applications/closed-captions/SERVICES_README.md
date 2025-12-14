# ASL Sign Language Translation - Microservices Architecture

This is a microservices-based version of the ASL sign language translation system, split into two separate services:

1. **Frontend Computer Vision Service (Port 3000)** - Handles webcam capture, pose estimation, and sign detection
2. **Backend LLM Service (Port 4000)** - Handles Gemini API calls for sentence construction

## Architecture

**File-Based Non-Blocking Communication** (Camera NEVER freezes!)

```
┌─────────────────────────────────┐
│  Frontend CV Service (Port 3000)│
│  - Webcam capture               │
│  - Pose estimation (MediaPipe)  │
│  - Sign detection               │
│  - Display interface            │
└───────────┬─────────────────────┘
            │
            │ Writes glosses to file ASYNC
            │ (detected_glosses.txt)
            │ NEVER BLOCKS CAMERA!
            ↓
┌─────────────────────────────────┐
│  detected_glosses.txt           │
│  {"gloss": "HELLO", "conf": 0.8}│
│  {"gloss": "MY", "conf": 0.9}   │
│  {"gloss": "NAME", "conf": 0.7} │
└───────────┬─────────────────────┘
            │
            │ Backend watches file
            │ (every 0.5s)
            ↓
┌─────────────────────────────────┐
│  Backend LLM Service (Port 4000)│
│  - File watcher (background)    │
│  - Auto-translate at 3 words    │
│  - Gemini API integration       │
│  - Context management           │
└───────────┬─────────────────────┘
            │
            │ Writes sentence to file
            │ (translated_sentence.txt)
            ↓
┌─────────────────────────────────┐
│  translated_sentence.txt        │
│  "Hello, my name is..."         │
└───────────┬─────────────────────┘
            │
            │ Frontend reads file
            │ (every frame)
            ↓
┌─────────────────────────────────┐
│  Frontend CV Service            │
│  - Displays sentence on screen  │
│  - Camera keeps running smooth! │
└─────────────────────────────────┘
```

## Prerequisites

### Required Python Packages
```bash
pip install flask
pip install requests
pip install opencv-python
pip install mediapipe
pip install numpy
pip install google-generativeai
```

### Environment Variables
Set your Gemini API key:
```bash
# Windows
set GEMINI_API_KEY=your-api-key-here

# Linux/Mac
export GEMINI_API_KEY=your-api-key-here
```

## Running the Services

### Step 1: Start Backend LLM Service (Port 4000)

Open a terminal and run:
```bash
cd applications
python backend_llm_service.py
```

You should see:
```
======================================================================
BACKEND LLM SERVICE (Port 4000)
======================================================================
Gemini API Key: Configured
Frontend URL: http://localhost:3000
======================================================================
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:4000
```

### Step 2: Start Frontend CV Service (Port 3000)

Open a **second terminal** and run:
```bash
cd applications
python frontend_cv_service.py --checkpoint "path/to/your/checkpoint"

# Or without checkpoint (uses default model):
python frontend_cv_service.py
```

You should see:
```
======================================================================
FRONTEND COMPUTER VISION SERVICE (Port 3000)
======================================================================
Model: Loaded
Backend URL: http://localhost:4000
======================================================================

INSTRUCTIONS:
  - Perform signs clearly in front of camera
  - Pause 0.5-1s between signs
  - AUTO-TRANSLATE: Every 3 words
  - Press SPACEBAR to force translate
  - Press 'q' or ESC to quit
======================================================================
```

## How It Works

### Sign Detection Flow (Non-Blocking Architecture)

1. **Frontend captures video** from webcam (continuous, never stops)
2. **MediaPipe extracts pose** from each frame
3. **Motion detection** identifies when signing starts and stops
4. **Model predicts gloss** for each detected sign
5. **Gloss written to file IMMEDIATELY** in background thread (camera keeps running!)
6. **Backend file watcher** checks file every 0.5 seconds
7. **Auto-translate triggers** when 3+ new glosses detected in file
8. **Backend calls Gemini API** to construct grammatically correct sentence
9. **Backend writes sentence to file** (translated_sentence.txt)
10. **Frontend reads sentence** from file every frame and displays it

**KEY ADVANTAGE**: Camera NEVER freezes because all I/O is asynchronous via files!

### Communication Files

#### detected_glosses.txt (Written by Frontend)
Each line is a JSON object representing one detected gloss:
```json
{"gloss": "HELLO", "confidence": 0.85, "top_k_predictions": [...]}
{"gloss": "MY", "confidence": 0.92, "top_k_predictions": [...]}
{"gloss": "NAME", "confidence": 0.78, "top_k_predictions": [...]}
```

#### translated_sentence.txt (Written by Backend, Read by Frontend)
Contains the current translated sentence:
```
Hello, my name is...
```

### API Endpoints (Optional - for manual control)

#### Backend Service (Port 4000)

- `GET /health` - Health check
- `POST /reset_session` - Reset session context and clear files

#### Frontend Service (Port 3000)

- `GET /health` - Health check
- `POST /sentence_update` - Manual sentence update (legacy endpoint)

## Configuration

### Frontend CV Service (`frontend_cv_service.py`)

```python
# File Communication (Non-Blocking)
GLOSS_FILE = "detected_glosses.txt"     # Where glosses are written
SENTENCE_FILE = "translated_sentence.txt" # Where sentences are read from

# Sign Detection
MOTION_THRESHOLD = 2000000              # Motion sensitivity
COOLDOWN_FRAMES = 12                    # Frames between signs
MIN_SIGN_FRAMES = 15                    # Minimum frames per sign
SIGN_DEBOUNCE_TIME = 0.5                # Seconds between signs
MIN_PREDICTION_CONFIDENCE = 0.0         # Confidence threshold
```

### Backend LLM Service (`backend_llm_service.py`)

```python
# File Communication (Non-Blocking)
GLOSS_FILE = "detected_glosses.txt"     # File to watch for new glosses
SENTENCE_FILE = "translated_sentence.txt" # File to write sentences to
CAPTION_BUFFER_SIZE = 3                 # Auto-translate after N glosses
FILE_CHECK_INTERVAL = 0.5               # Seconds between file checks

# Gemini Configuration
CONTEXT_MODE = "full_history"           # Context mode
MAX_CONTEXT_SENTENCES = 2               # Sentences to keep for context
ENABLE_INCREMENTAL_MODE = True          # Continuous caption mode
ENABLE_DYNAMIC_RECONSTRUCTION = True    # Smart reconstruction
RECONSTRUCTION_MODE = "sliding_window"  # Reconstruction strategy
RECONSTRUCTION_WINDOW_SIZE = 15         # Words to reconstruct
```

## Usage Tips

1. **Position yourself** in good lighting with hands visible
2. **Sign clearly** with deliberate movements
3. **Pause briefly** (0.5-1 sec) between each sign
4. **Watch the status indicator**:
   - Green "READY" - System is idle, ready for signing
   - Green "SIGNING" - Motion detected, recording frames
   - Orange "PROCESSING..." - Model is analyzing the sign
5. **Auto-translate** happens every 3 words automatically (backend watches file)
6. **Camera never freezes** - all communication is via files in background
7. **Sentences appear** automatically on screen after 3 signs
8. **Quit** by pressing 'q' or ESC

## Troubleshooting

### Backend service not starting
- Check if port 4000 is already in use
- Verify Gemini API key is set: `echo %GEMINI_API_KEY%` (Windows) or `echo $GEMINI_API_KEY` (Linux/Mac)

### Frontend can't connect to backend
- Make sure backend service is running first
- Check firewall settings for localhost connections
- Verify `BACKEND_URL` in frontend matches backend address

### Sentence not displaying
- Check backend terminal for Gemini API errors
- Verify frontend received callback (check frontend terminal logs)
- Try manual translate with SPACEBAR

### Low prediction accuracy
- Use a trained checkpoint with `--checkpoint` flag
- Improve lighting conditions
- Sign more deliberately and slower
- Increase `MIN_SIGN_FRAMES` to require longer signs

## Development

### Adding New Features

**Frontend changes**: Edit [frontend_cv_service.py](frontend_cv_service.py)
- Modify `SignLanguageApp` class for CV logic
- Update Flask routes for new endpoints

**Backend changes**: Edit [backend_llm_service.py](backend_llm_service.py)
- Modify `construct_sentence_with_gemini()` for prompt changes
- Update Flask routes for new translation logic

### Testing Endpoints

Test backend:
```bash
# Health check
curl http://localhost:4000/health

# Send glosses
curl -X POST http://localhost:4000/translate \
  -H "Content-Type: application/json" \
  -d '{"glosses": [{"gloss": "HELLO", "confidence": 0.9}]}'
```

Test frontend:
```bash
# Health check
curl http://localhost:3000/health

# Send sentence (simulating backend callback)
curl -X POST http://localhost:3000/sentence_update \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Hello, world!"}'
```

## Benefits of Microservices Architecture

1. **Separation of Concerns**: CV and LLM logic are independent
2. **Scalability**: Can run multiple frontend instances with one backend
3. **Maintainability**: Easier to debug and update each service
4. **Flexibility**: Can swap out backend LLM without touching CV code
5. **Deployment**: Can deploy services on different machines/containers

## Future Enhancements

- [ ] Add authentication between services
- [ ] Implement WebSocket for real-time updates (replace polling)
- [ ] Add database for persistent session storage
- [ ] Support multiple simultaneous users/sessions
- [ ] Add metrics and monitoring endpoints
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS, Azure, GCP)

## Original Monolithic Version

The original all-in-one version is still available in [predict_sentence_with_gemini_streaming.py](predict_sentence_with_gemini_streaming.py).
