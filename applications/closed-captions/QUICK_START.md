# Quick Start Guide - ASL Translation Microservices

## The Problem We Solved

**Original Issue**: When the frontend sent glosses to the backend via HTTP, the **camera would freeze** while waiting for Gemini API response (which can take 2-5 seconds).

**Solution**: File-based asynchronous communication - the camera **NEVER freezes**!

## How It Works Now

1. **Frontend detects a sign** → Writes to `detected_glosses.txt` in background thread → **Camera keeps running!**
2. **Backend watches file** every 0.5 seconds → When 3+ glosses found → Calls Gemini API
3. **Backend writes sentence** to `translated_sentence.txt`
4. **Frontend reads sentence** from file every frame → Displays on screen
5. **Cooldown period** (2 seconds) ensures you can see each intermediate sentence before reconstruction

## Running the Services

### Terminal 1: Start Backend
```bash
cd c:\Users\kavsk\asl\asl-v1\applications
python backend_llm_service.py
```

**Expected Output:**
```
======================================================================
BACKEND LLM SERVICE (Port 4000)
======================================================================
Gemini API Key: Configured
Gloss File: detected_glosses.txt
Sentence File: translated_sentence.txt
Auto-translate after: 3 glosses
File check interval: 0.5s
Min delay between translations: 2.0s
======================================================================

[BACKEND] File watcher thread started
[BACKEND] File watcher started - monitoring glosses...

 * Running on http://127.0.0.1:4000
```

### Terminal 2: Start Frontend
```bash
cd c:\Users\kavsk\asl\asl-v1\applications
python frontend_cv_service.py --checkpoint "..\models\training-scripts\models\wlasl_50_class_model"
```

**Expected Output:**
```
======================================================================
FRONTEND COMPUTER VISION SERVICE (Port 3000)
======================================================================
Model: Loaded
Backend URL: http://localhost:4000
======================================================================

[CV] Initialized gloss file: detected_glosses.txt
[CV] Loading model from: ..\models\training-scripts\models\wlasl_50_class_model
[CV] Model loaded successfully
[CV] Starting webcam capture from camera 0
[CV] Trying backend: 700
[CV] Successfully opened camera with backend 700
[CV] Webcam capture started successfully
[CV] Ready. Press 'q' to quit
```

## What You Should See

### When You Sign:

**Frontend Terminal:**
```
[CV] Sign started
[CV] Sign completed (42 frames)
[CV] PREDICTED: 'HELLO' (85.2%)
[CV] TOP-3:
  1. HELLO (85.2%)
  2. HI (10.3%)
  3. WAVE (4.5%)
[CV] ✓ Wrote gloss to file: HELLO
```

**Camera keeps running smoothly - NO FREEZE!**

### After 3 Signs:

**Backend Terminal:**
```
[BACKEND] ⚡ Auto-translate triggered! (3 new glosses)
[BACKEND] Processing 3 new glosses:
  1. HELLO (85.2%)
  2. MY (92.1%)
  3. NAME (78.3%)
[BACKEND] Extending caption (incremental mode)
[BACKEND] ✓ Constructed: 'Hello, my name is...'
[BACKEND] ✓ Wrote sentence to file
```

**Frontend Terminal:**
```
[CV] ✓ New sentence received: 'Hello, my name is...'
```

**On Screen:** You see the sentence displayed at the bottom of the video!

## Files Created

While running, you'll see these files in the `applications` folder:

- `detected_glosses.txt` - All glosses detected (one per line, JSON format)
- `translated_sentence.txt` - Current translated sentence

These files persist between runs. To reset:
```bash
curl -X POST http://localhost:4000/reset_session
```

## Advantages of This Approach

✅ **Camera never freezes** - All I/O is asynchronous
✅ **Simple architecture** - Just file reads/writes
✅ **No HTTP timeouts** - No network calls during detection
✅ **Persistent state** - Can see accumulated glosses in file
✅ **Easy debugging** - Just open the .txt files!
✅ **Scalable** - Backend can process multiple sessions with different files

## Troubleshooting

### Camera freezing?
**This should NEVER happen now!** If it does:
1. Check if frontend is actually writing to file: `type detected_glosses.txt`
2. Make sure gloss writing happens in background thread (check code)

### Sentences not appearing?
1. Check backend terminal - is file watcher running?
2. Check if `translated_sentence.txt` exists and has content
3. Verify backend Gemini API key is set

### Backend not detecting new glosses?
1. Check `detected_glosses.txt` - are new lines being added?
2. Backend checks every 0.5s - wait at least 1 second after 3rd sign
3. Check backend terminal for file watcher errors

## Your Checkpoint Path

```
c:\Users\kavsk\asl\asl-v1\models\training-scripts\models\wlasl_50_class_model
```

## Quick Commands

**Start both services** (use two terminal windows):
```bash
# Terminal 1
python backend_llm_service.py

# Terminal 2
python frontend_cv_service.py --checkpoint "..\models\training-scripts\models\wlasl_50_class_model"
```

**Test camera**:
```bash
python test_camera.py
```

**Reset session**:
```bash
curl -X POST http://localhost:4000/reset_session
```

**View accumulated glosses**:
```bash
type detected_glosses.txt
```

**View current sentence**:
```bash
type translated_sentence.txt
```

Enjoy your freeze-free ASL translation! 🎉
