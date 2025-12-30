# ASL Show-and-Tell Demo Application

A step-by-step interactive web application demonstrating the complete ASL-to-English translation pipeline for science fair presentation.

## Overview

This application demonstrates the full pipeline:
1. **Capture**: Record ASL signing via webcam
2. **Convert**: Extract pose keypoints using MediaPipe
3. **Segment**: Identify individual signs using motion-based segmentation
4. **Predict**: Classify each sign using the OpenHands model
5. **Construct**: Generate grammatical English using LLM (Google Gemini)
6. **Evaluate**: Compare raw vs. LLM output using BLEU, BERT, and Quality metrics

## Prerequisites

- Python 3.11+ with the project virtual environment activated
- All project dependencies installed (pose-format, mediapipe, torch, etc.)
- Google AI Studio API key configured in `.env` file

## Setup

1. Activate the project virtual environment:
   ```bash
   cd C:\Users\ashwi\Projects\WLASL-proj\asl-v1
   .\venv\Scripts\activate
   ```

2. Install Flask (if not already installed):
   ```bash
   pip install flask
   ```

3. Ensure LLM is configured:
   - Check `project-utilities/llm_interface/.env` has `GOOGLE_API_KEY` set
   - Or set environment variable: `set GOOGLE_API_KEY=your_key_here`

## Running the Application

```bash
cd applications/show-and-tell
python app.py
```

Then open your browser to: **http://localhost:5000**

## Usage

### Screen 1: Capture
1. Allow camera access when prompted
2. Click "Start Recording" and sign a sentence in ASL
3. Click "Stop Recording" when finished
4. Click "Convert to Pose" to extract keypoints
5. Watch the pose visualization on the right
6. Click "Segment Signs" to proceed

**Tip**: Pause briefly between each sign for better segmentation.

### Screen 2: Segments
- View each detected sign as a separate card
- See Top-1 prediction with confidence score
- See Top-3 alternative predictions
- Click "Construct Sentence" to proceed

### Screen 3: Results
- View raw gloss concatenation vs. LLM-constructed sentence
- Enter a reference sentence (what you intended to sign)
- Click "Calculate Scores" to see evaluation metrics
- Compare BLEU, BERT, Quality, and Composite scores

## Demo Mode

If webcam isn't available or for reliable demos:
- Click "Use Demo Video" on the Capture screen
- Uses pre-recorded sample video (if available in `static/demo/`)

## Architecture

This application is a **thin Flask wrapper** around existing project libraries:

| Component | Source |
|-----------|--------|
| Pose Extraction | `video_to_pose.exe` (pose-format library) |
| Pose Visualization | `visualize_pose.exe` (pose-format library) |
| Motion Segmentation | `motion_based_segmenter.py` |
| Model Inference | `openhands_modernized_inference.py` |
| LLM Interface | `llm_factory.py` (Google AI Studio) |
| LLM Prompt | `project-utilities/llm_interface/prompts/llm_prompt_topk.txt` |
| Evaluation | `evaluate_synthetic_dataset.py` |

**No business logic is duplicated** - all calls go to existing libraries.

## File Structure

```
show-and-tell/
├── app.py                 # Flask application (thin wrapper)
├── requirements.txt       # Dependencies
├── README.md              # This file
├── static/
│   ├── css/styles.css     # Professional styling
│   ├── js/
│   │   ├── capture.js     # Webcam recording
│   │   ├── segments.js    # Segment display
│   │   └── results.js     # Results & metrics
│   └── demo/              # Pre-recorded demo files
├── templates/
│   ├── base.html          # Base template
│   ├── capture.html       # Screen 1
│   ├── segments.html      # Screen 2
│   └── results.html       # Screen 3
└── temp/                  # Temporary files (auto-created)
```

## Troubleshooting

### Webcam not working
- Ensure browser has camera permissions
- Try a different browser (Chrome recommended)
- Use "Demo Mode" as fallback

### Conversion fails
- Check that `video_to_pose.exe` exists in venv/Scripts
- Verify MediaPipe is installed correctly

### Segmentation finds no signs
- Sign more clearly with distinct pauses between words
- Check video isn't too short or too long

### LLM construction fails
- Verify `GOOGLE_API_KEY` is set in environment
- Check internet connection
- See `project-utilities/llm_interface/.env` configuration

### Metrics not calculating
- BERTScore requires `bert-score` package
- Quality scoring requires `transformers` and `torch`
- These should already be installed in the project venv

## For Science Fair

- Use full-screen browser mode (F11) for presentations
- Pre-test with demo mode before live demos
- Have backup screenshots ready
- Print styles are included for posters (Ctrl+P)
