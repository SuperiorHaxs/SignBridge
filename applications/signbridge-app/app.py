#!/usr/bin/env python3
"""
SignBridge App — Standalone Flask app for doctor-patient ASL communication.

Modes:
    --mode demo   Scripted demo with pre-recorded sign videos (default)
    --mode live   Real-time ASL recognition via inference API + camera

Methods:
    --method speak  User communicates via speech (default)
    --method sign   User communicates via ASL signs
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"
MODELS_DIR = PROJECT_ROOT / "models"

# Add paths for imports
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))
sys.path.insert(0, str(APPLICATIONS_DIR / "show-and-tell"))  # for camera_processor

from llm_factory import create_llm_provider

# ============================================================================
# APP SETUP
# ============================================================================
app = Flask(__name__)

# Mode: "demo" (scripted) or "live" (real-time inference)
APP_MODE = os.environ.get("SIGNBRIDGE_MODE", "demo")
APP_METHOD = os.environ.get("SIGNBRIDGE_METHOD", "speak")

# Data paths
DEMO_DATA_DIR = Path(__file__).parent / "demo-data"
SIGN_BANK_DIR = APPLICATIONS_DIR / "show-and-tell" / "sign-bank"
SAMPLES_DIR = APPLICATIONS_DIR / "show-and-tell" / "demo-data" / "samples"

# Inference API URL (for live mode)
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:3006")

# LLM prompt template for live mode
LLM_PROMPT_PATH = PROJECT_UTILITIES_DIR / "llm_interface" / "prompts" / "llm_prompt_closed_captions.txt"

# Configurable API base URL (for when API is separated)
API_BASE_URL = os.environ.get("SIGNBRIDGE_API_URL", None)

# ============================================================================
# LIVE MODE — Inference API + Camera Processor
# ============================================================================
_camera_processor = None


def _get_camera_processor():
    """Lazy-load camera processor for live mode."""
    global _camera_processor
    if _camera_processor is not None:
        return _camera_processor

    from camera_processor import CameraProcessor

    processor = CameraProcessor()

    # Set predict function to route through inference API
    def predict_via_api(pickle_path, domain="doctor_visit"):
        import requests
        payload = {"pickle_path": str(pickle_path), "domain": domain}
        print(f"[SignBridge] predict_via_api -> {INFERENCE_API_URL}/predict domain={domain}")
        resp = requests.post(
            f"{INFERENCE_API_URL}/predict",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"[SignBridge] predict_via_api result: domain={result.get('domain')}, gloss={result.get('gloss')}")
        return result

    processor.set_predict_fn(predict_via_api)
    _camera_processor = processor
    return processor


def _load_llm_prompt():
    """Load the closed-captions LLM prompt template."""
    if LLM_PROMPT_PATH.exists():
        return LLM_PROMPT_PATH.read_text(encoding='utf-8')
    return None

# ============================================================================
# ROUTES — Pages
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html", mode=APP_MODE, method=APP_METHOD)


# ============================================================================
# ROUTES — API
# ============================================================================

@app.route("/api/conversation")
def get_conversation():
    """Return the conversation script."""
    conv_path = DEMO_DATA_DIR / "conversation.json"
    with open(conv_path) as f:
        return jsonify(json.load(f))


@app.route("/api/vocabulary/<domain>")
def get_vocabulary(domain):
    """Return the list of sign glosses available for a domain."""
    # Read registry to find model directory
    registry_file = MODELS_DIR / "openhands-modernized" / "production-models" / "registry.json"
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"generic": "wlasl_43_class_50s_model"}

    model_dir_name = registry.get(domain)
    if not model_dir_name:
        return jsonify({"error": f"Unknown domain: {domain}"}), 404

    class_file = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir_name / "class_index_mapping.json"
    if not class_file.exists():
        return jsonify({"error": "Class mapping not found"}), 404

    with open(class_file, 'r') as f:
        mapping = json.load(f)

    # Filter out masked classes if mask file exists
    masked_ids = set()
    mask_file = MODELS_DIR / "openhands-modernized" / "production-models" / model_dir_name / "masked_classes.json"
    if mask_file.exists():
        with open(mask_file, 'r') as f:
            masked_ids = set(json.load(f).get("masked_class_ids", []))

    glosses = sorted(v for k, v in mapping.items() if int(k) not in masked_ids)
    return jsonify({"domain": domain, "count": len(glosses), "glosses": glosses})


@app.route("/api/construct-sentence", methods=["POST"])
def construct_sentence():
    """Use LLM to construct a natural sentence from ASL glosses."""
    glosses = []
    try:
        data = request.get_json()
        glosses = data.get("glosses", [])
        conversation_history = data.get("conversation_history", [])

        if not glosses:
            return jsonify({"success": False, "error": "No glosses provided"}), 400

        raw_glosses = " ".join(glosses)

        # Build conversation context
        context_lines = []
        for msg in conversation_history:
            speaker = msg.get("speaker", "unknown").capitalize()
            text = msg.get("text", "")
            context_lines.append(f"{speaker}: {text}")
        context_str = "\n".join(context_lines) if context_lines else "No prior conversation."

        prompt = f"""You are translating ASL glosses into natural English for a Deaf patient communicating with a doctor during a medical visit.

Conversation so far:
{context_str}

The patient just signed these ASL glosses: {raw_glosses}

Convert these glosses into a natural, grammatically correct English sentence that makes sense in the context of this doctor visit conversation.

Rules:
- Use all the glosses provided, in order
- Add appropriate filler words (the, a, is, are, etc.) for natural English
- The sentence should fit naturally as the patient's response to the doctor's last statement
- Keep it simple and conversational
- Output ONLY the English sentence, nothing else

English sentence:"""

        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=100,
            timeout=15,
        )
        response = llm.generate(prompt)

        generated = response.strip()
        if generated.startswith('"') and generated.endswith('"'):
            generated = generated[1:-1]

        print(f"[DoctorDemo LLM] Glosses: {glosses} -> {generated}")

        return jsonify({"success": True, "glosses": glosses, "sentence": generated})

    except Exception as e:
        print(f"[DoctorDemo LLM] Error: {e}")
        return jsonify({"success": False, "error": str(e), "fallback": " ".join(glosses)}), 500


# ============================================================================
# ROUTES — Live Mode API
# ============================================================================

@app.route("/api/process-sign", methods=["POST"])
def process_sign():
    """
    Process a video blob of a single sign (live mode).
    Returns gloss prediction from the inference API.
    """
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video provided"}), 400

    video_file = request.files["video"]
    video_bytes = video_file.read()
    domain = request.form.get("domain", "doctor_visit")
    print(f"[SignBridge] /api/process-sign domain={domain}")

    if len(video_bytes) == 0:
        return jsonify({"success": False, "error": "Empty video"}), 400

    try:
        processor = _get_camera_processor()
        prediction = processor.process_video_bytes(video_bytes, video_format="webm", domain=domain)

        if prediction is None:
            return jsonify({"success": False, "error": "Could not process video"}), 400

        return jsonify({
            "success": True,
            "gloss": prediction["gloss"],
            "confidence": prediction["confidence"],
            "top_k": prediction.get("top_k_predictions", [])[:5],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/construct-sentence-live", methods=["POST"])
def construct_sentence_live():
    """
    Construct a sentence from detected glosses using the closed-captions
    LLM prompt template (live mode).
    """
    try:
        data = request.get_json()
        gloss_predictions = data.get("gloss_predictions", [])
        conversation_history = data.get("conversation_history", [])

        if not gloss_predictions:
            return jsonify({"success": False, "error": "No gloss predictions"}), 400

        # Build context section
        context_lines = []
        for msg in conversation_history:
            speaker = msg.get("speaker", "unknown").capitalize()
            text = msg.get("text", "")
            context_lines.append(f"{speaker}: {text}")
        context_str = "\n".join(context_lines) if context_lines else "No prior conversation."

        # Build gloss details with top-k predictions
        gloss_details_lines = []
        for i, pred in enumerate(gloss_predictions):
            gloss_details_lines.append(f"Position {i + 1}:")
            top_k = pred.get("top_k", [])
            if top_k:
                for j, option in enumerate(top_k[:3]):
                    conf = option.get("confidence", 0) * 100
                    gloss_details_lines.append(
                        f"  Option {j + 1}: '{option['gloss']}' (confidence: {conf:.1f}%)"
                    )
            else:
                conf = pred.get("confidence", 0) * 100
                gloss_details_lines.append(
                    f"  Option 1: '{pred['gloss']}' (confidence: {conf:.1f}%)"
                )
        gloss_details = "\n".join(gloss_details_lines)
        print(f"[Live LLM] Input to LLM:\n{gloss_details}")

        # Try loading the prompt template
        prompt_template = _load_llm_prompt()
        if prompt_template:
            context_section = f"Conversation context:\n{context_str}"
            prompt = prompt_template.replace("{context_section}", context_section)
            prompt = prompt.replace("{gloss_details}", gloss_details)
        else:
            # Fallback to inline prompt
            raw_glosses = " ".join(p["gloss"] for p in gloss_predictions)
            prompt = f"""You are translating ASL glosses into natural English for a Deaf patient communicating with a doctor.

Conversation so far:
{context_str}

The patient just signed these ASL glosses: {raw_glosses}

Convert these glosses into a natural, grammatically correct English sentence.
Output ONLY the English sentence, nothing else.

English sentence:"""

        llm = create_llm_provider(
            provider="googleaistudio",
            model_name="gemini-2.0-flash",
            max_tokens=150,
            timeout=15,
        )
        response = llm.generate(prompt)

        # Parse response — prompt template expects JSON with "sentence" key
        generated = response.strip()
        # Strip markdown code fences if present (```json ... ```)
        if generated.startswith('```'):
            lines = generated.split('\n')
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith('```')]
            generated = '\n'.join(lines).strip()

        sentence = generated  # fallback
        try:
            parsed = json.loads(generated)
            sentence = parsed.get("sentence", generated)
        except (json.JSONDecodeError, TypeError):
            # Plain text response
            if sentence.startswith('"') and sentence.endswith('"'):
                sentence = sentence[1:-1]

        print(f"[Live LLM] Glosses: {[p['gloss'] for p in gloss_predictions]} -> {sentence}")

        return jsonify({"success": True, "sentence": sentence})

    except Exception as e:
        import traceback
        print(f"[Live LLM] ERROR: {e}")
        traceback.print_exc()
        glosses = [p.get("gloss", "") for p in data.get("gloss_predictions", [])]
        return jsonify({"success": False, "error": str(e), "fallback": " ".join(glosses)}), 500


# ============================================================================
# ROUTES — Static file serving
# ============================================================================

@app.route("/cert")
def download_cert():
    """Download the SSL certificate for mobile trust."""
    cert_path = Path(__file__).parent / "cert.pem"
    if cert_path.exists():
        return send_from_directory(Path(__file__).parent, "cert.pem",
                                  mimetype="application/x-pem-file",
                                  as_attachment=True,
                                  download_name="signbridge.pem")
    return "No certificate found", 404


@app.route("/sign-bank/<path:filename>")
def serve_sign_bank(filename):
    """Serve sign-bank video files."""
    return send_from_directory(SIGN_BANK_DIR, filename)


@app.route("/samples/<sample_id>/original_video.mp4")
def serve_sample_video(sample_id):
    """Serve full-sentence original video from demo-data samples."""
    sample_dir = SAMPLES_DIR / sample_id
    video_path = sample_dir / "original_video.mp4"
    if video_path.exists():
        return send_from_directory(str(sample_dir), "original_video.mp4")
    print(f"[Samples] Not found: {video_path}")
    return "Sample video not found", 404


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignBridge App")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo",
                        help="demo = scripted sign videos, live = real-time inference via API")
    parser.add_argument("--method", choices=["sign", "speak"], default="speak",
                        help="sign = user communicates in ASL, speak = user communicates via speech")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    APP_MODE = args.mode
    APP_METHOD = args.method

    print(f"\n  SignBridge App")
    print(f"  Mode:      {APP_MODE}")
    print(f"  Method:    {APP_METHOD}")
    print(f"  Sign bank: {SIGN_BANK_DIR}")
    if APP_MODE == "live":
        print(f"  Inference: {INFERENCE_API_URL}")
        print(f"  LLM prompt: {LLM_PROMPT_PATH}")
    print(f"  API base:  {API_BASE_URL or 'local'}\n")

    import ssl
    cert_dir = Path(__file__).parent
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_file), str(key_file))
        print("  Running with HTTPS (speech recognition enabled)\n")
        app.run(host="0.0.0.0", port=args.port, debug=True, ssl_context=context)
    else:
        print("  WARNING: No SSL certs found — speech recognition won't work on mobile\n")
        app.run(host="0.0.0.0", port=args.port, debug=True)
