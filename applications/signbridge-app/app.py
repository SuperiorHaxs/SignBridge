#!/usr/bin/env python3
"""
Doctor Demo — Standalone Flask app
Simulates a doctor-patient conversation where the doctor speaks English
and a Deaf patient responds in ASL, translated by SignBridge.
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
APPLICATIONS_DIR = PROJECT_ROOT / "applications"
PROJECT_UTILITIES_DIR = PROJECT_ROOT / "project-utilities"

# Add paths for LLM import
sys.path.insert(0, str(PROJECT_UTILITIES_DIR))
sys.path.insert(0, str(PROJECT_UTILITIES_DIR / "llm_interface"))

from llm_factory import create_llm_provider

# ============================================================================
# APP SETUP
# ============================================================================
app = Flask(__name__)

# Data paths
DEMO_DATA_DIR = Path(__file__).parent / "demo-data"
SIGN_BANK_DIR = APPLICATIONS_DIR / "show-and-tell" / "sign-bank"

# Configurable API base URL (for when API is separated)
# Set env var SIGNBRIDGE_API_URL to point to the separate API
API_BASE_URL = os.environ.get("SIGNBRIDGE_API_URL", None)

# ============================================================================
# ROUTES — Pages
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================================
# ROUTES — API
# ============================================================================

@app.route("/api/conversation")
def get_conversation():
    """Return the conversation script."""
    conv_path = DEMO_DATA_DIR / "conversation.json"
    with open(conv_path) as f:
        return jsonify(json.load(f))


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


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    print(f"\n  Doctor Demo")
    print(f"  Sign bank: {SIGN_BANK_DIR}")
    print(f"  API base:  {API_BASE_URL or 'local'}\n")
    import ssl
    cert_dir = Path(__file__).parent
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if cert_file.exists() and key_file.exists():
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_file), str(key_file))
        print("  Running with HTTPS (speech recognition enabled)\n")
        app.run(host="0.0.0.0", port=5001, debug=True, ssl_context=context)
    else:
        print("  WARNING: No SSL certs found — speech recognition won't work on mobile\n")
        app.run(host="0.0.0.0", port=5001, debug=True)
