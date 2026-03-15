#!/usr/bin/env python3
"""
ASL Video Captioning — Web App

Simple Flask UI: upload a signing video or pick a demo sample,
get back the same video with burned-in English captions.

Usage:
    python app.py
    Then open http://localhost:5001
"""

import os
import sys
import uuid
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

# Import the captioning pipeline
sys.path.insert(0, str(Path(__file__).parent))
from caption_video import caption_video, caption_demo_sample, list_demo_samples

app = Flask(__name__)

# Store jobs: job_id -> {status, message, output_path, error, ...}
jobs = {}
jobs_lock = threading.Lock()

UPLOAD_DIR = Path(__file__).parent / "uploads"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/demo-samples")
def get_demo_samples():
    """List available demo samples from show-and-tell."""
    return jsonify({"samples": list_demo_samples()})


@app.route("/upload", methods=["POST"])
def upload():
    """Handle video upload and start captioning in background."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save upload
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_captioned.mp4"
    file.save(str(input_path))

    # Init job
    with jobs_lock:
        jobs[job_id] = {
            "status": "processing",
            "message": "Starting pipeline...",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "original_name": file.filename,
            "mode": "upload",
            "error": None,
        }

    t = threading.Thread(target=_run_upload_pipeline, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/caption-demo", methods=["POST"])
def caption_demo():
    """Caption a demo sample using precomputed predictions."""
    data = request.get_json()
    sample_path = data.get("sample_path")
    if not sample_path or not os.path.isdir(sample_path):
        return jsonify({"error": "Invalid sample path"}), 400

    job_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{job_id}_captioned.mp4"

    with jobs_lock:
        jobs[job_id] = {
            "status": "processing",
            "message": "Starting demo captioning...",
            "sample_path": sample_path,
            "output_path": str(output_path),
            "mode": "demo",
            "error": None,
        }

    t = threading.Thread(target=_run_demo_pipeline, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


def _run_upload_pipeline(job_id):
    """Run full pipeline on uploaded video."""
    job = jobs[job_id]
    try:
        _update_job(job_id, "processing", "Running captioning pipeline...")
        caption_video(job["input_path"], job["output_path"])
        _update_job(job_id, "done", "Captioning complete!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job(job_id, "error", str(e), error=str(e))


def _run_demo_pipeline(job_id):
    """Run demo sample captioning with precomputed data."""
    job = jobs[job_id]
    try:
        _update_job(job_id, "processing", "Captioning demo sample...")
        caption_demo_sample(job["sample_path"], job["output_path"])
        _update_job(job_id, "done", "Captioning complete!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_job(job_id, "error", str(e), error=str(e))


def _update_job(job_id, status, message, error=None):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["message"] = message
            if error:
                jobs[job_id]["error"] = error


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "message": job["message"],
        "error": job["error"],
    })


@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Not ready yet"}), 400

    output_path = job["output_path"]
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file missing"}), 500

    name = job.get("original_name", "video")
    download_name = f"{Path(name).stem}_captioned.mp4"
    return send_file(output_path, as_attachment=True, download_name=download_name)


@app.route("/preview/<job_id>")
def preview(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 400
    return send_file(job["output_path"], mimetype="video/mp4")


if __name__ == "__main__":
    print("=" * 50)
    print("  ASL Video Captioning")
    print("  Open http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False)
