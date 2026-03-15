#!/usr/bin/env python3
"""
ASL Inference API — lightweight standalone service for sign language prediction.

Provides domain-aware model inference that any application can call.
Models are loaded lazily and cached per domain.

Usage:
    python api.py                          # Start on default port 3006
    python api.py --port 3007              # Custom port
    INFERENCE_API_PORT=3007 python api.py  # Via env var

Endpoints:
    POST /predict         — Predict sign from pose pickle file
    GET  /domains         — List available domains and their glosses
    GET  /health          — Health check
"""

import os
import sys
import json
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "openhands-modernized" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "openhands-modernized" / "src" / "util"))

from model_registry import ModelRegistry
from openhands_modernized_inference import predict_pose_file

app = Flask(__name__)
CORS(app)

# Initialize model registry
registry = ModelRegistry()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'asl-inference-api',
        'domains': list(registry._registry.keys()),
    })


@app.route('/domains', methods=['GET'])
def list_domains():
    """List all available domains with their vocabulary info."""
    return jsonify(registry.get_domains())


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sign from a pose pickle file.

    Accepts:
        - multipart/form-data with 'file' (pickle file) and optional 'domain' field
        - JSON with 'pickle_path' (local path) and optional 'domain' field

    Returns:
        JSON: {domain, gloss, confidence, top_k_predictions: [{gloss, confidence}, ...]}
    """
    domain = "generic"

    # Handle multipart file upload
    if 'file' in request.files:
        domain = request.form.get('domain', 'generic')
        file = request.files['file']

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            file.save(tmp.name)
            pickle_path = tmp.name

        try:
            model, id_to_gloss, masked_class_ids = registry.get_model(domain)
            result = predict_pose_file(
                pickle_path, model=model, tokenizer=id_to_gloss,
                masked_class_ids=masked_class_ids
            )
            result['domain'] = domain
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        finally:
            os.unlink(pickle_path)

    # Handle JSON with local path
    elif request.is_json:
        data = request.get_json()
        pickle_path = data.get('pickle_path')
        domain = data.get('domain', 'generic')

        if not pickle_path:
            return jsonify({'error': 'Missing pickle_path'}), 400
        if not Path(pickle_path).exists():
            return jsonify({'error': f'File not found: {pickle_path}'}), 404

        try:
            model, id_to_gloss, masked_class_ids = registry.get_model(domain)
            result = predict_pose_file(
                pickle_path, model=model, tokenizer=id_to_gloss,
                masked_class_ids=masked_class_ids
            )
            result['domain'] = domain
            return jsonify(result)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    else:
        return jsonify({
            'error': 'Send either multipart/form-data with "file" field, '
                     'or JSON with "pickle_path" field'
        }), 400


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict signs from multiple pose pickle files.

    Accepts JSON:
        {
            "pickle_paths": ["/path/to/file1.pkl", "/path/to/file2.pkl"],
            "domain": "healthcare"
        }

    Returns:
        JSON: {domain, predictions: [{gloss, confidence, top_k_predictions}, ...]}
    """
    if not request.is_json:
        return jsonify({'error': 'Expected JSON body'}), 400

    data = request.get_json()
    pickle_paths = data.get('pickle_paths', [])
    domain = data.get('domain', 'generic')

    if not pickle_paths:
        return jsonify({'error': 'Missing pickle_paths'}), 400

    try:
        model, id_to_gloss, masked_class_ids = registry.get_model(domain)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({'error': str(e)}), 400

    predictions = []
    for path in pickle_paths:
        if not Path(path).exists():
            predictions.append({'error': f'File not found: {path}'})
            continue
        try:
            result = predict_pose_file(
                path, model=model, tokenizer=id_to_gloss,
                masked_class_ids=masked_class_ids
            )
            predictions.append(result)
        except Exception as e:
            predictions.append({'error': str(e)})

    return jsonify({
        'domain': domain,
        'predictions': predictions,
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASL Inference API Server")
    parser.add_argument('--port', type=int,
                       default=int(os.environ.get('INFERENCE_API_PORT', 3006)),
                       help='Port to run on (default: 3006)')
    parser.add_argument('--host', type=str,
                       default=os.environ.get('INFERENCE_API_HOST', '0.0.0.0'),
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--preload', nargs='*',
                       help='Pre-load models for these domains at startup')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    args = parser.parse_args()

    if args.preload is not None:
        domains = args.preload if args.preload else None  # empty list = all
        registry.preload(domains)

    print(f"ASL Inference API starting on {args.host}:{args.port}")
    print(f"Registered domains: {list(registry._registry.keys())}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
