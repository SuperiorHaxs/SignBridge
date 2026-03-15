#!/usr/bin/env python3
"""
Inference API Client — helper for applications to call the inference service.

Usage:
    from inference_api.client import InferenceClient

    client = InferenceClient()  # defaults to http://localhost:3006
    result = client.predict("/path/to/pose.pkl", domain="healthcare")
    print(result['gloss'], result['confidence'])
"""

import os
import requests
from pathlib import Path
from typing import Optional, List


class InferenceClient:
    """Client for the ASL Inference API service."""

    def __init__(self, base_url: str = None):
        self.base_url = (
            base_url
            or os.environ.get('INFERENCE_API_URL')
            or 'http://localhost:3006'
        )

    def health(self) -> dict:
        """Check if the inference API is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            return {'status': 'unreachable', 'url': self.base_url}

    def is_available(self) -> bool:
        """Quick check if the API is reachable."""
        return self.health().get('status') == 'ok'

    def get_domains(self) -> dict:
        """Get available domains and their vocabulary info."""
        resp = requests.get(f"{self.base_url}/domains", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def predict(self, pickle_path: str, domain: str = "generic") -> dict:
        """
        Predict sign from a local pose pickle file.

        Args:
            pickle_path: Path to pose pickle file
            domain: Model domain to use

        Returns:
            dict: {gloss, confidence, top_k_predictions, domain}
        """
        resp = requests.post(
            f"{self.base_url}/predict",
            json={"pickle_path": pickle_path, "domain": domain},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def predict_file(self, file_path: str, domain: str = "generic") -> dict:
        """
        Predict sign by uploading a pose pickle file.
        Use this when the API server can't access the local filesystem.

        Args:
            file_path: Path to pose pickle file to upload
            domain: Model domain to use

        Returns:
            dict: {gloss, confidence, top_k_predictions, domain}
        """
        with open(file_path, 'rb') as f:
            resp = requests.post(
                f"{self.base_url}/predict",
                files={"file": (Path(file_path).name, f)},
                data={"domain": domain},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, pickle_paths: List[str], domain: str = "generic") -> dict:
        """
        Predict signs from multiple pose pickle files.

        Args:
            pickle_paths: List of paths to pose pickle files
            domain: Model domain to use

        Returns:
            dict: {domain, predictions: [{gloss, confidence, top_k_predictions}, ...]}
        """
        resp = requests.post(
            f"{self.base_url}/predict/batch",
            json={"pickle_paths": pickle_paths, "domain": domain},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
