#!/usr/bin/env python3
"""
Model Registry — loads and caches domain-specific ASL models.

Each domain (generic, healthcare, education, etc.) maps to a model directory
containing config.json, class_index_mapping.json, and pytorch_model.bin.
Models are loaded lazily on first request and cached in memory.
"""

import os
import sys
import json
import threading
from pathlib import Path

# Add model source paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src"))
sys.path.insert(0, str(MODELS_DIR / "openhands-modernized" / "src" / "util"))

from openhands_modernized_inference import load_model_from_checkpoint

# Default production models directory
PRODUCTION_MODELS_DIR = MODELS_DIR / "openhands-modernized" / "production-models"

# Domain-to-model directory mapping
# Can be overridden via DOMAIN_REGISTRY_PATH env var pointing to a JSON file
DEFAULT_REGISTRY = {
    "generic": "wlasl_43_class_50s_model",
}


class ModelRegistry:
    """Thread-safe registry for loading and caching domain-specific models."""

    def __init__(self, models_dir: Path = None, registry: dict = None):
        self._models_dir = models_dir or PRODUCTION_MODELS_DIR
        self._registry = registry or self._load_registry()
        self._cache = {}  # domain -> (model, id_to_gloss, masked_class_ids)
        self._lock = threading.Lock()

        # Log registry on startup for debugging
        print(f"[ModelRegistry] Models dir: {self._models_dir}")
        print(f"[ModelRegistry] Domain registry:")
        for domain, model_dir in self._registry.items():
            model_path = self._models_dir / model_dir
            exists = model_path.exists()
            # Read class count if available
            class_file = model_path / "class_index_mapping.json"
            num_classes = "?"
            if class_file.exists():
                with open(class_file, 'r') as f:
                    num_classes = len(json.load(f))
            print(f"  {domain:>12s} -> {model_dir} ({num_classes} classes, exists={exists})")

    def _load_registry(self) -> dict:
        """Load domain registry from JSON file or use defaults."""
        registry_path = os.environ.get("DOMAIN_REGISTRY_PATH")
        if registry_path and Path(registry_path).exists():
            print(f"[ModelRegistry] Registry source: env DOMAIN_REGISTRY_PATH={registry_path}")
            with open(registry_path, 'r') as f:
                return json.load(f)

        # Check for registry.json in models dir
        local_registry = self._models_dir / "registry.json"
        if local_registry.exists():
            print(f"[ModelRegistry] Registry source: {local_registry}")
            with open(local_registry, 'r') as f:
                return json.load(f)

        print(f"[ModelRegistry] Registry source: DEFAULT_REGISTRY (no registry.json found)")
        return DEFAULT_REGISTRY.copy()

    def register_domain(self, domain: str, model_dir_name: str):
        """Register a new domain -> model mapping."""
        self._registry[domain] = model_dir_name

    def get_model(self, domain: str = "generic"):
        """
        Get model for a domain. Loads from disk on first call, cached thereafter.

        Args:
            domain: Domain name (e.g., "generic", "healthcare")

        Returns:
            tuple: (model, id_to_gloss, masked_class_ids)

        Raises:
            ValueError: If domain is not registered
            FileNotFoundError: If model directory doesn't exist
        """
        with self._lock:
            if domain in self._cache:
                return self._cache[domain]

        # Load outside the lock (IO-bound)
        if domain not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown domain '{domain}'. Available: {available}")

        model_dir_name = self._registry[domain]
        model_path = self._models_dir / model_dir_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found for domain '{domain}': {model_path}"
            )

        print(f"[ModelRegistry] Loading model for domain '{domain}' from {model_path}")
        model, id_to_gloss, masked_class_ids = load_model_from_checkpoint(str(model_path))

        with self._lock:
            self._cache[domain] = (model, id_to_gloss, masked_class_ids)

        return model, id_to_gloss, masked_class_ids

    def get_domains(self) -> dict:
        """
        Get all registered domains with their vocabulary info.

        Returns:
            dict: {domain: {model_dir, num_classes, glosses, loaded}}
        """
        result = {}
        for domain, model_dir_name in self._registry.items():
            model_path = self._models_dir / model_dir_name
            info = {
                "model_dir": model_dir_name,
                "exists": model_path.exists(),
                "loaded": domain in self._cache,
            }

            # Read class mapping if available
            class_mapping_file = model_path / "class_index_mapping.json"
            if class_mapping_file.exists():
                with open(class_mapping_file, 'r') as f:
                    mapping = json.load(f)
                info["num_classes"] = len(mapping)
                info["glosses"] = sorted(mapping.values())
            else:
                info["num_classes"] = None
                info["glosses"] = []

            result[domain] = info

        return result

    def unload_domain(self, domain: str):
        """Unload a cached model to free memory."""
        with self._lock:
            if domain in self._cache:
                del self._cache[domain]
                print(f"[ModelRegistry] Unloaded model for domain '{domain}'")

    def preload(self, domains: list = None):
        """Pre-load models for specified domains (or all registered)."""
        domains = domains or list(self._registry.keys())
        for domain in domains:
            try:
                self.get_model(domain)
            except Exception as e:
                print(f"[ModelRegistry] Warning: Failed to preload '{domain}': {e}")
