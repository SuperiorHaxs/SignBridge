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

from openhands_modernized_inference import load_model_from_checkpoint, predict_dual_model

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
        # Optional per-domain version map, {domain: {version: model_dir}} + active
        # version, parsed from registry.json's domains.<d>.versions. Empty for
        # domains that haven't been versioned by the retrain utility.
        self._versions, self._active_version = self._load_versions()
        self._cache = {}  # cache key (domain or "domain@version") -> (model, id_to_gloss, masked_class_ids)
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
        """Load domain registry from JSON file or use defaults.

        Supports both the new format ({"domains": {...}, "fallback_model": ...})
        and the legacy flat format ({"domain": "model_dir"}).
        Returns a flat dict of {domain: model_dir_name} for internal use.
        """
        raw = None
        registry_path = os.environ.get("DOMAIN_REGISTRY_PATH")
        if registry_path and Path(registry_path).exists():
            print(f"[ModelRegistry] Registry source: env DOMAIN_REGISTRY_PATH={registry_path}")
            with open(registry_path, 'r') as f:
                raw = json.load(f)
        else:
            local_registry = self._models_dir / "registry.json"
            if local_registry.exists():
                print(f"[ModelRegistry] Registry source: {local_registry}")
                with open(local_registry, 'r') as f:
                    raw = json.load(f)

        if raw is None:
            print(f"[ModelRegistry] Registry source: DEFAULT_REGISTRY (no registry.json found)")
            return DEFAULT_REGISTRY.copy()

        # New format: {"domains": {key: {model_dir, status, ...}}, "fallback_model": ..., "common_model": ...}
        if "domains" in raw:
            flat = {}
            fallback = raw.get("fallback_model")
            for domain_key, entry in raw["domains"].items():
                model_dir = entry.get("model_dir")
                if model_dir:
                    flat[domain_key] = model_dir
            # Register fallback as "generic" if not already present
            if fallback and "generic" not in flat:
                flat["generic"] = fallback
            # Register common model (shared across all domains)
            common = raw.get("common_model", {})
            common_dir = common.get("model_dir") if isinstance(common, dict) else None
            if common_dir:
                flat["_common"] = common_dir
            return flat

        # Legacy flat format: {"domain": "model_dir_name"}
        return raw

    def _load_versions(self):
        """Parse per-domain versions from registry.json (new format only).
        Returns ({domain: {version: model_dir}}, {domain: active_version})."""
        versions, active = {}, {}
        registry_path = os.environ.get("DOMAIN_REGISTRY_PATH")
        path = Path(registry_path) if registry_path else (self._models_dir / "registry.json")
        try:
            if not path.exists():
                return versions, active
            with open(path, "r") as f:
                raw = json.load(f)
            for domain_key, entry in raw.get("domains", {}).items():
                if isinstance(entry, dict) and isinstance(entry.get("versions"), dict):
                    versions[domain_key] = dict(entry["versions"])
                    active[domain_key] = entry.get("active_version")
        except Exception as e:  # never let versioning break model loading
            print(f"[ModelRegistry] version parse skipped: {type(e).__name__}: {e}")
        return versions, active

    def list_versions(self, domain: str) -> dict:
        """{active, versions:[...]} for a domain. versions=[] means unversioned
        (only the single active model_dir exists)."""
        vmap = self._versions.get(domain, {})
        return {"active": self._active_version.get(domain),
                "versions": sorted(vmap.keys())}

    def register_domain(self, domain: str, model_dir_name: str):
        """Register a new domain -> model mapping."""
        self._registry[domain] = model_dir_name

    def get_model(self, domain: str = "generic", version: str = None):
        """
        Get model for a domain. Loads from disk on first call, cached thereafter.

        Args:
            domain: Domain name (e.g., "generic", "healthcare")
            version: Optional model version (e.g. "1.1"). When given and known,
                     loads that version's dir; otherwise loads the domain's
                     active model_dir (unchanged legacy behavior).

        Returns:
            tuple: (model, id_to_gloss, masked_class_ids)

        Raises:
            ValueError: If domain is not registered
            FileNotFoundError: If model directory doesn't exist
        """
        # Resolve which dir to load. Unknown/omitted version -> active model_dir.
        vmap = self._versions.get(domain, {})
        if version and version in vmap:
            model_dir_name = vmap[version]
            cache_key = f"{domain}@{version}"
        else:
            if domain not in self._registry:
                available = list(self._registry.keys())
                raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
            model_dir_name = self._registry[domain]
            cache_key = domain

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Load outside the lock (IO-bound)
        model_path = self._models_dir / model_dir_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found for domain '{domain}'"
                f"{f' version {version}' if version else ''}: {model_path}"
            )

        print(f"[ModelRegistry] Loading model for '{cache_key}' from {model_path}")
        model, id_to_gloss, masked_class_ids = load_model_from_checkpoint(str(model_path))

        with self._lock:
            self._cache[cache_key] = (model, id_to_gloss, masked_class_ids)

        return model, id_to_gloss, masked_class_ids

    def get_common_model(self):
        """Get the shared common words model. Returns (model, id_to_gloss, masked_class_ids) or None."""
        if "_common" not in self._registry:
            return None
        try:
            return self.get_model("_common")
        except (ValueError, FileNotFoundError):
            return None

    def predict(self, pickle_path: str, domain: str = "generic"):
        """
        Dual-model prediction: run domain model + common model, merge results.
        Falls back to domain-only if common model is unavailable.

        Returns:
            dict: {gloss, confidence, source, domain_result, common_result, top_k_predictions}
        """
        domain_model, domain_tokenizer, domain_masked = self.get_model(domain)
        common = self.get_common_model()

        if common:
            common_model, common_tokenizer, common_masked = common
            return predict_dual_model(
                pickle_path,
                domain_model=domain_model, domain_tokenizer=domain_tokenizer,
                domain_masked=domain_masked,
                common_model=common_model, common_tokenizer=common_tokenizer,
                common_masked=common_masked,
            )
        else:
            # No common model — single model prediction
            from openhands_modernized_inference import predict_pose_file
            result = predict_pose_file(
                pickle_path, model=domain_model, tokenizer=domain_tokenizer,
                masked_class_ids=domain_masked,
            )
            result['source'] = 'domain'
            return result

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
