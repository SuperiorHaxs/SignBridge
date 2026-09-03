"""
capture_store.py
----------------
Storage for user sign captures, backed by Cloudflare R2 with a local-disk
fallback. Deliberately self-contained -- only boto3 + stdlib + env vars, no
imports from app.py -- so BOTH the Flask app (which uploads captures) and the
offline retrain utility (which lists + downloads them) can share one layout and
one client.

Env (same names the rest of the app already uses):
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET
If any are missing (or boto3 isn't installed), the store transparently falls
back to local disk under datasets/user_captures/.

Object layout (identical in R2 and locally, so the retrain loop is agnostic):
    captures/<domain>/<GLOSS>/<clip_id>.pkl     # keypoints record (pickle)
    captures/<domain>/<GLOSS>/<clip_id>.webm    # raw clip (optional, for review)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

# applications/signbridge-app/ -> applications -> asl-v1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = PROJECT_ROOT / "datasets" / "user_captures"
PREFIX = "captures"

_CLIENT = None
_CLIENT_ATTEMPTED = False


def _client():
    """Lazy boto3 S3 client for R2, or None if creds/boto3 missing.
    Mirrors app.py's _r2_client() so behavior is identical."""
    global _CLIENT, _CLIENT_ATTEMPTED
    if _CLIENT is not None or _CLIENT_ATTEMPTED:
        return _CLIENT
    _CLIENT_ATTEMPTED = True
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("R2_ENDPOINT")
    if not (key_id and secret and endpoint):
        return None
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return None
    _CLIENT = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=key_id, aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", region_name="auto",
                      retries={"max_attempts": 3, "mode": "standard"}),
    )
    return _CLIENT


def _bucket() -> str | None:
    """Bucket for captures. Prefer a dedicated PRIVATE bucket so user webcam
    clips never share the public sign-bank bucket; fall back to R2_BUCKET."""
    return os.environ.get("R2_CAPTURES_BUCKET") or os.environ.get("R2_BUCKET")


def r2_enabled() -> bool:
    return _client() is not None and bool(_bucket())


def _key(domain: str, gloss: str, clip_id: str, ext: str) -> str:
    return f"{PREFIX}/{domain}/{gloss.upper()}/{clip_id}.{ext}"


def _local_path(domain: str, gloss: str, clip_id: str, ext: str) -> Path:
    return LOCAL_ROOT / domain / gloss.upper() / f"{clip_id}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────
def save_capture(domain: str, gloss: str, clip_id: str, record: dict,
                 webm_bytes: bytes | None = None) -> dict:
    """Persist one capture. Returns {'backend': 'r2'|'local', 'location': str}.
    `record` is the keypoints dict; it is pickled here."""
    pkl_bytes = pickle.dumps(record)
    if r2_enabled():
        client, bucket = _client(), _bucket()
        client.put_object(Bucket=bucket, Key=_key(domain, gloss, clip_id, "pkl"),
                          Body=pkl_bytes, ContentType="application/octet-stream")
        if webm_bytes:
            try:
                client.put_object(Bucket=bucket, Key=_key(domain, gloss, clip_id, "webm"),
                                  Body=webm_bytes, ContentType="video/webm")
            except Exception as e:  # noqa: BLE001 -- video is best-effort
                print(f"[capture_store] webm upload failed for {clip_id}: {e}")
        return {"backend": "r2", "location": _key(domain, gloss, clip_id, "pkl")}
    # local fallback
    p = _local_path(domain, gloss, clip_id, "pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pkl_bytes)
    if webm_bytes:
        try:
            p.with_suffix(".webm").write_bytes(webm_bytes)
        except Exception:
            pass
    return {"backend": "local", "location": str(p)}


# ─────────────────────────────────────────────────────────────────────────────
# Read / list  (used by /api/captures and the retrain utility)
# ─────────────────────────────────────────────────────────────────────────────
def list_captures(domain: str) -> list[dict]:
    """Every capture for a domain as {clip_id, gloss, ref}. `ref` is an opaque
    handle (R2 key or local path) to pass back to read_record()."""
    out: list[dict] = []
    if r2_enabled():
        client, bucket = _client(), _bucket()
        token = None
        prefix = f"{PREFIX}/{domain}/"
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = client.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".pkl"):
                    continue
                parts = key.split("/")  # captures/<domain>/<GLOSS>/<clip>.pkl
                if len(parts) < 4:
                    continue
                out.append({"clip_id": Path(parts[-1]).stem, "gloss": parts[-2].upper(),
                            "ref": key, "backend": "r2"})
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return out
    # local fallback
    base = LOCAL_ROOT / domain
    if base.exists():
        for pkl in base.glob("*/*.pkl"):
            out.append({"clip_id": pkl.stem, "gloss": pkl.parent.name.upper(),
                        "ref": str(pkl), "backend": "local"})
    return out


def read_record(capture: dict) -> dict:
    """Load the pickled keypoints record for a capture returned by list_captures()."""
    if capture.get("backend") == "r2":
        obj = _client().get_object(Bucket=_bucket(), Key=capture["ref"])
        return pickle.loads(obj["Body"].read())
    with open(capture["ref"], "rb") as f:
        return pickle.load(f)


def inventory(domain: str) -> dict:
    """Cheap per-gloss counts from the object listing (no per-file reads)."""
    by_gloss: dict[str, int] = {}
    for cap in list_captures(domain):
        by_gloss[cap["gloss"]] = by_gloss.get(cap["gloss"], 0) + 1
    return {"by_gloss": by_gloss, "total": sum(by_gloss.values()),
            "backend": "r2" if r2_enabled() else "local"}
