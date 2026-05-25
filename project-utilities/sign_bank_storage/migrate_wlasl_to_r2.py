"""
WLASL -> Cloudflare R2 sign-bank migration.

Walks the WLASL_v0.3.json manifest, picks ONE canonical video per gloss
(preferring signer 12, then signer 13, then first available), and uploads
it to the configured R2 bucket as `<gloss>.mp4`.

Skips:
  - Objects already present in R2 (idempotent -- safe to re-run).
  - Use --force to re-upload even those.

NOTE on local self-signed videos: this script uploads the WLASL version
to R2 for EVERY gloss, including those where we also have a local
self-signed video. The local files in `applications/show-and-tell/sign-bank/`
are NOT touched. The Flask endpoint decides which one to serve at
runtime (prefer-local-then-R2 is a sensible default).

Reads credentials from `applications/signbridge-app/.env`:
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET

Usage (from the SignBridge venv):

    python project-utilities/sign_bank_storage/migrate_wlasl_to_r2.py

Optional flags:
    --dry-run             Don't actually upload; print what WOULD upload.
    --limit N             Stop after N successful uploads (useful for
                          a quick smoke test before committing to 2000).
    --wlasl-root PATH     Override WLASL dataset root.
    --force               Re-upload even if the object already exists in
                          R2 (default: skip).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Repo layout discovery ────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent           # .../project-utilities/sign_bank_storage
ASL_V1_DIR   = SCRIPT_DIR.parent.parent                  # .../asl-v1
SIGNBRIDGE_APP_DIR = ASL_V1_DIR / "applications" / "signbridge-app"
SIGNER_12_MAP_PATH = ASL_V1_DIR / "dataset-utilities" / "categorization" / "signer_12_all_signs.json"

# Default WLASL root. Override with --wlasl-root if your drive letter changes.
DEFAULT_WLASL_ROOT = Path(r"D:\Projects\SignBridge\WLASL\datasets\wlasl-kaggle")


# ── Helpers ──────────────────────────────────────────────────────────

def gloss_to_filename(gloss: str) -> str:
    """
    Map a WLASL gloss to its R2 object key. Lowercased; whitespace becomes
    underscores; anything outside [a-z0-9_-] is dropped to keep the URL
    clean and shell-safe.
    """
    s = gloss.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return f"{s}.mp4"


def load_signer_12_map() -> dict:
    """Pre-built mapping gloss -> video_id (no extension) for signer 12."""
    if not SIGNER_12_MAP_PATH.exists():
        print(f"[warn] signer 12 map not found at {SIGNER_12_MAP_PATH}")
        return {}
    data = json.loads(SIGNER_12_MAP_PATH.read_text(encoding="utf-8"))
    return dict(data.get("signs", {}))


def build_signer_lookups(wlasl_json: list) -> tuple[dict, dict]:
    """
    Walk the WLASL master JSON and build:
      signer_12: gloss -> video_id (first signer-12 instance per gloss)
      signer_13: gloss -> video_id (first signer-13 instance per gloss)
    The on-disk signer_12 map covers most of this already, but we
    derive both here so the script is self-contained if the helper
    file is missing.
    """
    s12, s13 = {}, {}
    for entry in wlasl_json:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            sid = inst.get("signer_id")
            vid = str(inst.get("video_id"))
            if sid == 12 and gloss not in s12:
                s12[gloss] = vid
            elif sid == 13 and gloss not in s13:
                s13[gloss] = vid
    return s12, s13


def pick_video_for_gloss(
    gloss: str,
    entry: dict,
    signer_12_pref: dict,
    signer_13_built: dict,
    videos_root: Path,
) -> Path | None:
    """
    Apply the priority rule: signer 12 -> signer 13 -> first available.
    Returns the resolved Path on disk, or None if no video file exists.
    """
    # Try signer 12 first (use the pre-built map for accuracy; fall back
    # to whatever we derived from the WLASL JSON).
    candidates: list[str] = []
    vid12 = signer_12_pref.get(gloss) or signer_12_pref.get(gloss.strip())
    if vid12:
        candidates.append(vid12)
    vid13 = signer_13_built.get(gloss)
    if vid13 and vid13 not in candidates:
        candidates.append(vid13)
    # Last resort: any instance.
    for inst in entry.get("instances", []):
        vid = str(inst.get("video_id"))
        if vid not in candidates:
            candidates.append(vid)

    gloss_dir = videos_root / gloss
    if not gloss_dir.is_dir():
        return None
    for vid in candidates:
        # WLASL filenames are zero-padded; signer_12 map uses padded form
        # too ("00341"). The folder uses the same format.
        candidate_path = gloss_dir / f"{vid}.mp4"
        if candidate_path.exists():
            return candidate_path
    # Final fallback: just take whatever's in the folder (alphabetical).
    files = sorted(gloss_dir.glob("*.mp4"))
    return files[0] if files else None


def load_env_from_signbridge():
    """
    Load .env from applications/signbridge-app/.env using python-dotenv.
    Existing OS env vars take precedence (so production env vars set by
    the platform aren't overridden by a stale local .env).
    """
    env_path = SIGNBRIDGE_APP_DIR / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[fatal] python-dotenv not installed. Run:")
        print(f"  {ASL_V1_DIR / 'venv' / 'Scripts' / 'pip.exe'} install python-dotenv")
        sys.exit(1)
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        print(f"[warn] {env_path} not found; relying on OS env vars only.")


def make_s3_client():
    """
    boto3 S3 client configured for R2. R2 is S3-compatible; the trick is
    setting the endpoint URL + signature_version='s3v4'.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("[fatal] boto3 not installed. Run:")
        print(f"  {ASL_V1_DIR / 'venv' / 'Scripts' / 'pip.exe'} install boto3")
        sys.exit(1)

    key_id     = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint   = os.environ.get("R2_ENDPOINT")
    if not (key_id and secret_key and endpoint):
        print("[fatal] R2 credentials missing. Set R2_ACCESS_KEY_ID, "
              "R2_SECRET_ACCESS_KEY, R2_ENDPOINT in "
              f"{SIGNBRIDGE_APP_DIR / '.env'}")
        sys.exit(2)

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            # R2 quirk: 'auto' region works
            region_name="auto",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def list_existing_r2_keys(s3, bucket: str) -> set[str]:
    """Page through the bucket and collect existing keys. Used to skip
    already-uploaded objects so re-running the script is idempotent."""
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []) or []:
            keys.add(obj["Key"])
    return keys


def upload_one(s3, bucket: str, local_path: Path, key: str):
    """Upload a single MP4. Content-Type matters so the browser plays
    it inline rather than offering a download."""
    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=31536000, immutable",
        },
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't upload; print actions.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N uploads (0 = no limit).")
    ap.add_argument("--force", action="store_true",
                    help="Re-upload even if the object already exists.")
    ap.add_argument("--wlasl-root", type=Path, default=DEFAULT_WLASL_ROOT,
                    help=f"WLASL dataset root (default {DEFAULT_WLASL_ROOT}).")
    args = ap.parse_args()

    load_env_from_signbridge()

    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        print("[fatal] R2_BUCKET env var not set.")
        sys.exit(2)

    wlasl_json_path = args.wlasl_root / "WLASL_v0.3.json"
    videos_root     = args.wlasl_root / "videos"
    if not wlasl_json_path.exists():
        print(f"[fatal] WLASL manifest not found at {wlasl_json_path}")
        sys.exit(3)
    if not videos_root.is_dir():
        print(f"[fatal] WLASL videos dir not found at {videos_root}")
        sys.exit(3)

    print(f"WLASL root         : {args.wlasl_root}")
    print(f"R2 bucket          : {bucket}")
    print(f"Dry-run            : {args.dry_run}")
    print()
    # Local self-signed videos are intentionally NOT used to skip
    # uploads -- we want WLASL versions in R2 for every gloss. The
    # local files (in applications/show-and-tell/sign-bank/) remain
    # untouched; the Flask endpoint decides which to serve.

    # WLASL manifest + signer maps
    wlasl_data = json.loads(wlasl_json_path.read_text(encoding="utf-8"))
    signer_12_map = load_signer_12_map()
    _, signer_13_map = build_signer_lookups(wlasl_data)
    print(f"[lookup] signer 12 pre-built map : {len(signer_12_map)} entries")
    print(f"[lookup] signer 13 derived map   : {len(signer_13_map)} entries")
    print()

    # S3 client + existing keys (skip these unless --force)
    if not args.dry_run:
        s3 = make_s3_client()
        print(f"[probe] listing existing R2 keys in '{bucket}'...")
        existing_keys = list_existing_r2_keys(s3, bucket)
        print(f"[probe] {len(existing_keys)} objects already in bucket\n")
    else:
        s3 = None
        existing_keys = set()

    # Walk glosses, resolve video, upload
    stats = {"uploaded": 0, "skipped_r2": 0, "no_video": 0, "errors": 0}
    t0 = time.time()
    total_bytes = 0

    for entry in wlasl_data:
        gloss = entry["gloss"]
        key = gloss_to_filename(gloss)

        # Skip if already in R2 (unless --force). Local self-signed
        # glosses are NOT skipped -- we want the WLASL version of
        # everything in R2 as a fallback.
        if not args.force and key in existing_keys:
            stats["skipped_r2"] += 1
            continue

        # Resolve which video file to upload
        local_path = pick_video_for_gloss(
            gloss, entry, signer_12_map, signer_13_map, videos_root
        )
        if local_path is None:
            stats["no_video"] += 1
            print(f"[miss] {gloss!r:40} -> no video file found")
            continue

        size_mb = local_path.stat().st_size / 1024 / 1024
        prefix = "[dry] " if args.dry_run else "[up]  "
        print(f"{prefix}{gloss!r:40} -> {key:30} ({size_mb:.2f} MB)  "
              f"from {local_path.name}")

        if not args.dry_run:
            try:
                upload_one(s3, bucket, local_path, key)
                stats["uploaded"] += 1
                total_bytes += local_path.stat().st_size
            except Exception as e:
                stats["errors"] += 1
                print(f"[err]  {gloss!r}: {type(e).__name__}: {e}")

        if args.limit and stats["uploaded"] >= args.limit:
            print(f"\n[stop] --limit {args.limit} reached")
            break

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  uploaded     : {stats['uploaded']}")
    print(f"  skipped (R2) : {stats['skipped_r2']}   (already in bucket)")
    print(f"  no video     : {stats['no_video']}")
    print(f"  errors       : {stats['errors']}")
    if total_bytes:
        mb = total_bytes / 1024 / 1024
        print(f"  total uploaded: {mb:.1f} MB ({mb / max(elapsed, 0.001):.2f} MB/s)")


if __name__ == "__main__":
    main()
