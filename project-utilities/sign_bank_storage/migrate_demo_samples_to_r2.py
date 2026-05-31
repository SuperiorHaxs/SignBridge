"""
Demo-data samples -> Cloudflare R2 migration.

Uploads the breakdown/scripted-demo media (the `applications/show-and-tell/
demo-data/samples/<sample_id>/...` tree) to the same R2 bucket the sign
bank uses, under the `demo-samples/` prefix.

Why this exists: the original_video.mp4 files run 14-28 MB each. HuggingFace
Spaces rejects regular (non-LFS) files over ~10 MB on push, so when the
sync Action uploaded the repo, the small pose_video.mp4 made it but the
big original_video.mp4 did not -- demos couldn't play on the deployed
Space. Moving the videos to R2 (which has free egress and no per-file
HF limit) sidesteps the whole problem; the Flask app falls back to the
R2 URL when the local file is missing, mirroring the sign-bank pattern.

R2 key scheme:
    demo-samples/<sample_id>/<relative_path>

So e.g.:
    applications/show-and-tell/demo-data/samples/
        fire-happen-fast-explode-loud/original_video.mp4
becomes
    demo-samples/fire-happen-fast-explode-loud/original_video.mp4

What gets uploaded (default): every .mp4 in the samples tree -- original,
pose, and segments. Pass --include-pose to also upload .pose binaries
(they sit at 2-3 MB and currently sync fine, but you may want them in R2
for symmetry or future >10 MB additions). metadata.json stays in git --
it's tiny structured config the app reads directly.

Idempotent: existing R2 keys are skipped unless --force.

Reads credentials from `applications/signbridge-app/.env`:
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET

Usage (from the SignBridge venv):

    python project-utilities/sign_bank_storage/migrate_demo_samples_to_r2.py
    python project-utilities/sign_bank_storage/migrate_demo_samples_to_r2.py --dry-run
    python project-utilities/sign_bank_storage/migrate_demo_samples_to_r2.py --force
    python project-utilities/sign_bank_storage/migrate_demo_samples_to_r2.py --include-pose
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Repo layout discovery ────────────────────────────────────────────
SCRIPT_DIR         = Path(__file__).resolve().parent
ASL_V1_DIR         = SCRIPT_DIR.parent.parent
SIGNBRIDGE_APP_DIR = ASL_V1_DIR / "applications" / "signbridge-app"
SAMPLES_DIR        = ASL_V1_DIR / "applications" / "show-and-tell" / "demo-data" / "samples"

R2_KEY_PREFIX = "demo-samples"   # all sample objects live under this prefix


# ── Helpers (mirror migrate_wlasl_to_r2.py for credentials + S3 setup) ──

def load_env_from_signbridge():
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
            region_name="auto",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def list_existing_r2_keys(s3, bucket: str, prefix: str) -> set[str]:
    """Page through the bucket under the given prefix; collect existing keys."""
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.add(obj["Key"])
    return keys


# Filename extension -> (ContentType, CacheControl)
_CONTENT_TYPES = {
    ".mp4":  "video/mp4",
    ".pose": "application/octet-stream",
}
_CACHE_HEADERS = "public, max-age=31536000, immutable"


def upload_one(s3, bucket: str, local_path: Path, key: str):
    ext = local_path.suffix.lower()
    content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={
            "ContentType": content_type,
            "CacheControl": _CACHE_HEADERS,
        },
    )


def discover_files(samples_root: Path, include_pose: bool) -> list[tuple[Path, str]]:
    """
    Walk samples_root and return [(local_path, r2_key), ...].
    r2_key = '<R2_KEY_PREFIX>/<sample_id>/<relative_path_with_forward_slashes>'.
    """
    out: list[tuple[Path, str]] = []
    if not samples_root.is_dir():
        return out
    for sample_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        sample_id = sample_dir.name
        for path in sorted(sample_dir.rglob("*")):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext == ".mp4" or (include_pose and ext == ".pose"):
                rel = path.relative_to(sample_dir).as_posix()
                key = f"{R2_KEY_PREFIX}/{sample_id}/{rel}"
                out.append((path, key))
    return out


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't upload; print what would upload.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N uploads (0 = no limit).")
    ap.add_argument("--force", action="store_true",
                    help="Re-upload even if the object already exists in R2.")
    ap.add_argument("--include-pose", action="store_true",
                    help="Also upload .pose binaries (default: only .mp4).")
    ap.add_argument("--samples-root", type=Path, default=SAMPLES_DIR,
                    help=f"Samples directory (default {SAMPLES_DIR}).")
    args = ap.parse_args()

    load_env_from_signbridge()

    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        print("[fatal] R2_BUCKET env var not set.")
        sys.exit(2)

    if not args.samples_root.is_dir():
        print(f"[fatal] samples dir not found at {args.samples_root}")
        sys.exit(3)

    files = discover_files(args.samples_root, include_pose=args.include_pose)
    if not files:
        print(f"[fatal] no matching files under {args.samples_root}")
        sys.exit(3)

    total_mb = sum(p.stat().st_size for p, _ in files) / 1024 / 1024
    print(f"Samples root  : {args.samples_root}")
    print(f"R2 bucket     : {bucket}")
    print(f"R2 prefix     : {R2_KEY_PREFIX}/")
    print(f"Include .pose : {args.include_pose}")
    print(f"Discovered    : {len(files)} files ({total_mb:.1f} MB)")
    print(f"Dry-run       : {args.dry_run}")
    print()

    if not args.dry_run:
        s3 = make_s3_client()
        print(f"[probe] listing existing keys under '{R2_KEY_PREFIX}/' ...")
        existing_keys = list_existing_r2_keys(s3, bucket, R2_KEY_PREFIX + "/")
        print(f"[probe] {len(existing_keys)} objects already in bucket under that prefix\n")
    else:
        s3 = None
        existing_keys = set()

    stats = {"uploaded": 0, "skipped_r2": 0, "errors": 0}
    t0 = time.time()
    total_bytes = 0

    for local_path, key in files:
        if not args.force and key in existing_keys:
            stats["skipped_r2"] += 1
            continue

        size_mb = local_path.stat().st_size / 1024 / 1024
        prefix = "[dry] " if args.dry_run else "[up]  "
        print(f"{prefix}{key}  ({size_mb:.2f} MB)")

        if not args.dry_run:
            try:
                upload_one(s3, bucket, local_path, key)
                stats["uploaded"] += 1
                total_bytes += local_path.stat().st_size
            except Exception as e:
                stats["errors"] += 1
                print(f"[err]  {key}: {type(e).__name__}: {e}")

        if args.limit and stats["uploaded"] >= args.limit:
            print(f"\n[stop] --limit {args.limit} reached")
            break

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  uploaded     : {stats['uploaded']}")
    print(f"  skipped (R2) : {stats['skipped_r2']}   (already in bucket)")
    print(f"  errors       : {stats['errors']}")
    if total_bytes:
        mb = total_bytes / 1024 / 1024
        print(f"  total uploaded: {mb:.1f} MB ({mb / max(elapsed, 0.001):.2f} MB/s)")


if __name__ == "__main__":
    main()
