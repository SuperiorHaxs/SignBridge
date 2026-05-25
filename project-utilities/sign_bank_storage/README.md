# sign_bank_storage

Utilities for managing the SignBridge sign-bank video library.

The sign-bank is the collection of canonical reference videos for each
ASL sign that the SignBridge app shows users (for practice / review).
Source: the WLASL public dataset (2000 glosses). Hosted on Cloudflare
R2 so the app works without the external drive being attached.

## Files

- `migrate_wlasl_to_r2.py` — one-time (and idempotent) script that
  walks the WLASL master JSON, picks one canonical video per gloss
  (signer 12 → signer 13 → first available), and uploads it to R2 as
  `<gloss>.mp4`. Skips glosses already self-signed locally + objects
  already present in R2.

## Prerequisites

- `python-dotenv` and `boto3` installed in the SignBridge venv
- Cloudflare R2 bucket + S3-compatible API token (see
  `applications/signbridge-app/.env.example` for the required vars)
- Local copy of the WLASL Kaggle dataset (default path:
  `D:\Projects\SignBridge\WLASL\datasets\wlasl-kaggle\`)

## Run

From the repo root:

```powershell
# Dry-run first to see what would happen (no uploads, no R2 calls
# beyond the initial bucket listing).
C:\Users\ashwi\Projects\SignBridge\asl-v1\venv\Scripts\python.exe ^
  project-utilities\sign_bank_storage\migrate_wlasl_to_r2.py --dry-run

# Smoke-test with a small upload (e.g. first 5 successful uploads).
... migrate_wlasl_to_r2.py --limit 5

# Full migration (uploads ~1985 videos, ~420 MB total).
... migrate_wlasl_to_r2.py
```

The script is safe to re-run -- it lists existing R2 keys at startup
and skips anything already uploaded. Use `--force` to re-upload.

## After the migration

Each video will be accessible at:

```
https://<R2_PUBLIC_BASE_URL>/<gloss>.mp4
```

Example:

```
https://pub-356ccbb34efd4ada8ba91c03fe330713.r2.dev/hello.mp4
```

The Flask `/api/sign-bank` endpoint will be updated separately to
return these URLs to the browser.
