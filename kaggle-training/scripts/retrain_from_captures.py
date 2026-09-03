#!/usr/bin/env python3
"""
retrain_from_captures.py
------------------------
Local admin utility: retrain a domain's ISOLATED model, but ONLY if there are
new user captures for glosses already in that model's vocabulary. No UI --
you run this from your machine.

Pipeline (all local up to the GPU step):
    1. Load the domain's *active* vocabulary from the deployed model's
       masked_classes.json (the authoritative "existing glosses").
    2. Scan captures in datasets/user_captures/<GLOSS>/*.pkl.
    3. Diff against the ledger -> the "pending" set (captured but not yet
       trained into the current model). If nothing pending -> exit, no retrain.
    4. Ingest pending captures into augmented_pool/pickle/<gloss>/ as the
       (T,83,3) pool schema the OpenHands trainer expects.
    5. Force sign-safe augmentation of just those new clips (no horizontal flip;
       flipping changes handedness).
    6. Hand off to the existing, proven pipeline:
         - backend=stage  : prepare_domain_kaggle.py <domain> --stage-only  (manifests+staging)
         - backend=upload : prepare_domain_kaggle.py <domain>               (+ Kaggle upload)
         - backend=local  : stage-only, then train_asl.py locally, relocate
                            output into a NEW minor-version model dir.
    7. Snapshot which clip_ids went into this batch so `--finalize` can register
       the trained weights as a new minor version and mark those clips ingested.

This never flips the *active* model version. Promotion to the new version is a
separate, deliberate step (`--promote`) gated on your own eval -- see Milestone 6.

Examples
    # Dry run: show what would happen for education, change nothing
    python retrain_from_captures.py education --dry-run

    # Local end-to-end retrain if >=10 new captures exist
    python retrain_from_captures.py education --backend local --min-new 10

    # Stage + upload to Kaggle (train on GPU there), then later:
    python retrain_from_captures.py education --backend upload
    #   ...train on Kaggle, download artifacts to some dir, then:
    python retrain_from_captures.py education --finalize ./downloaded_education_v1_1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# scripts -> kaggle-training -> asl-v1  (matches prepare_domain_kaggle.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MODELS = PROJECT_ROOT / "models" / "openhands-modernized" / "production-models"
REGISTRY_JSON = PRODUCTION_MODELS / "registry.json"
CAPTURES_DIR = PROJECT_ROOT / "datasets" / "user_captures"
POOL_DIR = PROJECT_ROOT / "datasets" / "augmented_pool" / "pickle"
LEDGER_PATH = CAPTURES_DIR / "_retrain_ledger.json"
PREPARE_SCRIPT = PROJECT_ROOT / "kaggle-training" / "scripts" / "prepare_domain_kaggle.py"
TRAIN_SCRIPT = PROJECT_ROOT / "models" / "training-scripts" / "train_asl.py"

def _load_env_file() -> None:
    """Load applications/signbridge-app/.env into os.environ (shell env wins) so a
    local retrain can read R2 captures without `source .env` first. Dependency-free."""
    env_path = PROJECT_ROOT / "applications" / "signbridge-app" / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't override an explicit shell value
                os.environ[key] = val
    except Exception as e:  # never let env loading block the run
        print(f"[env] could not read {env_path}: {type(e).__name__}: {e}")


_load_env_file()

# Shared capture store (R2 with local fallback) -- same module the Flask app uses.
sys.path.insert(0, str(PROJECT_ROOT / "applications" / "signbridge-app"))
import capture_store  # noqa: E402

# Sign-safe augmentation count per new capture (originals + this many aug clips).
DEFAULT_AUG_PER_CLIP = 8


# ─────────────────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────────────────
def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ledger(ledger: dict, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would write ledger -> {LEDGER_PATH}")
        return
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def domain_ledger(ledger: dict, domain: str) -> dict:
    """The per-domain ledger slice. Base (deployed) model is version 1.0 with an
    empty clip set, so on first run every capture is 'pending'."""
    return ledger.setdefault(domain, {
        "active_version": "1.0",
        "incorporated": {},          # clip_id -> version it was trained into
        "versions": {"1.0": {"base": True, "clips": []}},
        "pending_batch": None,       # snapshot written at stage time, consumed by --finalize
    })


def next_minor_version(dl: dict) -> str:
    majors_minors = []
    for v in dl["versions"]:
        try:
            maj, minor = v.split(".")
            majors_minors.append((int(maj), int(minor)))
        except ValueError:
            continue
    maj = max((m for m, _ in majors_minors), default=1)
    minor = max((n for m, n in majors_minors if m == maj), default=0)
    return f"{maj}.{minor + 1}"


# ─────────────────────────────────────────────────────────────────────────────
# Domain vocabulary + capture scanning
# ─────────────────────────────────────────────────────────────────────────────
def load_domain(domain: str) -> tuple[str, set[str]]:
    """Returns (model_dir_name, active_glosses_upper) from the deployed model."""
    reg = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    domains = reg.get("domains", reg)
    if domain not in domains:
        sys.exit(f"ERROR: domain '{domain}' not in registry. Known: {sorted(domains)}")
    entry = domains[domain]
    model_dir = entry["model_dir"] if isinstance(entry, dict) else entry
    mc_path = PRODUCTION_MODELS / model_dir / "masked_classes.json"
    if not mc_path.exists():
        sys.exit(f"ERROR: {mc_path} not found -- cannot read active vocabulary.")
    mc = json.loads(mc_path.read_text(encoding="utf-8"))
    active = {str(g).strip().upper() for g in (mc.get("active_glosses") or [])}
    if not active:
        sys.exit(f"ERROR: no active_glosses in {mc_path}")
    return model_dir, active


def scan_captures(domain: str, vocab: set[str]) -> list[dict]:
    """Captures from the shared store (R2 or local) whose gloss is in the
    domain's active vocabulary. Each item is {clip_id, gloss, ref, backend}."""
    return [cap for cap in capture_store.list_captures(domain)
            if cap["gloss"].strip().upper() in vocab]


# ─────────────────────────────────────────────────────────────────────────────
# Ingest + augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _augment_once(kp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sign-safe augmentation on (T,83,3). No horizontal flip (handedness)."""
    out = kp.copy()
    scale = rng.uniform(0.92, 1.08)
    out[..., :2] *= scale                                   # isotropic scale (x,y)
    out[..., 0] += rng.uniform(-0.03, 0.03)                 # translate x
    out[..., 1] += rng.uniform(-0.03, 0.03)                 # translate y
    out += rng.normal(0.0, 0.004, size=out.shape).astype(out.dtype)  # jitter
    if rng.random() < 0.5 and len(out) > 8:                 # mild temporal resample
        factor = rng.uniform(0.9, 1.1)
        new_t = max(8, int(round(len(out) * factor)))
        idx = np.clip(np.round(np.linspace(0, len(out) - 1, new_t)).astype(int), 0, len(out) - 1)
        out = out[idx]
    return out.astype(np.float32)


def ingest_and_augment(pending: list[dict], aug_per_clip: int, dry_run: bool) -> list[str]:
    """Write each pending capture (+ augmentations) into augmented_pool/pickle/<gloss>/.
    Returns the affected gloss set (lowercase pool dir names)."""
    affected: set[str] = set()
    rng = np.random.default_rng(1234)
    for cap in pending:
        gloss_lower = cap["gloss"].lower()
        affected.add(gloss_lower)
        gdir = POOL_DIR / gloss_lower
        rec = capture_store.read_record(cap)
        kp = np.asarray(rec["keypoints"], dtype=np.float32)
        if kp.ndim != 3 or kp.shape[1] != 83:
            print(f"  SKIP {cap['clip_id']}: unexpected keypoints shape {kp.shape}")
            continue
        kp = kp[:, :, :3]  # drop visibility channel -> (T,83,3)
        vid = cap["clip_id"]
        writes = [(f"{vid}.pkl", kp, False, None)]
        for i in range(aug_per_clip):
            writes.append((f"{vid}_aug_{i:02d}.pkl", _augment_once(kp, rng), True, vid))
        if dry_run:
            print(f"  [dry-run] {cap['gloss']}: would write {len(writes)} pkls to {gdir}")
            continue
        gdir.mkdir(parents=True, exist_ok=True)
        for fname, arr, is_aug, orig in writes:
            with open(gdir / fname, "wb") as f:
                pickle.dump({
                    "keypoints": arr,
                    "video_id": Path(fname).stem,
                    "gloss": gloss_lower,
                    "augmented": is_aug,
                    "original_video_id": orig or Path(fname).stem,
                    "landmark_config": "83pt",
                    "source": "user_capture",
                }, f)
    return sorted(affected)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline hand-off
# ─────────────────────────────────────────────────────────────────────────────
def run(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [dry-run] would run: {printable}")
        return
    print(f"  $ {printable}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        sys.exit(f"ERROR: command failed ({res.returncode}): {printable}")


def model_vocab(domain: str) -> list[str]:
    """The domain model's exact class list, in index order, from its
    class_index_mapping.json. This -- not education.json -- is the source of
    truth for the vocabulary (so masked_classes.json stays valid)."""
    model_dir = load_domain(domain)[0]
    cim = json.loads((PRODUCTION_MODELS / model_dir / "class_index_mapping.json").read_text(encoding="utf-8"))
    return [cim[str(i)] for i in range(len(cim))]


def _carry_over_vocab_files(domain: str, dest_dir: Path) -> None:
    """Copy the base model's class_index_mapping.json + masked_classes.json into a
    freshly trained version dir. train_asl sorts glosses alphabetically (same as
    the deployed model) so the indices align; carrying these over keeps the exact
    same active-gloss masking. train_asl never writes masked_classes.json itself."""
    base = PRODUCTION_MODELS / load_domain(domain)[0]
    for f in ("class_index_mapping.json", "masked_classes.json"):
        if (base / f).exists():
            shutil.copy2(base / f, dest_dir / f)


def _vocab_cfg(domain: str):
    """prepare_domain_kaggle cfg overridden to train the MODEL's own vocabulary
    (from class_index_mapping.json) instead of education.json. Returns (pdk, cfg, n)."""
    sys.path.insert(0, str(PREPARE_SCRIPT.parent))
    import prepare_domain_kaggle as pdk  # noqa: E402
    lower = [g.lower() for g in model_vocab(domain)]
    n = len(lower)
    cfg = pdk.get_domain_config(domain)
    cfg["all_glosses"] = lower
    cfg["total_classes"] = n
    cfg["splits_dir"] = POOL_DIR.parent / "splits" / f"{n}_{domain}"
    return pdk, cfg, n


def build_vocab_manifests(domain: str, dry_run: bool) -> tuple[Path, int]:
    """Build train/val/test manifests over the model's own vocabulary. train_asl
    sorts glosses alphabetically -> same class indices as the deployed model, so
    the carried-over masked_classes.json keeps exactly the same active glosses.
    Returns (splits_dir, num_classes)."""
    lower_n = len(model_vocab(domain))
    splits_dir = POOL_DIR.parent / "splits" / f"{lower_n}_{domain}"
    if dry_run:
        print(f"  [dry-run] would build manifests for {lower_n}-gloss model vocab -> {splits_dir}")
        return splits_dir, lower_n
    pdk, cfg, n = _vocab_cfg(domain)
    print(f"  building manifests for {n}-gloss model vocabulary ({domain})")
    if not pdk.step_split(cfg, dry_run=False):
        sys.exit("ERROR: manifest build (step_split) failed")
    return cfg["splits_dir"], n


def prepare_kaggle_vocab(domain: str, backend: str, dry_run: bool) -> None:
    """Stage (and, for backend='upload', upload) to Kaggle using the MODEL's own
    vocabulary, so a Kaggle-trained version matches the deployed class set /
    active glosses just like the local path does."""
    if dry_run:
        act = "stage + upload" if backend == "upload" else "stage"
        print(f"  [dry-run] would {act} {domain} for Kaggle using model vocabulary")
        return
    pdk, cfg, n = _vocab_cfg(domain)
    print(f"  preparing {n}-gloss model vocabulary for Kaggle ({domain}, backend={backend})")
    # augment (no-op: all model glosses already have pool data) -> split -> stage -> validate
    if not pdk.step_augment(cfg, dry_run=False):
        sys.exit("ERROR: step_augment failed")
    if not pdk.step_split(cfg, dry_run=False):
        sys.exit("ERROR: step_split failed")
    if not pdk.step_stage(cfg, dry_run=False):
        sys.exit("ERROR: step_stage failed")
    if not pdk.step_validate(cfg):
        sys.exit("ERROR: step_validate failed")
    if backend == "upload":
        if not pdk.step_upload(cfg, dry_run=False):
            sys.exit("ERROR: step_upload failed")


def train_local(domain: str, version: str, splits_dir: Path, num_classes: int, dry_run: bool, *,
                early_stopping: int = 80, max_samples_per_class: int | None = None,
                model_size: str = "small", epochs: int | None = None) -> Path:
    """Run train_asl.py on the model-vocab manifests, then relocate its hardcoded
    ./models/wlasl_<N>_class_model output into a versioned production dir."""
    new_dir = PRODUCTION_MODELS / f"wlasl_{num_classes}_{domain}_model_v{version.replace('.', '_')}"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--classes", str(num_classes),
        "--dataset", "augmented",
        "--augmented-path", str(POOL_DIR),
        "--manifest-dir", str(splits_dir),
        "--architecture", "openhands",
        "--model-size", model_size,
        "--early-stopping", str(early_stopping),
    ]
    if max_samples_per_class is not None:
        cmd += ["--max-samples-per-class", str(max_samples_per_class)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    run(cmd, dry_run)
    trained = PROJECT_ROOT / "models" / f"wlasl_{num_classes}_class_model"
    if dry_run:
        print(f"  [dry-run] would relocate {trained} -> {new_dir}")
        return new_dir
    if not (trained / "pytorch_model.bin").exists():
        sys.exit(f"ERROR: training produced no weights at {trained}")
    if new_dir.exists():
        shutil.rmtree(new_dir)
    shutil.copytree(trained, new_dir)
    _carry_over_vocab_files(domain, new_dir)  # keep the same active-gloss masking
    print(f"  new version weights -> {new_dir}")
    return new_dir


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────
def register_version(domain: str, version: str, model_dir_name: str,
                     clips: list[str], promote: bool, dry_run: bool) -> None:
    """Add the new version to registry.json (+ ledger). Keeps the old version;
    only flips 'active' when --promote is given."""
    ledger = load_ledger()
    dl = domain_ledger(ledger, domain)
    dl["versions"][version] = {
        "model_dir": model_dir_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": clips,
        "num_new": len(clips),
    }
    for cid in clips:
        dl["incorporated"][cid] = version
    dl["pending_batch"] = None
    if promote:
        dl["active_version"] = version
    save_ledger(ledger, dry_run)

    reg = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    domains = reg.get("domains", reg)
    entry = domains[domain]
    if not isinstance(entry, dict):
        entry = {"model_dir": entry}
        domains[domain] = entry
    versions = entry.setdefault("versions", {})
    # schema: {version_string: model_dir}. Seed base 1.0 = the dir the registry
    # currently points at, so the app can A/B the original against new versions.
    if not versions:
        versions["1.0"] = entry.get("model_dir", model_dir_name)
    versions[version] = model_dir_name
    entry["active_version"] = version if promote else entry.get("active_version", "1.0")
    if promote:
        entry["model_dir"] = model_dir_name  # app reads model_dir -> active version
    if dry_run:
        print(f"  [dry-run] would register version {version} ({model_dir_name}); "
              f"promote={promote}")
        return
    REGISTRY_JSON.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"  registered version {version} -> {model_dir_name} (active={entry['active_version']})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain", help="domain key, e.g. education")
    ap.add_argument("--backend", choices=["stage", "upload", "local"], default="stage",
                    help="stage: manifests+staging only; upload: + Kaggle upload; local: train here")
    ap.add_argument("--min-new", type=int, default=1, help="minimum new captures required to retrain")
    ap.add_argument("--aug-per-clip", type=int, default=DEFAULT_AUG_PER_CLIP)
    # local-training controls (passed to train_asl.py; ignored for stage/upload)
    ap.add_argument("--early-stopping", type=int, default=80, help="early-stop patience (small = fast)")
    ap.add_argument("--max-samples-per-class", type=int, default=None, help="cap dataset size (fast runs)")
    ap.add_argument("--epochs", type=int, default=None, help="cap total epochs (quick local runs)")
    ap.add_argument("--model-size", choices=["tiny", "small", "large"], default="small")
    ap.add_argument("--promote", action="store_true", help="flip active version to the new one")
    ap.add_argument("--finalize", metavar="DIR",
                    help="register an already-trained model DIR (e.g. Kaggle download) as the pending batch's version")
    ap.add_argument("--dry-run", action="store_true", help="print actions, change nothing")
    args = ap.parse_args()

    model_dir_name, vocab = load_domain(args.domain)
    print(f"Domain '{args.domain}': model={model_dir_name}, {len(vocab)} active glosses")

    ledger = load_ledger()
    dl = domain_ledger(ledger, args.domain)

    # ── finalize path: register a trained model as the pending batch's version ──
    if args.finalize:
        batch = dl.get("pending_batch")
        if not batch:
            sys.exit("ERROR: no pending_batch recorded. Run a stage/upload retrain first.")
        src = Path(args.finalize)
        if not (src / "pytorch_model.bin").exists() and not args.dry_run:
            sys.exit(f"ERROR: {src} has no pytorch_model.bin")
        version = batch["version"]
        new_dir_name = f"{model_dir_name}_v{version.replace('.', '_')}"
        dest = PRODUCTION_MODELS / new_dir_name
        if not args.dry_run:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            # train_asl never writes masked_classes.json, so a Kaggle download
            # lacks the active-gloss masking. Carry it over from the base model
            # (indices align: train_asl sorts alphabetically like the deployed model).
            _carry_over_vocab_files(args.domain, dest)
        else:
            print(f"  [dry-run] would copy {src} -> {dest} and carry over masked_classes.json")
        register_version(args.domain, version, new_dir_name, batch["clips"], args.promote, args.dry_run)
        return

    # ── detect new captures ──
    captures = scan_captures(args.domain, vocab)
    pending = [c for c in captures if c["clip_id"] not in dl["incorporated"]]
    by_gloss: dict[str, int] = {}
    for c in pending:
        by_gloss[c["gloss"]] = by_gloss.get(c["gloss"], 0) + 1
    print(f"Captures: {len(captures)} total, {len(pending)} new (not yet in a model).")
    if by_gloss:
        print("  new per gloss: " + ", ".join(f"{g}={n}" for g, n in sorted(by_gloss.items())))

    if len(pending) < args.min_new:
        print(f"Nothing to do: {len(pending)} new < --min-new {args.min_new}. No retrain.")
        return

    version = next_minor_version(dl)
    print(f"\nRetraining {args.domain} -> new version {version} (backend={args.backend})")

    affected = ingest_and_augment(pending, args.aug_per_clip, args.dry_run)
    print(f"  ingested {len(pending)} captures into pool; affected glosses: {affected}")

    clips = [c["clip_id"] for c in pending]
    if args.backend == "local":
        # Train the MODEL's own vocabulary so the new version has the same class
        # set (and, via the copied masked_classes.json, the same active glosses).
        splits_dir, num_classes = build_vocab_manifests(args.domain, args.dry_run)
        new_dir = train_local(args.domain, version, splits_dir, num_classes, args.dry_run,
                              early_stopping=args.early_stopping,
                              max_samples_per_class=args.max_samples_per_class,
                              model_size=args.model_size, epochs=args.epochs)
        register_version(args.domain, version, new_dir.name, clips, args.promote, args.dry_run)
        print(f"\nDone. Trained + registered version {version}.")
    else:
        # Kaggle backends stage/upload the MODEL's own vocabulary so a
        # Kaggle-trained version matches the deployed class set / active glosses,
        # just like the local path.
        prepare_kaggle_vocab(args.domain, args.backend, dry_run=args.dry_run)
        # record the batch so --finalize can register the Kaggle-trained weights later
        dl["pending_batch"] = {"version": version, "clips": clips,
                               "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_ledger(ledger, args.dry_run)
        print(f"\nStaged version {version}. Next:")
        if args.backend == "upload":
            print("  1. Run the Kaggle notebook to train on GPU.")
        else:
            print("  1. Upload/train (staging done; run prepare_domain_kaggle upload or the notebook).")
        print(f"  2. Download the trained model, then:")
        print(f"     python {Path(__file__).name} {args.domain} --finalize <downloaded_dir>"
              f"{' --promote' if args.promote else ''}")


if __name__ == "__main__":
    main()
