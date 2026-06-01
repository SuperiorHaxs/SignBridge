"""
Build a single test paragraph video for the Upload tab (backlog item 10),
by stitching the per-gloss WLASL videos already on R2 (which are signer-12-
preferred from the migration script).

The video is a 5-sentence emergency narrative -- see SENTENCES below. Each
gloss segment is normalized to 480x360 @ 30fps so the motion gate sees the
same frame cadence the live stream is calibrated for. Pauses are inserted
between glosses (1.5s) and between sentences (3.0s) using ffmpeg's tpad
filter (freeze the last frame for the pause), so the motion gate cleanly
closes each segment and the LLM has temporal cues for sentence boundaries.

Output (default):
    project-utilities/test_data/out/emergency_paragraph.mp4

Usage (from repo root, in the SignBridge venv):

    python project-utilities/test_data/build_paragraph_video.py
    python project-utilities/test_data/build_paragraph_video.py --keep-temp

The script is one-shot and idempotent: re-running overwrites the output.
Add new narratives by adding a dict to NARRATIVES and selecting it via
--narrative <name>.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ── Repo layout ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent           # .../project-utilities/test_data
ASL_V1_DIR = SCRIPT_DIR.parent.parent                   # .../asl-v1
DEFAULT_OUT = SCRIPT_DIR / "out"

# ── ffmpeg discovery ────────────────────────────────────────────────
def find_ffmpeg() -> str:
    """Same priority as app._find_ffmpeg: venv-bundled imageio_ffmpeg first,
    then PATH. Raises if neither works."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from shutil import which
    p = which("ffmpeg") or which("ffmpeg.exe")
    if p:
        return p
    print("[fatal] ffmpeg not found. Install imageio_ffmpeg in the venv:")
    print(f"  {ASL_V1_DIR / 'venv' / 'Scripts' / 'python.exe'} -m pip install imageio_ffmpeg")
    sys.exit(2)


# ── Narratives ──────────────────────────────────────────────────────
# Each list is one sentence; each item in the sentence is (GLOSS, pause_after_seconds).
# Pause after the LAST gloss of a sentence is the sentence-break pause (longer).
NARRATIVES = {
    "emergency": [
        # Sentence 1: Discover the fire
        [("ATTENTION", 1.5), ("FIRE", 1.5), ("BATHROOM", 3.0)],
        # Sentence 2: Alarm rouses everyone
        [("ALARM", 1.5), ("LOUD", 1.5), ("AWAKE", 1.5), ("ALL", 3.0)],
        # Sentence 3: Direct people to exit
        [("HURRY", 1.5), ("APPROACH", 1.5), ("DOOR", 3.0)],
        # Sentence 4: Help vulnerable evacuees
        [("ASSIST", 1.5), ("BLIND", 1.5), ("ADULT", 1.5), ("CAREFUL", 3.0)],
        # Sentence 5: Resolution -- responders on scene
        [("FIREFIGHTER", 1.5), ("ARRIVE", 1.5), ("HOSPITAL", 1.0)],   # small trail
    ],
    # Built from the emergency-active glosses that ALSO have a signer-12
    # clip in the WLASL sign bank (so the whole paragraph is the same
    # signer -- visual consistency is essential for the recognizer's
    # motion-gate + pose-feature path). Targets ~60s runtime; arc is
    # warning -> shelter -> injuries -> rescue -> recovery.
    "hurricane": [
        # Sentence 1: Warning / wake-up
        [("ALARM", 1.5), ("LOUD", 1.5), ("HURRICANE", 1.5), ("AWAKE", 3.0)],
        # Sentence 2: Take shelter
        [("ALL", 1.5), ("HURRY", 1.5), ("BATHROOM", 1.5), ("CAREFUL", 3.0)],
        # Sentence 3: Personal injuries / distress
        [("COLD", 1.5), ("ABDOMEN", 1.5), ("HEADACHE", 1.5), ("BREATHE", 3.0)],
        # Sentence 4: Call for help
        [("ASSIST", 1.5), ("HURRY", 1.5), ("HOSPITAL", 3.0)],
        # Sentence 5: Resolution / recovery
        [("MEDICINE", 1.5), ("BETTER", 1.5), ("ALL", 1.0)],   # small trail
    ],
}


# ── R2 URL builder (mirrors migrate_wlasl_to_r2.gloss_to_filename) ──
R2_PUBLIC_BASE_URL = os.environ.get(
    "R2_PUBLIC_BASE_URL",
    "https://pub-356ccbb34efd4ada8ba91c03fe330713.r2.dev",
).rstrip("/")


def gloss_to_r2_url(gloss: str) -> str:
    s = gloss.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return f"{R2_PUBLIC_BASE_URL}/{s}.mp4"


# ── Per-clip ffmpeg pipeline ────────────────────────────────────────
# Normalize size + fps + add freeze-frame pause at end.
#   scale: fit inside 480x360 preserving aspect.
#   pad:   letterbox/pillarbox with black so output is exactly 480x360.
#   fps:   30 (matches typical webcam capture; tpad clones at this fps).
#   tpad:  clone last frame for `pause` seconds.
TARGET_W, TARGET_H, TARGET_FPS = 480, 360, 30

def _vf_normalize(pause_s: float) -> str:
    # scale "fit-inside-W-x-H": shrink dim that's already constrained.
    scale = (
        f"scale='if(gt(iw/ih,{TARGET_W}/{TARGET_H}),{TARGET_W},-2)':"
        f"'if(gt(iw/ih,{TARGET_W}/{TARGET_H}),-2,{TARGET_H})'"
    )
    pad = f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black"
    fps = f"fps={TARGET_FPS}"
    tpad = f"tpad=stop_mode=clone:stop_duration={pause_s:.2f}"
    return ",".join([scale, pad, fps, tpad])


def normalize_clip(ffmpeg: str, src: Path, dst: Path, pause_s: float):
    """Normalize a single WLASL clip to 480x360@30fps and pad with a freeze
    of `pause_s` seconds at the end. Re-encodes to H.264 yuv420p, no audio."""
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vf", _vf_normalize(pause_s),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize failed for {src.name}:\n{r.stderr[-600:]}")


def concat_clips(ffmpeg: str, clip_paths: list[Path], out: Path):
    """Concat MP4s using the concat demuxer. All clips MUST be the same
    codec / size / fps -- normalize_clip guarantees that."""
    list_path = out.parent / "_concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            # ffmpeg concat list paths are quoted single-quote-style; escape any quotes
            f.write(f"file '{p.as_posix()}'\n")
    cmd = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        # -c copy works because every clip is normalized to identical codec params.
        "-c", "copy", "-movflags", "+faststart",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{r.stderr[-600:]}")


# Cloudflare's R2 public bucket blocks the default urllib User-Agent
# ("Python-urllib/3.x") with 403. A browser-like UA gets through.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def download(url: str, dst: Path):
    """Stream-download to dst. Raises on HTTP error."""
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        dst.write_bytes(resp.read())


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--narrative", default="emergency",
                    choices=list(NARRATIVES.keys()),
                    help="Which narrative to build (default: emergency).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"Output dir (default: {DEFAULT_OUT}).")
    ap.add_argument("--keep-temp", action="store_true",
                    help="Keep the per-clip normalized files for inspection.")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
    print(f"[ffmpeg] {ffmpeg}")

    narrative = NARRATIVES[args.narrative]
    # Flatten to (gloss, pause) sequence for processing; remember sentence breaks for the log.
    flat: list[tuple[str, float]] = [g for sent in narrative for g in sent]
    print(f"\n[narrative={args.narrative}] {sum(len(s) for s in narrative)} glosses across "
          f"{len(narrative)} sentences:")
    for i, sent in enumerate(narrative, 1):
        print(f"  Sentence {i}: " + " · ".join(g for g, _ in sent))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"{args.narrative}_paragraph.mp4"

    tmp_dir = Path(tempfile.mkdtemp(prefix="signbridge_paragraph_"))
    print(f"\n[temp] {tmp_dir}")

    try:
        normalized: list[Path] = []
        for idx, (gloss, pause) in enumerate(flat, start=1):
            url = gloss_to_r2_url(gloss)
            raw = tmp_dir / f"{idx:02d}_{gloss.replace(' ', '_')}_raw.mp4"
            norm = tmp_dir / f"{idx:02d}_{gloss.replace(' ', '_')}_norm.mp4"
            print(f"  [{idx:02d}] {gloss:14s}  pause={pause:.1f}s  src={url}")
            try:
                download(url, raw)
            except Exception as e:
                print(f"     ! download failed: {e}")
                print(f"       (gloss not in R2 sign-bank? -- skipping this clip)")
                continue
            try:
                normalize_clip(ffmpeg, raw, norm, pause_s=pause)
            except Exception as e:
                print(f"     ! normalize failed: {e}")
                continue
            normalized.append(norm)

        if not normalized:
            print("[fatal] no clips were prepared; nothing to concat.")
            sys.exit(3)

        print(f"\n[concat] {len(normalized)} clips -> {out_file}")
        concat_clips(ffmpeg, normalized, out_file)
        size_mb = out_file.stat().st_size / 1024 / 1024
        print(f"[done] {out_file}  ({size_mb:.1f} MB)")
        print()
        print("Drop this file into the Upload tab to test end-to-end.")

    finally:
        if args.keep_temp:
            print(f"[temp kept] {tmp_dir}")
        else:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
