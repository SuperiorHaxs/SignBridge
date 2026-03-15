#!/usr/bin/env python3
"""
ASL Video Crawler

Searches YouTube (and custom URLs) for ASL sign videos, downloads them,
and clips individual signs based on timestamp annotations.

Workflow:
  1. SEARCH:  Find candidate videos for each target word
  2. REVIEW:  Preview results and add clip timestamps to config
  3. DOWNLOAD: Download selected videos
  4. CLIP:    Extract individual sign clips using timestamps

Requirements:
  pip install yt-dlp ffmpeg-python

Usage:
  python crawl_asl_videos.py --search              # Step 1: search and save candidates
  python crawl_asl_videos.py --download             # Step 2: download selected videos
  python crawl_asl_videos.py --clip                 # Step 3: clip signs from downloaded videos
  python crawl_asl_videos.py --auto-annotate         # Step 2b: auto-detect clip timestamps
  python crawl_asl_videos.py --all                   # Run full pipeline
  python crawl_asl_videos.py --download-url URL WORD # Download a specific URL for a word
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Common yt-dlp flags for all commands
YT_DLP_BASE = ['yt-dlp', '--js-runtimes', 'node']


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def save_config(config_path, config):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def ensure_dirs(config, base_dir):
    for key in ['raw_videos_dir', 'clipped_videos_dir', 'metadata_dir']:
        d = base_dir / config['output'][key]
        d.mkdir(parents=True, exist_ok=True)


def search_youtube(word, config):
    """Search YouTube for ASL sign videos using yt-dlp."""
    templates = config['search_templates']
    yt_config = config['sources']['youtube']
    max_results = yt_config['max_results_per_word']
    min_dur = yt_config['min_duration_sec']
    max_dur = yt_config['max_duration_sec']

    all_results = []
    seen_ids = set()

    for template in templates:
        query = template.format(word=word)
        print(f"  Searching: '{query}'")

        cmd = YT_DLP_BASE + [
            f'ytsearch{max_results}:{query}',
            '--dump-json',
            '--flat-playlist',
            '--no-download',
            '--quiet',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    vid_id = entry.get('id', '')
                    if vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)

                    duration = entry.get('duration', 0) or 0
                    if duration < min_dur or duration > max_dur:
                        continue

                    all_results.append({
                        'id': vid_id,
                        'title': entry.get('title', ''),
                        'url': entry.get('url', f'https://www.youtube.com/watch?v={vid_id}'),
                        'duration': duration,
                        'channel': entry.get('channel', entry.get('uploader', '')),
                        'view_count': entry.get('view_count', 0),
                    })
                except json.JSONDecodeError:
                    continue
        except subprocess.TimeoutExpired:
            print(f"    Timeout searching for '{query}'")
        except FileNotFoundError:
            print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
            sys.exit(1)

    # Sort by view count (higher = more likely quality content)
    all_results.sort(key=lambda x: x.get('view_count', 0), reverse=True)

    # Boost preferred channels
    preferred = set(ch.lower() for ch in yt_config.get('preferred_channels', []))
    boosted = []
    rest = []
    for r in all_results:
        if r.get('channel', '').lower() in preferred:
            boosted.append(r)
        else:
            rest.append(r)

    return boosted + rest


def do_search(config, base_dir):
    """Search for candidate videos for all target words."""
    metadata_dir = base_dir / config['output']['metadata_dir']
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_candidates = {}

    for word in config['target_words']:
        print(f"\nSearching for: {word.upper()}")

        if config['sources']['youtube']['enabled']:
            results = search_youtube(word, config)
            all_candidates[word] = results
            print(f"  Found {len(results)} candidates")

            for i, r in enumerate(results[:5]):
                print(f"    [{i}] {r['title'][:60]}  ({r['duration']}s)  {r['channel']}")
                print(f"        {r['url']}")

    # Save search results
    results_file = metadata_dir / 'search_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'candidates': all_candidates,
        }, f, indent=2)

    print(f"\nSearch results saved to: {results_file}")
    print("\nNext steps:")
    print("  1. Review the search results")
    print("  2. Add clip timestamps to crawler_config.json under 'clip_annotations'")
    print("     Format: {word: [{url, start_sec, end_sec, label}]}")
    print("  3. Run: python crawl_asl_videos.py --download")


def do_download(config, base_dir, max_per_word=5):
    """Download videos from search results and/or clip annotations."""
    raw_dir = base_dir / config['output']['raw_videos_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = base_dir / config['output']['metadata_dir']
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Build download list from two sources:
    # 1. clip_annotations (manually or previously annotated)
    # 2. search_results (auto-download top candidates)
    download_targets = {}  # word -> [{url, vid_id}]

    # Source 1: clip annotations
    annotations = config.get('clip_annotations', {})
    for word, clips in annotations.items():
        for clip in clips:
            url = clip.get('url', '')
            if url:
                download_targets.setdefault(word, []).append(url)

    # Source 2: search results (fill up to max_per_word)
    search_file = metadata_dir / 'search_results.json'
    if search_file.exists():
        with open(search_file) as f:
            search_data = json.load(f)
        for word, candidates in search_data.get('candidates', {}).items():
            existing = set(download_targets.get(word, []))
            for candidate in candidates[:max_per_word]:
                url = candidate.get('url', '')
                if url and url not in existing:
                    download_targets.setdefault(word, []).append(url)
                    existing.add(url)

    download_log = []

    for word, urls in download_targets.items():
        word_dir = raw_dir / word.lower()
        word_dir.mkdir(parents=True, exist_ok=True)

        for i, url in enumerate(urls[:max_per_word]):
            vid_id = url.split('v=')[-1].split('&')[0] if 'youtube' in url else f'video_{i}'
            output_path = word_dir / f'{vid_id}.mp4'

            if output_path.exists():
                print(f"  Already downloaded: {word}/{output_path.name}")
                download_log.append({'word': word, 'url': url, 'file': str(output_path)})
                continue

            print(f"  Downloading: {word.upper()} - {url}")
            cmd = YT_DLP_BASE + [
                url,
                '-f', 'best[ext=mp4]/best',
                '-o', str(output_path),
                '--no-playlist',
                '--quiet',
                '--progress',
            ]

            try:
                result = subprocess.run(cmd, timeout=120)
                if result.returncode == 0 and output_path.exists():
                    print(f"    Saved: {output_path}")
                    download_log.append({
                        'word': word,
                        'url': url,
                        'file': str(output_path),
                        'timestamp': datetime.now().isoformat(),
                    })
                else:
                    print(f"    FAILED: {url}")
            except subprocess.TimeoutExpired:
                print(f"    Timeout: {url}")
            except FileNotFoundError:
                print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
                sys.exit(1)

    # Save download log
    log_file = metadata_dir / 'download_log.json'
    with open(log_file, 'w') as f:
        json.dump(download_log, f, indent=2)

    print(f"\nDownloaded {len(download_log)} videos. Log: {log_file}")


def download_single_url(url, word, config, base_dir):
    """Download a single URL for a specific word."""
    raw_dir = base_dir / config['output']['raw_videos_dir'] / word.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    vid_id = url.split('v=')[-1].split('&')[0] if 'youtube' in url else 'video_custom'
    output_path = raw_dir / f'{vid_id}.mp4'

    print(f"Downloading: {word.upper()} - {url}")
    cmd = YT_DLP_BASE + [
        url,
        '-f', 'best[ext=mp4]/best',
        '-o', str(output_path),
        '--no-playlist',
        '--progress',
    ]

    try:
        result = subprocess.run(cmd, timeout=120)
        if result.returncode == 0 and output_path.exists():
            print(f"Saved: {output_path}")
            return str(output_path)
        else:
            print(f"FAILED: {url}")
    except subprocess.TimeoutExpired:
        print(f"Timeout: {url}")
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)
    return None


def do_clip(config, base_dir):
    """Clip individual signs from downloaded videos using timestamp annotations."""
    raw_dir = base_dir / config['output']['raw_videos_dir']
    clip_dir = base_dir / config['output']['clipped_videos_dir']
    clip_dir.mkdir(parents=True, exist_ok=True)

    output_cfg = config['output']
    target_fps = output_cfg.get('target_fps', 30)
    target_res = output_cfg.get('target_resolution', [256, 256])
    max_clip_dur = output_cfg.get('max_clip_duration_sec', 5)

    annotations = config.get('clip_annotations', {})
    clip_count = 0

    for word, clips in annotations.items():
        if not clips:
            continue

        word_clip_dir = clip_dir / word.lower()
        word_clip_dir.mkdir(parents=True, exist_ok=True)

        for i, clip in enumerate(clips):
            url = clip.get('url', '')
            start = clip.get('start_sec', 0)
            end = clip.get('end_sec', None)
            label = clip.get('label', word)

            if end is None:
                print(f"  SKIP {word}[{i}]: no end_sec timestamp")
                continue

            duration = end - start
            if duration > max_clip_dur:
                print(f"  WARN {word}[{i}]: clip is {duration}s (max {max_clip_dur}s), trimming")
                end = start + max_clip_dur

            # Find source video
            vid_id = url.split('v=')[-1].split('&')[0] if 'youtube' in url else f'video_{i}'
            source_path = raw_dir / word.lower() / f'{vid_id}.mp4'

            if not source_path.exists():
                print(f"  SKIP {word}[{i}]: source not found: {source_path}")
                continue

            output_name = f'{word.lower()}_{vid_id}_{i:03d}.mp4'
            output_path = word_clip_dir / output_name

            if output_path.exists():
                print(f"  Already clipped: {output_name}")
                clip_count += 1
                continue

            print(f"  Clipping: {word.upper()} [{start}s - {end}s] from {vid_id}")

            cmd = [
                'ffmpeg',
                '-y',
                '-ss', str(start),
                '-i', str(source_path),
                '-t', str(end - start),
                '-vf', f'scale={target_res[0]}:{target_res[1]}:force_original_aspect_ratio=decrease,'
                       f'pad={target_res[0]}:{target_res[1]}:(ow-iw)/2:(oh-ih)/2,'
                       f'fps={target_fps}',
                '-an',  # no audio needed for pose extraction
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                str(output_path),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0 and output_path.exists():
                    print(f"    Saved: {output_path}")
                    clip_count += 1
                else:
                    stderr = result.stderr.decode()[-200:] if result.stderr else ''
                    print(f"    FAILED: {stderr}")
            except subprocess.TimeoutExpired:
                print(f"    Timeout clipping {source_path}")
            except FileNotFoundError:
                print("ERROR: ffmpeg not found. Install ffmpeg and add to PATH.")
                sys.exit(1)

    print(f"\nClipped {clip_count} sign videos to: {clip_dir}")
    print("\nNext steps:")
    print("  1. Review clipped videos for quality")
    print("  2. Run pose extraction on clipped videos to generate pickle files")
    print("  3. Add pickle files to the augmented pool")


def main():
    parser = argparse.ArgumentParser(description='ASL Video Crawler')
    parser.add_argument('--config', default=None,
                       help='Path to crawler config JSON')
    parser.add_argument('--search', action='store_true',
                       help='Search for candidate videos')
    parser.add_argument('--download', action='store_true',
                       help='Download annotated videos')
    parser.add_argument('--clip', action='store_true',
                       help='Clip signs from downloaded videos')
    parser.add_argument('--auto-annotate', action='store_true',
                       help='Auto-detect clip timestamps for downloaded videos')
    parser.add_argument('--all', action='store_true',
                       help='Run full pipeline (search + download + auto-annotate + clip)')
    parser.add_argument('--download-url', nargs=2, metavar=('URL', 'WORD'),
                       help='Download a specific URL for a word')
    args = parser.parse_args()

    # Find config
    base_dir = Path(__file__).parent
    config_path = Path(args.config) if args.config else base_dir / 'crawler_config.json'

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    ensure_dirs(config, base_dir)

    if args.download_url:
        url, word = args.download_url
        download_single_url(url, word, config, base_dir)
    elif args.search or args.all:
        do_search(config, base_dir)
    if args.download or args.all:
        do_download(config, base_dir)
    if args.auto_annotate or args.all:
        from auto_clip_detector import auto_annotate_config
        auto_annotate_config(config_path)
        config = load_config(config_path)  # reload after annotation
    if args.clip or args.all:
        do_clip(config, base_dir)

    if not any([args.search, args.download, args.clip, args.all,
                args.download_url, args.auto_annotate]):
        parser.print_help()
        print("\nExample workflow:")
        print("  1. python crawl_asl_videos.py --search")
        print("  2. python crawl_asl_videos.py --download")
        print("  3. python crawl_asl_videos.py --auto-annotate")
        print("  4. python crawl_asl_videos.py --clip")
        print("  OR: python crawl_asl_videos.py --all")


if __name__ == '__main__':
    main()
