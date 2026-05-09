"""
enrich_lyrics.py
----------------
Enriches a songs CSV with lyrics from lyrics.ovh.

Usage:
    # First pass — fast, parallel, single retry
    python enrich_lyrics.py --input songs.csv --output songs_with_lyrics.csv

    # Retry only rows that failed/weren't found
    python enrich_lyrics.py --input songs.csv --output songs_with_lyrics.csv --retry-misses

    # Tune workers and delay
    python enrich_lyrics.py --input songs.csv --output songs_with_lyrics.csv --workers 8 --delay 0.15

Expected input columns (case-insensitive):
    artist, title  (required)
    Any other columns are preserved as-is.

The script:
  - Parallel workers (default 5) — big speedup since bottleneck is network I/O
  - Thread-safe progress saving with a lock
  - Tunable --delay and --workers from the CLI
  - --retry-misses mode: re-attempts only previously empty/failed rows
  - Tries multiple artist/title variants per song to maximise recall
  - Logs which variant matched for post-hoc analysis
  - Prints a live ETA and summary at the end
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# ── Config (overridable via CLI) ──────────────────────────────────────────────

DEFAULT_WORKERS     = 5
DEFAULT_DELAY       = 0.2    # seconds between requests per worker
DEFAULT_TIMEOUT     = 8      # request timeout in seconds
DEFAULT_MAX_RETRIES = 1      # keep low on first pass; bump to 3 for --retry-misses
RETRY_DELAY         = 2      # seconds between network-error retries

BASE_URL = "https://api.lyrics.ovh/v1"

# ── Candidate generation ──────────────────────────────────────────────────────

def _strip_feat(s: str) -> str:
    return re.sub(r"\s*(feat\.?|ft\.?|featuring)\s+[^,\(\[\n]+", "", s, flags=re.I).strip()

def _extract_feat(s: str) -> str | None:
    m = re.search(r"(?:feat\.?|ft\.?|featuring)\s+([^,\(\[\n]+)", s, re.I)
    return m.group(1).strip() if m else None

def _strip_parens(s: str) -> str:
    """Remove (parens), [brackets], and '- Remix/Edit/Mix/...' dash suffixes."""
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)
    s = re.sub(
        r"\s*[-–—]\s*[^-–—]*(remix|edit|mix|version|dub|vip|bootleg|rework|flip|instrumental)\b.*",
        "", s, flags=re.I
    )
    return s.strip()

def _depunct(s: str) -> str:
    return re.sub(r"[^\w\s]", " ", s)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def candidate_queries(raw_artist: str, raw_title: str) -> list[tuple[str, str]]:
    a, t = _clean(raw_artist), _clean(raw_title)
    seen = set()

    def emit(artist, title):
        artist, title = _clean(artist), _clean(title)
        if artist and title and (artist, title) not in seen:
            seen.add((artist, title))
            return (artist, title)
        return None

    candidates = []

    def add(*pairs):
        for pair in pairs:
            if pair:
                candidates.append(pair)

    add(emit(a, t))

    t_no_feat   = _strip_feat(_strip_parens(t)) or _strip_feat(t)
    a_no_feat   = _strip_feat(a)
    t_no_parens = _strip_parens(t)

    add(emit(a, t_no_feat))
    add(emit(a_no_feat, t), emit(a_no_feat, t_no_feat))
    add(emit(a, t_no_parens), emit(a_no_feat, t_no_parens))

    feat_name = _extract_feat(a)
    if feat_name:
        add(emit(a_no_feat, f"{t} feat. {feat_name}"))
        add(emit(a_no_feat, f"{t_no_parens} feat. {feat_name}"))

    for ca, ct in list(candidates):
        add(emit(_depunct(ca), _depunct(ct)))
    for ca, ct in list(candidates):
        add(emit(ca.lower(), ct.lower()))

    return candidates


# ── Single URL attempt ────────────────────────────────────────────────────────

def _try_url(artist: str, title: str, timeout: int) -> tuple[str | None, bool]:
    url = f"{BASE_URL}/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("lyrics", "").strip(), False
        elif r.status_code == 404:
            return "", False
        elif r.status_code == 429:
            return None, True
        else:
            return None, False
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return None, True
    except requests.exceptions.RequestException:
        return None, False


# ── Fetch with candidate fallback ─────────────────────────────────────────────

def fetch_lyrics(
    artist: str,
    title: str,
    delay: float,
    max_retries: int,
    timeout: int,
) -> tuple[str, str | None]:
    candidates = candidate_queries(artist, title)

    for ca, ct in candidates:
        for attempt in range(1, max_retries + 1):
            lyrics, is_network_err = _try_url(ca, ct, timeout)

            if lyrics is not None:
                if lyrics:
                    label = "original" if (ca, ct) == (artist.strip(), title.strip()) else f"{ca} / {ct}"
                    return lyrics, label
                break  # clean 404 — try next candidate

            if is_network_err and attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)
            else:
                break

        time.sleep(delay)

    return "", None


# ── Progress helpers (thread-safe) ────────────────────────────────────────────

def load_progress(progress_file: Path) -> dict:
    if not progress_file.exists():
        return {}
    with open(progress_file) as f:
        data = json.load(f)
    needs_migration = any(not isinstance(v, dict) for v in data.values())
    if needs_migration:
        print("Migrating progress file from old format...")
    migrated = {}
    for k, v in data.items():
        if isinstance(v, dict):
            migrated[k] = v
        elif v == "__missing__" or v is None:
            migrated[k] = {"lyrics": None, "variant": None}
        else:
            migrated[k] = {"lyrics": v, "variant": "original"}
    return migrated


def save_progress(progress_file: Path, progress: dict, lock: threading.Lock) -> None:
    with lock:
        with open(progress_file, "w") as f:
            json.dump(progress, f, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich a songs CSV with lyrics from lyrics.ovh")
    parser.add_argument("--input",        "-i", required=True)
    parser.add_argument("--output",       "-o", required=True)
    parser.add_argument("--artist-col",   default=None)
    parser.add_argument("--title-col",    default=None)
    parser.add_argument("--workers",      type=int,   default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default {DEFAULT_WORKERS})")
    parser.add_argument("--delay",        type=float, default=DEFAULT_DELAY,
                        help=f"Seconds between requests per worker (default {DEFAULT_DELAY})")
    parser.add_argument("--max-retries",  type=int,   default=DEFAULT_MAX_RETRIES,
                        help=f"Retries per network error (default {DEFAULT_MAX_RETRIES}; use 3 with --retry-misses)")
    parser.add_argument("--retry-misses", action="store_true",
                        help="Re-attempt rows previously not found or errored")
    args = parser.parse_args()

    input_path    = Path(args.input)
    output_path   = Path(args.output)
    progress_file = output_path.with_suffix(".progress.json")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {input_path} …")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    col_map    = {c.lower(): c for c in df.columns}
    artist_col = (args.artist_col or col_map.get("artist") or col_map.get("artist_name")
                  or col_map.get("track_artist"))
    title_col  = (args.title_col  or col_map.get("title")  or col_map.get("song")
                  or col_map.get("track") or col_map.get("track_name"))

    if not artist_col or not title_col:
        print(f"ERROR: Could not detect artist/title columns. Found: {list(df.columns)}")
        print("Use --artist-col and --title-col to specify them manually.")
        return

    print(f"  Artist column : '{artist_col}'")
    print(f"  Title column  : '{title_col}'")
    print(f"  Rows          : {len(df)}")
    print(f"  Workers       : {args.workers}  |  Delay: {args.delay}s  |  Max retries: {args.max_retries}")

    # ── Load progress ─────────────────────────────────────────────────────────
    progress      = load_progress(progress_file)
    progress_lock = threading.Lock()

    if progress:
        print(f"\nResuming — {len(progress)} rows already in progress file.")

    # Decide which rows to process
    if args.retry_misses:
        print("Mode: retrying misses (empty or errored rows)")
        todo_indices = [
            i for i in df.index
            if progress.get(str(i), {}).get("lyrics") in ("", None)
        ]
        for i in todo_indices:
            progress.pop(str(i), None)
    else:
        todo_indices = [i for i in df.index if str(i) not in progress]

    total = len(df)
    print(f"  To fetch      : {len(todo_indices)}\n")

    if not todo_indices:
        print("Nothing to do — all rows already processed.")
    else:
        counters      = {"done": 0}
        counters_lock = threading.Lock()
        start_time    = time.time()

        def process_row(idx):
            row    = df.loc[idx]
            artist = str(row[artist_col])
            title  = str(row[title_col])

            lyrics, variant = fetch_lyrics(
                artist, title,
                delay=args.delay,
                max_retries=args.max_retries,
                timeout=DEFAULT_TIMEOUT,
            )

            entry = {"lyrics": lyrics if lyrics else "", "variant": variant}
            with progress_lock:
                progress[str(idx)] = entry
            save_progress(progress_file, progress, progress_lock)

            with counters_lock:
                counters["done"] += 1
                done_total = counters["done"]
                elapsed    = time.time() - start_time
                rate       = done_total / elapsed if elapsed > 0 else 0
                remaining  = (len(todo_indices) - done_total) / rate if rate > 0 else 0
                eta_str    = f"{int(remaining // 60)}m {int(remaining % 60)}s"

                status = "✓" if lyrics else "✗"
                flag   = " (fallback)" if variant and variant != "original" else ""
                print(f"  [{done_total}/{len(todo_indices)}] {status} {artist} — {title}{flag}  |  ETA {eta_str}")

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_row, idx): idx for idx in todo_indices}
                for future in as_completed(futures):
                    exc = future.exception()
                    if exc:
                        idx = futures[future]
                        print(f"  [!] Unhandled error for row {idx}: {exc}")
        except KeyboardInterrupt:
            print("\n\nInterrupted — progress saved. Re-run the same command to resume.")

    # ── Write output ──────────────────────────────────────────────────────────
    df["lyrics"]         = [progress.get(str(i), {}).get("lyrics") for i in df.index]
    df["lyrics_variant"] = [progress.get(str(i), {}).get("variant") for i in df.index]
    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched dataset → {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    done     = sum(1 for k in map(str, df.index) if k in progress)
    found    = sum(1 for v in progress.values() if v.get("lyrics"))
    missing  = sum(1 for v in progress.values() if v.get("lyrics") == "")
    errors   = sum(1 for v in progress.values() if v.get("lyrics") is None)
    fallback = sum(1 for v in progress.values() if v.get("variant") and v["variant"] != "original")

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Total rows      : {total}")
    print(f"  Processed       : {done}")
    print(f"  Found lyrics    : {found}  ({found/max(done,1)*100:.1f}% of processed)")
    print(f"    via original  : {found - fallback}")
    print(f"    via fallback  : {fallback}")
    print(f"  Not found       : {missing}")
    print(f"  Errors          : {errors}")
    print(f"  Overall coverage: {found/total*100:.1f}%")

    if done < total:
        print(f"\n  {total - done} rows still pending — re-run to continue.")
        print(f"  Tip: re-run with --retry-misses --max-retries 3 to chase down misses.")
    else:
        if progress_file.exists():
            progress_file.unlink()
        print("\nAll rows processed. Progress file removed.")


if __name__ == "__main__":
    main()