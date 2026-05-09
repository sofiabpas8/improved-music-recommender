"""
enrich_lyrics.py
----------------
Enriches a songs CSV with lyrics from lyrics.ovh.

Usage:
    python enrich_lyrics.py --input songs.csv --output songs_with_lyrics.csv

Expected input columns (case-insensitive):
    artist, title  (required)
    Any other columns are preserved as-is.

The script:
  - Saves progress after every row (safe to Ctrl+C and resume)
  - Retries failed rows up to MAX_RETRIES times with backoff on network errors
  - Tries multiple artist/title variants per song to maximise recall
    (handles feat., remixes, punctuation differences, token placement, etc.)
  - Logs which variant matched for post-hoc analysis
  - Prints a summary at the end
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL     = "https://api.lyrics.ovh/v1"
DELAY        = 0.6    # seconds between requests (be polite to the free API)
TIMEOUT      = 8      # request timeout in seconds
MAX_RETRIES  = 3      # per-row retry attempts on transient network errors
RETRY_DELAY  = 3      # seconds between retries

# ── Candidate generation ──────────────────────────────────────────────────────

def _strip_feat(s: str) -> str:
    """Remove 'feat. X', 'ft. X', 'featuring X' from a string."""
    return re.sub(
        r"\s*(feat\.?|ft\.?|featuring)\s+[^,\(\[\n]+", "", s, flags=re.I
    ).strip()

def _extract_feat(s: str) -> str | None:
    """Return the featured artist string if present, else None."""
    m = re.search(r"(?:feat\.?|ft\.?|featuring)\s+([^,\(\[\n]+)", s, re.I)
    return m.group(1).strip() if m else None

def _strip_parens(s: str) -> str:
    """Remove parenthesised/bracketed suffixes AND dash-separated remix/edit/version suffixes.
    Examples:
      'Song (feat. X) - Marnik Remix'  -> 'Song'
      'Song [Radio Edit]'              -> 'Song'
      'Song - Steve Void Remix'        -> 'Song'
    """
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s)
    s = re.sub(
        r"\s*[-–—]\s*[^-–—]*(remix|edit|mix|version|dub|vip|bootleg|rework|flip|instrumental)\b.*",
        "", s, flags=re.I
    )
    return s.strip()

def _depunct(s: str) -> str:
    """Strip punctuation and normalise whitespace."""
    return re.sub(r"[^\w\s]", " ", s)

def _clean(s: str) -> str:
    """Basic normalisation: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", s).strip()

def candidate_queries(raw_artist: str, raw_title: str):
    """
    Yield (artist, title) string pairs to try, best-first.
    Stops at the first hit in the caller — later variants are only tried on misses.
    """
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

    # 1. Original as-is
    add(emit(a, t))

    # 2. Strip feat from title only (common: "Song (feat. X)")
    t_no_feat = _strip_feat(_strip_parens(t)) or _strip_feat(t)
    add(emit(a, t_no_feat))

    # 3. Strip feat from artist only (common: "Artist feat. X")
    a_no_feat = _strip_feat(a)
    add(emit(a_no_feat, t), emit(a_no_feat, t_no_feat))

    # 4. Strip all parenthesised content from title (remixes, edits, remaster…)
    t_no_parens = _strip_parens(t)
    add(emit(a, t_no_parens), emit(a_no_feat, t_no_parens))

    # 5. If feat is in artist, try appending featured artist to title instead
    feat_name = _extract_feat(a)
    if feat_name:
        add(emit(a_no_feat, f"{t} feat. {feat_name}"))
        add(emit(a_no_feat, f"{t_no_parens} feat. {feat_name}"))

    # 6. Depunctuated versions of the best candidates so far
    for ca, ct in list(candidates):
        add(emit(_depunct(ca), _depunct(ct)))

    # 7. Lowercase everything (some API entries are inconsistently cased)
    for ca, ct in list(candidates):
        add(emit(ca.lower(), ct.lower()))

    return candidates


# ── Single URL attempt ────────────────────────────────────────────────────────

def _try_url(artist: str, title: str) -> tuple[str | None, bool]:
    """
    Make one HTTP request for (artist, title).
    Returns (lyrics_or_empty, is_network_error).
      - (lyrics, False)  → hit
      - ("",    False)   → clean 404 miss
      - (None,  True)    → network/timeout error worth retrying
      - (None,  False)   → non-retryable HTTP error
    """
    url = f"{BASE_URL}/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("lyrics", "").strip(), False
        elif r.status_code == 404:
            return "", False
        elif r.status_code == 429:
            return None, True   # rate-limited → retry
        else:
            return None, False  # unexpected status, don't retry
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return None, True       # transient network issue → retry
    except requests.exceptions.RequestException:
        return None, False


# ── Main fetch with candidate fallback ───────────────────────────────────────

def fetch_lyrics(artist: str, title: str) -> tuple[str | None, str | None]:
    """
    Try each candidate (artist, title) variant in turn.
    Returns (lyrics, matched_variant_label) where:
      - lyrics is the lyrics string (may be empty), or None on total failure
      - matched_variant_label describes which variant worked (for logging)
    """
    candidates = candidate_queries(artist, title)

    for ca, ct in candidates:
        for attempt in range(1, MAX_RETRIES + 1):
            lyrics, is_network_err = _try_url(ca, ct)

            if lyrics is not None:                  # hit or clean 404
                if lyrics:
                    label = f"{ca} / {ct}" if (ca, ct) != (artist.strip(), title.strip()) else "original"
                    return lyrics, label
                break                               # clean 404 for this variant, try next

            if is_network_err:
                wait = RETRY_DELAY * attempt
                print(f"    Network error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                break                               # non-retryable, move to next variant

        time.sleep(DELAY)

    return "", None  # all variants exhausted


# ── Progress file ─────────────────────────────────────────────────────────────

def load_progress(progress_file: Path) -> dict:
    """Load previously saved progress, migrating old string-value format if needed."""
    if not progress_file.exists():
        return {}
    with open(progress_file) as f:
        data = json.load(f)
    # Migrate old format: {idx: lyrics_string} -> {idx: {"lyrics": ..., "variant": ...}}
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


def save_progress(progress_file: Path, progress: dict) -> None:
    with open(progress_file, "w") as f:
        json.dump(progress, f, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enrich a songs CSV with lyrics from lyrics.ovh")
    parser.add_argument("--input",      "-i", required=True,  help="Input CSV path")
    parser.add_argument("--output",     "-o", required=True,  help="Output CSV path")
    parser.add_argument("--artist-col", default=None, help="Artist column name (auto-detected if omitted)")
    parser.add_argument("--title-col",  default=None, help="Title column name (auto-detected if omitted)")
    args = parser.parse_args()

    input_path    = Path(args.input)
    output_path   = Path(args.output)
    progress_file = output_path.with_suffix(".progress.json")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading {input_path} …")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    col_map = {c.lower(): c for c in df.columns}
    artist_col = args.artist_col or col_map.get("artist") or col_map.get("artist_name")
    title_col  = args.title_col  or col_map.get("title")  or col_map.get("song") or col_map.get("track")

    if not artist_col or not title_col:
        print(f"ERROR: Could not detect artist/title columns. Found: {list(df.columns)}")
        print("Use --artist-col and --title-col to specify them manually.")
        return

    print(f"  Artist column : '{artist_col}'")
    print(f"  Title column  : '{title_col}'")
    print(f"  Rows          : {len(df)}\n")

    # ── Load progress ─────────────────────────────────────────────────────────
    progress = load_progress(progress_file)
    if progress:
        print(f"Resuming — {len(progress)} rows already fetched.\n")

    # Running counters (initialise from saved progress)
    found   = sum(1 for v in progress.values() if v.get("lyrics"))
    missing = sum(1 for v in progress.values() if v.get("lyrics") == "")
    errors  = sum(1 for v in progress.values() if v.get("lyrics") is None)

    try:
        for idx, row in df.iterrows():
            key = str(idx)
            if key in progress:
                continue

            artist = str(row[artist_col])
            title  = str(row[title_col])
            print(f"[{idx+1}/{len(df)}] {artist} — {title}")

            lyrics, variant = fetch_lyrics(artist, title)

            if lyrics is None:
                progress[key] = {"lyrics": None, "variant": None}
                errors += 1
                print("    → error")
            elif lyrics == "":
                progress[key] = {"lyrics": "", "variant": None}
                missing += 1
                print("    → not found")
            else:
                progress[key] = {"lyrics": lyrics, "variant": variant}
                found += 1
                preview = lyrics[:70].replace("\n", " ")
                flag = " (via fallback)" if variant != "original" else ""
                print(f"    → ✓ {len(lyrics)} chars{flag}  \"{preview}…\"")
                if variant and variant != "original":
                    print(f"       matched as: {variant}")

            save_progress(progress_file, progress)
            time.sleep(DELAY)

    except KeyboardInterrupt:
        print("\n\nInterrupted — progress saved. Re-run the same command to resume.")

    # ── Write output ──────────────────────────────────────────────────────────
    df["lyrics"]         = [progress.get(str(i), {}).get("lyrics") for i in df.index]
    df["lyrics_variant"] = [progress.get(str(i), {}).get("variant") for i in df.index]

    df.to_csv(output_path, index=False)
    print(f"\nSaved enriched dataset → {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(df)
    done     = sum(1 for k in map(str, df.index) if k in progress)
    fallback = sum(1 for v in progress.values() if v.get("variant") and v["variant"] != "original")

    print(f"\n── Summary ──────────────────────────")
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
    else:
        if progress_file.exists():
            progress_file.unlink()
        print("\nAll rows processed. Progress file removed.")


if __name__ == "__main__":
    main()