"""
STEP 1 - Collect and store lyrics
Uses lyricsgenius to fetch lyrics and saves them as JSON files.
Install: pip install lyricsgenius

Requires: GENIUS_TOKEN in .env file
  → https://genius.com/api-clients (create an account, make an API client)
"""

import os
import json
import lyricsgenius
from dotenv import load_dotenv
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")
OUTPUT_DIR = "data/songs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Song list ─────────────────────────────────────────────────────────────────
# Add or remove songs here. Format: ("Song Title", "Artist Name")
SONGS = [
    ("Bohemian Rhapsody", "Queen"),
    ("Alright", "Kendrick Lamar"),
    ("Someone Like You", "Adele"),
    ("Lose Yourself", "Eminem"),
    ("Imagine", "John Lennon"),
    ("Bad Guy", "Billie Eilish"),
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Shape of You", "Ed Sheeran"),
    ("God's Plan", "Drake"),
    ("Creep", "Radiohead"),
    ("Espresso", "Sabrina Carpenter"),
    ("Chiquitita", "ABBA"),
    ("Yesterday", "The Beatles"),
    ("Blinding Lights", "The Weeknd"),
    ("I Gotta Feeling", "Black Eyed Peas"),
]

# ── Fetch and save ────────────────────────────────────────────────────────────
genius = lyricsgenius.Genius(
    GENIUS_TOKEN,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
)

for title, artist in SONGS:
    filename = f"{artist.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"Already saved: {title} - {artist}")
        continue

    print(f"Fetching: {title} - {artist}...")
    try:
        song = genius.search_song(title, artist)
        if song:
            data = {
                "title": song.title,
                "artist": song.artist,
                "lyrics": song.lyrics,
                "filename": filename,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Saved to {filepath}")
        else:
            print(f"  ✗ Not found: {title}")
    except Exception as e:
        print(f"  ✗ Error fetching {title}: {e}")

print(f"\nDone! Songs saved to '{OUTPUT_DIR}/'")
