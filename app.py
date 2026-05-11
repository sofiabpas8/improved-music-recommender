"""
app.py — Streamlit interface for the Music Recommender Arena.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
from huggingface_hub import snapshot_download


# ── Data download (runs before anything else) ─────────────────────────────────
def download_data():
    if os.path.exists("data/songs_clean.csv"):
        return

    os.makedirs("data", exist_ok=True)
    snapshot_download(
        repo_id="cosita2000/music-recommender-data",
        repo_type="dataset",
        local_dir="data/",
    )


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Music Recommender Arena",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e8e4dc;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(139, 92, 246, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(236, 72, 153, 0.05) 0%, transparent 50%),
        #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2.5rem 3rem 4rem 3rem; max-width: 1100px; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Playfair Display', serif; }
p, label, div, span, input, button {
    font-family: 'DM Mono', monospace;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid rgba(232, 228, 220, 0.08);
    margin-bottom: 2.5rem;
}
.app-header h1 {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e8e4dc;
    margin: 0;
    line-height: 1.1;
}
.app-header .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: rgba(232, 228, 220, 0.4);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.8rem;
}

/* ── Search area ── */
.search-container {
    background: rgba(232, 228, 220, 0.03);
    border: 1px solid rgba(232, 228, 220, 0.1);
    border-radius: 2px;
    padding: 2rem 2.5rem;
    margin-bottom: 2.5rem;
}

/* ── Streamlit input overrides ── */
[data-testid="stTextInput"] input {
    background: rgba(10, 10, 15, 0.8) !important;
    border: 1px solid rgba(232, 228, 220, 0.15) !important;
    border-radius: 2px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.1) !important;
}
[data-testid="stTextInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(232, 228, 220, 0.45) !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    background: #e8e4dc !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
[data-testid="stButton"] button:hover {
    background: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(232, 228, 220, 0.15) !important;
}

/* ── Result cards ── */
.result-card {
    background: rgba(232, 228, 220, 0.03);
    border: 1px solid rgba(232, 228, 220, 0.08);
    border-radius: 2px;
    padding: 1.8rem 2rem;
    height: 100%;
    transition: border-color 0.3s;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.card-knn::before      { background: #f59e0b; }
.card-minilm::before   { background: #8b5cf6; }
.card-mpnet::before    { background: #ec4899; }

.result-card:hover { border-color: rgba(232, 228, 220, 0.18); }

.model-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(232, 228, 220, 0.07);
}
.label-knn    { color: #f59e0b; }
.label-minilm { color: #8b5cf6; }
.label-mpnet  { color: #ec4899; }

.rec-song {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #e8e4dc;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.rec-artist {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: rgba(232, 228, 220, 0.45);
    letter-spacing: 0.05em;
    margin-bottom: 1.4rem;
}

/* ── Score pills ── */
.scores {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.4rem;
}
.score-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.3rem 0.7rem;
    border-radius: 2px;
    background: rgba(232, 228, 220, 0.05);
    border: 1px solid rgba(232, 228, 220, 0.1);
    color: rgba(232, 228, 220, 0.6);
}
.score-pill span {
    color: #e8e4dc;
    font-weight: 500;
}

/* ── Explanation ── */
.explanation {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.75;
    color: rgba(232, 228, 220, 0.65);
    border-left: 2px solid rgba(232, 228, 220, 0.1);
    padding-left: 1rem;
    margin-top: 0.5rem;
}

/* ── Error state ── */
.error-box {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 2px;
    padding: 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #fca5a5;
    margin-top: 1rem;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(232, 228, 220, 0.06);
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: rgba(232, 228, 220, 0.2);
    letter-spacing: 0.1em;
}

/* ── Column gap ── */
[data-testid="column"] { padding: 0 0.6rem; }
</style>
""", unsafe_allow_html=True)


# ── Load models (cached so they only run once) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models():
    from recommender import load_models
    load_models(use_llm=True)
    return True

@st.cache_data(show_spinner=False)
def get_song_list():
    """Load unique song titles for autocomplete hint."""
    df = pd.read_csv("data/songs_clean.csv", usecols=["track_name", "track_artist"])
    df = df.drop_duplicates(subset=["track_name", "track_artist"])
    return df.sort_values("track_name")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Music Recommender Arena</h1>
    <div class="subtitle">k-NN baseline &nbsp;·&nbsp; RAG MiniLM &nbsp;·&nbsp; RAG MPNet</div>
</div>
""", unsafe_allow_html=True)


# ── Data download + model loading ─────────────────────────────────────────────
with st.spinner("Downloading data and loading models — this takes about a minute on first run..."):
    try:
        download_data()
        get_models()
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠ Could not load models: {e}</div>', unsafe_allow_html=True)
        st.stop()


# ── Search form ───────────────────────────────────────────────────────────────
st.markdown('<div class="search-container">', unsafe_allow_html=True)

col_song, col_artist, col_btn = st.columns([3, 2, 1], gap="medium")

with col_song:
    song_input = st.text_input(
        "Song title",
        placeholder="e.g. Bohemian Rhapsody",
        key="song",
    )

with col_artist:
    artist_input = st.text_input(
        "Artist (optional)",
        placeholder="e.g. Queen",
        key="artist",
    )

with col_btn:
    st.markdown("<div style='height: 1.95rem'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Find similar →")

st.markdown('</div>', unsafe_allow_html=True)


# ── Results ───────────────────────────────────────────────────────────────────
if search_clicked:
    if not song_input.strip():
        st.markdown('<div class="error-box">Please enter a song title.</div>', unsafe_allow_html=True)
    else:
        from recommender import recommend

        with st.spinner(f'Searching for songs similar to "{song_input}"...'):
            try:
                results = recommend(song_input.strip(), artist_input.strip())
            except ValueError as e:
                st.markdown(
                    f'<div class="error-box">Song not found in the dataset.<br>'
                    f'<small style="opacity:0.6">{e}</small></div>',
                    unsafe_allow_html=True,
                )
                results = None
            except Exception as e:
                st.markdown(
                    f'<div class="error-box">Something went wrong: {e}</div>',
                    unsafe_allow_html=True,
                )
                results = None

        if results:
            knn    = results.get("knn", {})
            minilm = results.get("rag_minilm", {})
            mpnet  = results.get("rag_mpnet", {})

            cards = [
                ("k-NN Baseline",      "knn",    "card-knn",    "label-knn",    knn),
                ("RAG · MiniLM-L6-v2", "minilm", "card-minilm", "label-minilm", minilm),
                ("RAG · MPNet",        "mpnet",  "card-mpnet",  "label-mpnet",  mpnet),
            ]

            cols = st.columns(3, gap="medium")

            for col, (label, key, card_cls, label_cls, data) in zip(cols, cards):
                with col:
                    if not data or "error" in data:
                        error_msg = data.get("error", "No result returned.") if data else "No result."
                        st.markdown(f"""
                        <div class="result-card {card_cls}">
                            <div class="model-label {label_cls}">{label}</div>
                            <div style="color: rgba(232,228,220,0.35); font-size: 0.8rem;">
                                {error_msg}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        continue

                    rec_song    = data.get("recommended_song",   "—")
                    rec_artist  = data.get("recommended_artist", "—")
                    audio_sim   = data.get("audio_similarity",   None)
                    text_sim    = data.get("text_similarity",    None)
                    combined    = data.get("combined_score",     None)
                    explanation = data.get("explanation",        "")

                    def pill(name, val):
                        if val is None:
                            return ""
                        return f'<div class="score-pill">{name}&nbsp;<span>{val:.3f}</span></div>'

                    scores_html = (
                        pill("audio", audio_sim)
                        + pill("text",  text_sim)
                        + pill("score", combined)
                    )

                    explanation_html = (
                        f'<div class="explanation">{explanation}</div>'
                        if explanation else ""
                    )

                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                        <div class="model-label {label_cls}">{label}</div>
                        <div class="rec-song">{rec_song}</div>
                        <div class="rec-artist">{rec_artist}</div>
                        <div class="scores">{scores_html}</div>
                        {explanation_html}
                    </div>
                    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Music Recommender Arena &nbsp;·&nbsp; k-NN vs RAG comparison
</div>
""", unsafe_allow_html=True)