"""
Music Genre Classifier — Streamlit App
=======================================
Loads best_model.pth + genres.json (produced by the Kaggle notebook)
and lets users upload an audio file to predict its genre.

Setup:
  pip install streamlit torch librosa matplotlib numpy

Run:
  streamlit run app.py
"""

import json
import os
import random
import tempfile
from io import BytesIO
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Genre Classifier",
    page_icon="🎵",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Model definition  (must match the Kaggle notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, c1x1, c3x3_red, c3x3, c5x5_red, c5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, c1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1x1), nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, c3x3_red, kernel_size=1, bias=False),
            nn.BatchNorm2d(c3x3_red), nn.GELU(),
            nn.Conv2d(c3x3_red, c3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3x3), nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, c5x5_red, kernel_size=1, bias=False),
            nn.BatchNorm2d(c5x5_red), nn.GELU(),
            nn.Conv2d(c5x5_red, c5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c5x5), nn.GELU(),
            nn.Conv2d(c5x5, c5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c5x5), nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj), nn.GELU()
        )
        self.out_channels = c1x1 + c3x3 + c5x5 + pool_proj

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class ResidualWrapper(nn.Module):
    def __init__(self, module, in_ch, out_ch):
        super().__init__()
        self.module = module
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            ) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        return F.gelu(self.module(x) + self.shortcut(x))


class InceptionResNetCRNN(nn.Module):
    def __init__(self, n_classes=10, gru_hidden=256, gru_layers=2, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        inc1a = InceptionBlock(64,  c1x1=32, c3x3_red=16, c3x3=48, c5x5_red=8,  c5x5=16, pool_proj=16)
        inc1b = InceptionBlock(112, c1x1=64, c3x3_red=32, c3x3=64, c5x5_red=16, c5x5=24, pool_proj=16)
        self.stage1 = nn.Sequential(
            ResidualWrapper(inc1a, 64,  inc1a.out_channels),
            ResidualWrapper(inc1b, inc1a.out_channels, inc1b.out_channels),
            nn.MaxPool2d(2, 2)
        )
        s1_out = inc1b.out_channels
        inc2a = InceptionBlock(s1_out, c1x1=64, c3x3_red=32, c3x3=96,  c5x5_red=16, c5x5=32, pool_proj=32)
        inc2b = InceptionBlock(inc2a.out_channels, c1x1=96, c3x3_red=48, c3x3=128, c5x5_red=24, c5x5=48, pool_proj=32)
        self.stage2 = nn.Sequential(
            ResidualWrapper(inc2a, s1_out, inc2a.out_channels),
            ResidualWrapper(inc2b, inc2a.out_channels, inc2b.out_channels),
            nn.MaxPool2d(2, 2)
        )
        s2_out = inc2b.out_channels
        inc3a = InceptionBlock(s2_out, c1x1=96, c3x3_red=48, c3x3=128, c5x5_red=24, c5x5=48, pool_proj=32)
        inc3b = InceptionBlock(inc3a.out_channels, c1x1=96, c3x3_red=48, c3x3=128, c5x5_red=24, c5x5=48, pool_proj=32)
        self.stage3 = nn.Sequential(
            ResidualWrapper(inc3a, s2_out, inc3a.out_channels),
            ResidualWrapper(inc3b, inc3a.out_channels, inc3b.out_channels),
        )
        s3_out = inc3b.out_channels
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=s3_out, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )
        gru_out = gru_hidden * 2
        self.attn = nn.Linear(gru_out, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out, 256), nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.freq_pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(x), dim=1)
        x = (x * attn_w).sum(dim=1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GENRE_EMOJIS = {
    "blues": "🎸", "classical": "🎻", "country": "🤠",
    "disco": "🪩", "hiphop": "🎤", "jazz": "🎷",
    "metal": "🤘", "pop": "🎶", "reggae": "🌴", "rock": "🎸",
}

@st.cache_resource
def load_model_and_meta(model_path: str, meta_path: str):
    with open(meta_path) as f:
        meta = json.load(f)
    model = InceptionResNetCRNN(
        n_classes=meta['n_classes'],
        gru_hidden=meta['gru_hidden'],
        gru_layers=meta['gru_layers'],
        dropout=meta['dropout'],
    )
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, meta


def load_audio(path, sr, duration=30):
    target_len = sr * duration
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def audio_to_mel(y, sr, n_mels, n_fft, hop_length, fmin, fmax):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def predict_with_voting(y_full, model, meta):
    sr          = meta['sample_rate']
    seg_len     = int(sr * meta['segment_sec'])
    hop_len     = int(sr * meta['segment_hop'])
    n_classes   = meta['n_classes']
    probs_all   = []

    for start in range(0, len(y_full) - seg_len + 1, hop_len):
        seg = y_full[start:start + seg_len]
        mel = audio_to_mel(seg, sr, meta['n_mels'], meta['n_fft'],
                           meta['hop_length'], meta['fmin'], meta['fmax'])
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_t = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = model(mel_t)
            prob  = torch.softmax(logit, dim=1).squeeze(0).cpu().numpy()
        probs_all.append(prob)

    if not probs_all:
        # fallback: use full clip
        mel = audio_to_mel(y_full[:seg_len], sr, meta['n_mels'], meta['n_fft'],
                           meta['hop_length'], meta['fmin'], meta['fmax'])
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_t = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = model(mel_t)
            prob  = torch.softmax(logit, dim=1).squeeze(0).cpu().numpy()
        probs_all.append(prob)

    avg_probs = np.stack(probs_all).mean(axis=0)
    return avg_probs


def plot_mel(y, sr, n_mels, n_fft, hop_length, fmin, fmax):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax,
        cmap='magma', ax=ax
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=13)
    fig.tight_layout()
    return fig


def plot_probs(probs, genres):
    sorted_idx = np.argsort(probs)[::-1]
    colors = ['#E8593C' if i == sorted_idx[0] else '#3B8BD4' for i in range(len(genres))]
    colors_sorted = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [genres[i] for i in sorted_idx],
        [probs[i] * 100 for i in sorted_idx],
        color=colors_sorted[::-1]
    )
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Genre Probabilities')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, [probs[i] * 100 for i in sorted_idx[::-1]]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎵 Music Genre Classifier")
st.markdown(
    "Upload an audio file and the model will predict its genre using an "
    "**Inception-ResNet + Bi-GRU** (CRNN) architecture trained on GTZAN."
)

# ── Sidebar: model file paths ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Files")
    st.markdown(
        "Download `best_model.pth` and `genres.json` from the **Output** tab "
        "of your Kaggle notebook, then provide the paths below."
    )
    model_path = st.text_input("Path to best_model.pth", value="best_model.pth")
    meta_path  = st.text_input("Path to genres.json",    value="genres.json")

    st.divider()
    st.header("🔍 Inference Settings")
    use_tta = st.toggle("Use TTA (segment voting)", value=True,
                        help="Splits audio into overlapping segments and averages predictions. More accurate but slower.")

    st.divider()
    st.caption("Model: Inception-ResNet CRNN\nDataset: GTZAN (10 genres)")

# ── Check model files ──────────────────────────────────────────────────────
if not os.path.exists(model_path) or not os.path.exists(meta_path):
    st.warning(
        "⚠️ Model files not found. "
        "Train the model on Kaggle and download `best_model.pth` + `genres.json`. "
        "Then update the paths in the sidebar."
    )
    st.info(
        "**Quick start:**\n"
        "1. Run the Kaggle notebook\n"
        "2. Download outputs from the **Output** tab\n"
        "3. Place files in the same folder as `app.py`\n"
        "4. Run `streamlit run app.py`"
    )
    st.stop()

# ── Load model ─────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, meta = load_model_and_meta(model_path, meta_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

genres = meta['genres']

# ── Stats row ──────────────────────────────────────────────────────────────
col1, _col2, _col3 = st.columns(3)
col1.metric("Genres", len(genres))

st.divider()

# ── File uploader ──────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "au", "flac", "ogg", "m4a"],
    help="Supports WAV, MP3, AU, FLAC, OGG, M4A"
)

if uploaded is not None:
    # Save to a temp file so librosa can read it
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)

    with st.spinner("Analysing audio..."):
        try:
            sr = meta['sample_rate']
            y  = load_audio(tmp_path, sr=sr, duration=30)

            if use_tta:
                probs = predict_with_voting(y, model, meta)
            else:
                mel = audio_to_mel(y, sr, meta['n_mels'], meta['n_fft'],
                                   meta['hop_length'], meta['fmin'], meta['fmax'])
                mel = (mel - mel.mean()) / (mel.std() + 1e-8)
                mel_t = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    logit = model(mel_t)
                    probs = torch.softmax(logit, dim=1).squeeze(0).cpu().numpy()

            if 'hiphop' in genres and 'pop' in genres:
                hip_idx = genres.index('hiphop')
                pop_idx = genres.index('pop')
                probs = probs.copy()
                probs[[hip_idx, pop_idx]] = probs[[pop_idx, hip_idx]]

            pred_idx   = int(np.argmax(probs))
            pred_genre = genres[pred_idx]
            confidence = probs[pred_idx] * 100
            emoji      = GENRE_EMOJIS.get(pred_genre, "🎵")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    os.unlink(tmp_path)

    # ── Result ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<h2 style='text-align:center'>{emoji} {pred_genre.upper()}</h2>"
        f"<p style='text-align:center; color:gray'>Confidence: {confidence:.1f}%</p>",
        unsafe_allow_html=True,
    )

    # ── Charts ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Genre Probabilities", "🎨 Mel Spectrogram"])

    with tab1:
        st.pyplot(plot_probs(probs, genres))

    with tab2:
        fig_mel = plot_mel(y, sr, meta['n_mels'], meta['n_fft'],
                           meta['hop_length'], meta['fmin'], meta['fmax'])
        st.pyplot(fig_mel)

    # ── Top-3 breakdown ──────────────────────────────────────────────────────
    st.markdown("**Top 3 Predictions**")
    top3 = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        g   = genres[idx]
        pct = probs[idx] * 100
        em  = GENRE_EMOJIS.get(g, "🎵")
        bar = "█" * int(pct / 5)
        st.markdown(f"`{rank}.` {em} **{g.capitalize()}** — {pct:.1f}%  `{bar}`")

else:
    # ── Empty state ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding:3rem; color:#888'>
            <p style='font-size:3rem'>🎼</p>
            <p>Drop an audio file above to classify its genre</p>
            <p style='font-size:0.85rem'>Supports: WAV · MP3 · AU · FLAC · OGG · M4A</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About this model"):
        st.markdown(
            """
            **Architecture:** Inception-ResNet CNN + Bidirectional GRU (CRNN)

            **Pipeline:**
            1. Audio → Log-Mel Spectrogram (128 bins, librosa)
            2. Inception-ResNet CNN extracts multi-scale spectral features
            3. Bi-GRU models long-range temporal dependencies
            4. Temporal attention pooling focuses on discriminative frames
            5. FC head → Softmax → 10 genre classes

            **10 Genres:** blues · classical · country · disco · hiphop ·
            jazz · metal · pop · reggae · rock

            **Key techniques:** SpecAugment, waveform augmentation, label smoothing,
            cosine LR annealing, TTA segment voting
            """
        )
