import argparse
import gc
import json
import os
import struct
import sys
import time
import urllib.request
import wave

import numpy as np
import torch
import torch.nn.functional as F

SUBSPLIT_VERSION = "3"

KARA_CKPT_NAME = "bs_roformer_karaoke_frazer_becruily.ckpt"
KARA_CKPT_URL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/bs_roformer_karaoke_frazer_becruily.ckpt"
KARA_MIN_BYTES = 180_000_000

KARA_PARAMS = dict(
    dim=256,
    depth=12,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64,
    heads=8,
    attn_dropout=0.0,
    ff_dropout=0.0,
    flash_attn=True,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=512,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    mlp_expansion_factor=4,
)

DRUMSEP_CKPT_NAME = "drumsep.th"
DRUMSEP_CKPT_URL = "https://huggingface.co/Eddycrack864/Drumsep/resolve/main/modelo_final.th"
DRUMSEP_MIN_BYTES = 140_000_000


def emit(**kwargs):
    print(json.dumps(kwargs), flush=True)


def fail(message):
    emit(type="error", message=str(message))
    sys.exit(1)


def default_models_dir():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~\\AppData\\Roaming"))
        return os.path.join(base, "stemkit", "models")
    elif sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/stemkit/models")
    else:
        return os.path.expanduser("~/.config/stemkit/models")


def download_model_file(url, dest, label, min_bytes=10_000_000):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return dest
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    start = 0
    headers = {"User-Agent": "stemkit"}
    if os.path.exists(part):
        start = os.path.getsize(part)
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    last_emit = 0.0
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            total = int(res.headers.get("Content-Length", 0)) + (
                start if res.status == 206 else 0
            )
            mode = "ab" if res.status == 206 and start > 0 else "wb"
            with open(part, mode) as f:
                while True:
                    chunk = res.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    now = time.time()
                    if now - last_emit >= 0.8 and total > 0:
                        pct = int(f.tell() / total * 100)
                        emit(
                            type="progress",
                            pct=min(45, 10 + int(pct * 0.35)),
                            message=f"Downloading {label} ({total // (1024 * 1024)}MB): {pct}%",
                        )
                        last_emit = now
    except Exception as e:
        raise RuntimeError(f"{label} download failed: {e}")
    if os.path.getsize(part) < min_bytes:
        raise RuntimeError(f"{label} download incomplete ({os.path.getsize(part)} bytes)")
    os.replace(part, dest)
    return dest


def load_wav(path):
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.T, sr
    except Exception:
        pass

    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except Exception as e:
        fail(f"cannot read wav {path}: {e}")
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        fail(f"unsupported sample width {width}")
    if channels == 0:
        fail("empty wav")
    return audio.reshape(-1, channels).T, sr


def save_wav_f32(path, data, sr):
    """Write a 32-bit float wav (fmt tag 3); values above 1.0 are preserved"""
    channels, _ = data.shape
    block = np.ascontiguousarray(data.T, dtype="<f4")
    size = block.nbytes
    block_align = channels * 4
    header = b"RIFF" + struct.pack("<I", 36 + size) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 3, channels, sr, sr * block_align, block_align, 32
    )
    header += b"data" + struct.pack("<I", size)
    with open(path, "wb") as f:
        f.write(header)
        block.tofile(f)


def smooth_2d(tensor, kf=5, kt=9):
    """2D average smoothing on (F, T) mask tensor"""
    x = tensor.unsqueeze(0).unsqueeze(0)
    pad_f = kf // 2
    pad_t = kt // 2
    x = F.pad(x, (pad_t, pad_t, pad_f, pad_f), mode="reflect")
    x = F.avg_pool2d(x, kernel_size=(kf, kt), stride=1)
    return x.squeeze(0).squeeze(0)


def median_filter_time(mag, kt=25, chunk_f=256):
    """True 1D median filter along time axis (dim=1) in chunks to keep RAM low"""
    F_bins, T_frames = mag.shape
    pad_t = kt // 2
    padded = F.pad(mag.unsqueeze(0).unsqueeze(0), (pad_t, pad_t, 0, 0), mode="reflect").squeeze(0).squeeze(0)
    out = torch.empty_like(mag)
    for f0 in range(0, F_bins, chunk_f):
        f1 = min(F_bins, f0 + chunk_f)
        sub = padded[f0:f1, :].unfold(dimension=1, size=kt, step=1)
        out[f0:f1, :] = sub.median(dim=-1).values
    return out


def median_filter_freq(mag, kf=25, chunk_t=1024):
    """True 1D median filter along frequency axis (dim=0) in chunks to keep RAM low"""
    F_bins, T_frames = mag.shape
    pad_f = kf // 2
    padded = F.pad(mag.unsqueeze(0).unsqueeze(0), (0, 0, pad_f, pad_f), mode="reflect").squeeze(0).squeeze(0)
    out = torch.empty_like(mag)
    for t0 in range(0, T_frames, chunk_t):
        t1 = min(T_frames, t0 + chunk_t)
        sub = padded[:, t0:t1].unfold(dimension=0, size=kf, step=1)
        out[:, t0:t1] = sub.median(dim=-1).values
    return out


def compute_hpss_masks(mag, kf=27, kt=27, power=6.0):
    harm = median_filter_time(mag, kt=kt)
    perc = median_filter_freq(mag, kf=kf)
    h_p = harm.pow(power)
    p_p = perc.pow(power)
    denom = (h_p + p_p).clamp(min=1e-12)
    return h_p / denom, p_p / denom, harm, perc


def split_vocals_ai(audio_np, sr, device, models_dir):
    """
    AI Neural Network Lead vs. Backing Vocal / Duet Separator using BS-RoFormer Karaoke.
    Separates lead singer (e.g. Anthony Kiedis / 1a Voz) from backing vocals & duets
    (e.g. John Frusciante / 2a Voz / Coro), even when both are panned center.
    """
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    ckpt_path = os.path.join(models_dir, KARA_CKPT_NAME)
    download_model_file(KARA_CKPT_URL, ckpt_path, "AI Duet/Karaoke model", KARA_MIN_BYTES)

    from models.bs_roformer.bs_roformer import BSRoformer

    # Pascal GPUs (GTX 10-series, compute capability 6.x) have 1:64 throttled FP16
    # and slow SDPA math fallback; native FP32 + cuBLAS matmul is >5x faster!
    if device == "cuda" and torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_half = cap_major >= 7
        use_flash = cap_major >= 8
    elif device == "mps":
        use_half = True
        use_flash = True
    else:
        use_half = False
        use_flash = False

    emit(type="progress", pct=48, message=f"Loading AI Duet Vocal model on {device}")
    params = dict(KARA_PARAMS, flash_attn=use_flash)
    model = BSRoformer(**params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    gc.collect()
    model.to(device)
    if use_half:
        model.half()
        model.band_split.to(torch.float32)
    model.eval()

    mix = torch.from_numpy(audio_np).to(device)
    chunk_size = 176400  # 4s at 44.1kHz (fastest throughput on CUDA)
    fade_size = 22050    # 0.5s crossfade
    step = chunk_size - fade_size
    border = fade_size
    length_init = mix.shape[-1]

    if length_init > 2 * border and border > 0:
        mix_pad = F.pad(mix, (border, border), mode="reflect")
    else:
        mix_pad = mix

    window = torch.ones(chunk_size, device=device)
    window[:fade_size] = torch.linspace(0, 1, fade_size, device=device)
    window[-fade_size:] = torch.linspace(1, 0, fade_size, device=device)

    result = torch.zeros_like(mix_pad)
    counter = torch.zeros_like(mix_pad)

    with torch.inference_mode():
        i = 0
        total_len = mix_pad.shape[-1]
        while i < total_len:
            part = mix_pad[:, i : i + chunk_size]
            chunk_len = part.shape[-1]
            win = window[:chunk_len].clone()
            if i == 0:
                win[:fade_size] = 1
            elif i + step >= total_len:
                win[-fade_size:] = 1

            # Skip neural inference on silent instrumental sections in vocals.wav
            if part.abs().max().item() < 0.004:
                counter[:, i : i + chunk_len] += win
                i += step
                pct = 50 + int(min(1.0, i / total_len) * 35)
                emit(type="progress", pct=pct, message="AI separating Lead vs Backing Vocals")
                continue

            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part_padded = F.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)
            if use_half:
                part_padded = part_padded.to(torch.float16)
            x = model(part_padded.unsqueeze(0))[0, :, :chunk_len].float()
            result[:, i : i + chunk_len] += x * win
            counter[:, i : i + chunk_len] += win
            i += step
            pct = 50 + int(min(1.0, i / total_len) * 35)
            emit(type="progress", pct=pct, message="AI separating Lead vs Backing Vocals")

        lead = result / counter.clamp(min=1e-8)
        if length_init > 2 * border and border > 0:
            lead = lead[:, border:-border]

        # Spectral residual refinement so backing_vocals has zero faint lead breath bleed
        n_fft = 4096
        hop = 1024
        stft_win = torch.hann_window(n_fft, device=device)
        S_mix = torch.stft(mix, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=stft_win, return_complex=True)
        S_lead = torch.stft(lead, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=stft_win, return_complex=True)
        S_back_raw = S_mix - S_lead

        # Soft Wiener mask to suppress any tiny residual lead vocal ghosts in backing stem
        mag_lead = S_lead.abs().mean(dim=0)
        mag_back = S_back_raw.abs().mean(dim=0)
        back_ratio = mag_back.pow(2.5) / (mag_back.pow(2.5) + (mag_lead * 0.45).pow(2.5) + 1e-12)
        back_gate = smooth_2d(back_ratio, kf=3, kt=5)
        S_back = S_back_raw * back_gate.unsqueeze(0)
        backing = torch.istft(S_back, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=stft_win, length=length_init)

    lead_np = lead.cpu().numpy()
    backing_np = backing.cpu().numpy()
    del model, mix, mix_pad, result, counter, lead, backing
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "lead_vocals": lead_np,
        "backing_vocals": backing_np,
    }


def split_drums_ai(audio_np, sr, device, models_dir):
    """
    AI Neural Network Drum Kit Separator (DrumSep HDemucs).
    Separates drums.wav into:
    - drums_kick (Bumbo)
    - drums_snare (Caixa + Tons)
    - drums_cymbals (Pratos + Chimbal)
    """
    ckpt_path = os.path.join(models_dir, DRUMSEP_CKPT_NAME)
    download_model_file(DRUMSEP_CKPT_URL, ckpt_path, "AI DrumSep model", DRUMSEP_MIN_BYTES)

    emit(type="progress", pct=50, message=f"Loading AI DrumSep model on {device}")
    _torch_load = torch.load

    def _load_any(*a, **k):
        k.setdefault("weights_only", False)
        return _torch_load(*a, **k)

    torch.load = _load_any
    from demucs.apply import apply_model
    from demucs.states import load_model

    pkg = torch.load(ckpt_path, map_location="cpu")
    if isinstance(pkg, dict) and "klass" in pkg:
        model = load_model(pkg)
    elif hasattr(pkg, "eval"):
        model = pkg
    else:
        raise RuntimeError("Unsupported drumsep checkpoint format")

    model.to(device)
    model.eval()

    mix = torch.from_numpy(audio_np).to(device).unsqueeze(0)
    emit(type="progress", pct=60, message="AI separating Kick, Snare & Cymbals")
    with torch.inference_mode():
        sources = apply_model(model, mix, device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]

    # DrumSep sources order: ['bombo', 'redoblante', 'platillos', 'toms'] (or kick, snare, cymbals, toms)
    src_np = sources.cpu().numpy()
    del model, mix, sources
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    kick = src_np[0]
    snare_and_toms = src_np[1] + (src_np[3] if src_np.shape[0] > 3 else 0.0)
    cymbals = src_np[2]

    return {
        "drums_kick": kick,
        "drums_snare": snare_and_toms,
        "drums_cymbals": cymbals,
    }


def split_guitar(stft, freqs, device):
    """
    Phrase-Coherent Multichannel NMF + Spatial-Timbral Guitar Separator:
    Separates guitar.wav into:
    - acoustic_guitar (Violao / Base Acustica): Full-range strummed chords, acoustic body (85-235Hz)
      & bronze/piezo string harmonics (>3.6kHz), wide stereo rhythm.
    - electric_guitar (Guitarra / Solo & Drive): Band-limited amplifier cabinet midrange (280Hz-3.4kHz),
      sustained melodic solos/riffs & overdrive power chords.
    Crucially binds each note's attack and sustain together (~400ms phrase window) so notes are
    NEVER duplicated across both tracks.
    """
    L = stft[0]
    R = stft[1]
    mag_l = L.abs()
    mag_r = R.abs()
    mag = 0.5 * (mag_l + mag_r)
    f = freqs.unsqueeze(1)

    # 1. Guided 2-Source Non-Negative Matrix Factorization (NMF) templates
    # Acoustic template emphasizes woody body (90-240Hz) + bright string overtones (>3.5kHz)
    w_ac_prior = (
        0.85 * torch.sigmoid((240.0 - f) / 30.0) * torch.sigmoid((f - 80.0) / 15.0)
        + 0.35 * torch.sigmoid((f - 240.0) / 50.0) * torch.sigmoid((3400.0 - f) / 400.0)
        + 1.25 * torch.sigmoid((f - 3400.0) / 450.0)
    ).clamp(min=0.05)

    # Electric template emphasizes guitar amplifier cabinet midrange (280Hz-3.3kHz) with steep cab roll-off >4.2kHz
    w_el_prior = (
        0.15 * torch.sigmoid((260.0 - f) / 30.0)
        + 1.35 * torch.sigmoid((f - 270.0) / 45.0) * torch.sigmoid((3300.0 - f) / 350.0)
        + 0.05 * torch.sigmoid((f - 4200.0) / 300.0)
    ).clamp(min=0.02)

    # Active frames mask (ignore silence when computing song-adaptive medians)
    frame_energy = mag.mean(dim=0)
    active = frame_energy > (frame_energy.mean() * 0.15 + 1e-7)
    if active.sum() < 10:
        active = torch.ones_like(frame_energy, dtype=torch.bool)

    # 2. Frame-level Spectral Sparsity / Kurtosis in Midrange (300Hz - 3200Hz):
    # Electric guitar solos/riffs play 1-2 notes at a time (high peak-to-mean ratio across frequency).
    # Acoustic guitar strums 6-string chords (broad polyphonic energy + high-frequency brush).
    mid_bins = (freqs >= 300.0) & (freqs <= 3200.0)
    mid_mag = mag[mid_bins, :]
    mid_max = mid_mag.max(dim=0, keepdim=True).values.clamp(min=1e-7)
    mid_mean = mid_mag.mean(dim=0, keepdim=True).clamp(min=1e-7)
    sp_ratio = mid_max / mid_mean
    sp_med = sp_ratio[:, active].median()
    lead_sparsity = torch.sigmoid((sp_ratio - sp_med) * 0.9)  # (1, T)

    # 3. High-frequency Acoustic String Sheen + Low Body vs Midrange Ratio per frame:
    hi_bins = (freqs >= 3400.0) & (freqs <= 12000.0)
    body_bins = (freqs >= 85.0) & (freqs <= 240.0)
    hi_energy = mag[hi_bins, :].mean(dim=0, keepdim=True)
    body_energy = mag[body_bins, :].mean(dim=0, keepdim=True)
    ac_frame_ratio = (hi_energy * 2.5 + body_energy * 0.9) / (mid_mean + hi_energy * 2.5 + body_energy * 0.9 + 1e-7)
    ac_med = ac_frame_ratio[:, active].median()
    ac_std = ac_frame_ratio[:, active].std().clamp(min=0.02)
    ac_frame_score = torch.sigmoid(((ac_frame_ratio - ac_med) / ac_std) * 2.2)  # (1, T)

    # 4. Stereo Side vs Panning Subspace:
    M = 0.5 * (L + R)
    D = 0.5 * (L - R)
    side_ratio = D.abs() / (M.abs() + D.abs() + 1e-8)
    side_frame = side_ratio.mean(dim=0, keepdim=True)
    side_med = side_frame[:, active].median()
    side_score = torch.sigmoid((side_frame - side_med) * 8.0)

    # Combine phrase-level activations (smoothed over ~400ms = 17 frames so attack & sustain stay together!)
    h_acoustic = smooth_2d(0.55 * ac_frame_score + 0.25 * (1.0 - lead_sparsity) + 0.20 * side_score, kf=1, kt=17)
    h_electric = smooth_2d(0.55 * (1.0 - ac_frame_score) + 0.25 * lead_sparsity + 0.20 * (1.0 - side_score), kf=1, kt=17)

    # Frequency-band modulation centered around unit balance
    freq_ac_bias = w_ac_prior / (w_ac_prior + w_el_prior)
    freq_el_bias = w_el_prior / (w_ac_prior + w_el_prior)

    v_acoustic = h_acoustic * (0.65 + 0.70 * freq_ac_bias)
    v_electric = h_electric * (0.65 + 0.70 * freq_el_bias)

    # Smooth across 15 time frames (~350ms) and 7 freq bins so a note's pick attack NEVER separates from its sustain
    a_p = smooth_2d(v_acoustic, kf=7, kt=15).pow(8.0)
    e_p = smooth_2d(v_electric, kf=7, kt=15).pow(8.0)
    raw_mask = a_p / (a_p + e_p + 1e-12)

    # Winner-take-all decision gate (pushes mask to 0.0 or 1.0 per musical phrase)
    acoustic_mask = torch.sigmoid((raw_mask - 0.5) * 16.0)
    acoustic_mask = smooth_2d(acoustic_mask, kf=5, kt=11)
    electric_mask = 1.0 - acoustic_mask

    return {
        "acoustic_guitar": stft * acoustic_mask.unsqueeze(0),
        "electric_guitar": stft * electric_mask.unsqueeze(0),
    }


def split_drums_dsp(stft, freqs, device):
    mag = stft.abs().mean(dim=0)
    f = freqs.unsqueeze(1)
    low_cross = torch.sigmoid((135.0 - f) / 10.0)
    high_cross = torch.sigmoid((f - 3800.0) / 250.0)
    mid_cross = torch.clamp(1.0 - low_cross - high_cross, min=0.0)

    sub_band = (mag[(freqs >= 40) & (freqs <= 100), :]).mean(dim=0, keepdim=True)
    snare_band = (mag[(freqs >= 350) & (freqs <= 2200), :]).mean(dim=0, keepdim=True)
    kick_gate = torch.sigmoid(((sub_band / (snare_band + 1e-6)) - 1.5) * 4.0)
    beater_zone = torch.exp(-((f - 800.0) / 400.0).pow(2)) * (kick_gate * 0.45)

    w_kick = (low_cross + beater_zone).pow(4.0)
    w_snare = (mid_cross * (1.0 - 0.5 * kick_gate * torch.sigmoid((220.0 - f) / 25.0))).pow(4.0)
    w_cymbals = high_cross.pow(4.0)

    total_w = (w_kick + w_snare + w_cymbals).clamp(min=1e-12)
    return {
        "drums_kick": stft * (w_kick / total_w).unsqueeze(0),
        "drums_snare": stft * (w_snare / total_w).unsqueeze(0),
        "drums_cymbals": stft * (w_cymbals / total_w).unsqueeze(0),
    }


def split_piano(stft, freqs, device):
    f = freqs.unsqueeze(1)
    low_mask = torch.sigmoid((262.0 - f) / 10.0)
    high_mask = 1.0 - low_mask
    return {
        "piano_low": stft * low_mask.unsqueeze(0),
        "piano_high": stft * high_mask.unsqueeze(0),
    }


def split_other(stft, freqs, device):
    L = stft[0]
    R = stft[1]
    mag = 0.5 * (L.abs() + R.abs())
    mask_harm, _, _, _ = compute_hpss_masks(mag, kf=25, kt=35, power=8.0)
    pads_mask = smooth_2d(mask_harm, kf=5, kt=15)
    pads_mask = torch.sigmoid((pads_mask - 0.5) * 14.0)
    lead_mask = 1.0 - pads_mask
    return {
        "other_pads": stft * pads_mask.unsqueeze(0),
        "other_lead": stft * lead_mask.unsqueeze(0),
    }


SPLITTERS = {
    "vocals": split_vocals_ai,
    "guitar": split_guitar,
    "drums": split_drums_ai,
    "piano": split_piano,
    "other": split_other,
}


def main():
    parser = argparse.ArgumentParser(description="Hierarchical AI + DSP Sub-Stem Decomposition Engine")
    parser.add_argument("--input", required=True, help="Path to parent stem WAV file")
    parser.add_argument("--stem", required=True, choices=list(SPLITTERS.keys()), help="Parent stem type")
    parser.add_argument("--out", required=True, help="Directory to write sub-stems")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--models-dir", default="", help="Directory for on-demand AI checkpoints")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        fail(f"Input stem not found: {args.input}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    models_dir = args.models_dir or default_models_dir()
    os.makedirs(args.out, exist_ok=True)

    emit(type="progress", pct=10, message=f"Loading {args.stem} audio")
    audio, sr = load_wav(args.input)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    # AI neural network path for Vocals (BS-RoFormer Karaoke) and Drums (DrumSep HDemucs)
    if args.stem == "vocals":
        sub_wavs = split_vocals_ai(audio, sr, device, models_dir)
        written = []
        for idx, (sub_name, out_np) in enumerate(sub_wavs.items()):
            emit(type="progress", pct=88 + idx * 6, message=f"Saving {sub_name}")
            save_wav_f32(os.path.join(args.out, f"{sub_name}.wav"), out_np, sr)
            written.append(sub_name)
        with open(os.path.join(args.out, f".subsplit_{args.stem}"), "w") as vf:
            vf.write(SUBSPLIT_VERSION)
        emit(type="done", parent=args.stem, stems=written, out_dir=args.out)
        return

    if args.stem == "drums":
        try:
            sub_wavs = split_drums_ai(audio, sr, device, models_dir)
            written = []
            for idx, (sub_name, out_np) in enumerate(sub_wavs.items()):
                emit(type="progress", pct=85 + idx * 5, message=f"Saving {sub_name}")
                save_wav_f32(os.path.join(args.out, f"{sub_name}.wav"), out_np, sr)
                written.append(sub_name)
            with open(os.path.join(args.out, f".subsplit_{args.stem}"), "w") as vf:
                vf.write(SUBSPLIT_VERSION)
            emit(type="done", parent=args.stem, stems=written, out_dir=args.out)
            return
        except Exception:
            pass

    orig_len = audio.shape[1]
    tensor = torch.from_numpy(audio).to(device)
    del audio

    n_fft = 4096
    hop_length = 1024
    window = torch.hann_window(n_fft, device=device)

    emit(type="progress", pct=30, message=f"Analyzing {args.stem} spectrum")
    with torch.inference_mode():
        stft = torch.stft(
            tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
        )
        freqs = torch.linspace(0, sr / 2.0, stft.shape[1], device=device)

        emit(type="progress", pct=55, message=f"Decomposing {args.stem} sub-stems")
        splitter = split_drums_dsp if args.stem == "drums" else SPLITTERS[args.stem]
        sub_stfts = splitter(stft, freqs, device)
        del stft
        del tensor

        written = []
        items = list(sub_stfts.items())
        for idx, (sub_name, sub_spec) in enumerate(items):
            pct = 65 + int(((idx + 1) / len(items)) * 30)
            emit(type="progress", pct=pct, message=f"Saving {sub_name}")
            wav_tensor = torch.istft(
                sub_spec,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                length=orig_len,
                center=True,
            )
            out_np = wav_tensor.cpu().numpy()
            out_path = os.path.join(args.out, f"{sub_name}.wav")
            save_wav_f32(out_path, out_np, sr)
            written.append(sub_name)
            del wav_tensor
            del out_np

    with open(os.path.join(args.out, f".subsplit_{args.stem}"), "w") as vf:
        vf.write(SUBSPLIT_VERSION)
    emit(type="done", parent=args.stem, stems=written, out_dir=args.out)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(str(e))
