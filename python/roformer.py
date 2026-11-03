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


def release_device_memory(device):
    """torch's caching allocators keep blocks that refcounts already dropped,
    so a fallback that loads a second copy of the engine gets charged for the
    same ram twice — which is how one gpu failure turns into an out-of-memory
    on the retry"""
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and getattr(torch.backends, "mps", None):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception:
        pass


CKPT_NAME = "MelBandRoformer.ckpt"
CKPT_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"

SW6_CKPT_NAME = "BS-Roformer-SW.ckpt"
SW6_CKPT_URL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/BS-Roformer-SW.ckpt"
SW6_STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"]

# KimberleyJensen mel-band roformer vocal model (SDR vocals 10.98 on Multisong).
# https://github.com/ZFTurbo/Music-Source-Separation-Training
PARAMS = dict(
    dim=384,
    depth=6,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_bands=60,
    dim_head=64,
    heads=8,
    attn_dropout=0,
    ff_dropout=0,
    flash_attn=True,
    dim_freqs_in=1025,
    sample_rate=44100,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)

# jarredou BS-Roformer-SW 6-stem studio model (bass, drums, other, vocals, guitar, piano)
SW6_PARAMS = dict(
    dim=256,
    depth=12,
    stereo=True,
    num_stems=6,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=(2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129),
    dim_head=64,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    flash_attn=True,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=512,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
    mlp_expansion_factor=4,
    use_torch_checkpoint=False,
    skip_connection=False,
)

CHUNK_SIZE = 352800  # 8s at 44100
NUM_OVERLAP = 2
# batching >1 triggers pathological MPS paths (same runtime as fp32), so 1 it is
BATCH_SIZE = 1
MIN_CKPT_BYTES = 500_000_000


def emit(**kwargs):
    print(json.dumps(kwargs), flush=True)


def fail(message):
    emit(type="error", message=str(message))
    sys.exit(1)


def crash_message(e):
    """top-level crash handler text: unreadable stderr tails (a bare python
    traceback or an unrelated warnings.warn line) become actionable errors"""
    msg = f"{e}"
    low = msg.lower()
    if "no kernel image" in low or "invalid device function" in low:
        return (
            "GPU engine incompatible with this GPU (compute capability not supported by the "
            "installed engine) — update StemKit or turn off GPU acceleration in Settings"
        )
    return msg[:400] or e.__class__.__name__


def load_wav(path):
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
    """write a 32-bit float wav (fmt tag 3); values above 1.0 are preserved"""
    channels, _ = data.shape
    # one transposed contiguous copy written straight from memory: building a
    # bytes payload first would hold a second full copy of the stem in ram
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


def download_checkpoint(dest, url=CKPT_URL, label="vocals engine"):
    """download the checkpoint with resume support and progress emits"""
    if os.path.exists(dest):
        return
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
                    if now - last_emit >= 1.0 and total > 0:
                        pct = int(f.tell() / total * 100)
                        emit(
                            type="progress",
                            stage="separate",
                            pct=0,
                            message=f"Downloading {label} ({total // (1024 * 1024)}MB): {pct}%",
                        )
                        last_emit = now
    except Exception as e:
        fail(f"{label} download failed: {e}")
    if os.path.getsize(part) < MIN_CKPT_BYTES:
        fail(f"{label} download incomplete ({os.path.getsize(part)} bytes)")
    os.replace(part, dest)
    emit(type="progress", stage="separate", pct=0, message=f"{label.capitalize()} downloaded")


def resolve_device():
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        return "cuda", cap_major >= 7
    if torch.backends.mps.is_available():
        return "mps", True
    return "cpu", False


def load_model(ckpt_path, device, use_half, variant="vocals"):
    use_flash = True
    if device == "cuda" and torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        use_flash = cap_major >= 8
    elif device == "cpu":
        use_flash = False

    if variant == "sw6":
        from models.bs_roformer.bs_roformer import BSRoformer
        model = BSRoformer(**dict(SW6_PARAMS, flash_attn=use_flash))
    else:
        from models.bs_roformer.mel_band_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(PARAMS, flash_attn=use_flash))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    # the checkpoint is a second full copy of the weights sitting next to the
    # model that just consumed them: ~900MB of ram held for nothing through
    # the whole separation, which is often the difference between a split and
    # a kill
    del ckpt
    gc.collect()
    model.to(device)
    if use_half:
        model.half()
        # band_split accumulates large stft magnitudes and overflows fp16
        model.band_split.to(torch.float32)
    model.eval()
    return model


def separate(model, mix, device, use_half, on_progress, num_stems=1):
    """chunked inference with overlap; mix is a (2, n) tensor on device"""
    total = mix.shape[-1]
    step = CHUNK_SIZE // NUM_OVERLAP
    border = CHUNK_SIZE - step
    fade_size = CHUNK_SIZE // 10
    length_init = total
    if length_init > 2 * border and border > 0:
        mix = torch.nn.functional.pad(mix, (border, border), mode="reflect")

    window = torch.ones(CHUNK_SIZE, device=device)
    window[:fade_size] = torch.linspace(0, 1, fade_size, device=device)
    window[-fade_size:] = torch.linspace(1, 0, fade_size, device=device)

    if num_stems == 1:
        result = torch.zeros(mix.shape, device=device)
        counter = torch.zeros(mix.shape, device=device)
    else:
        result = torch.zeros((num_stems, mix.shape[0], mix.shape[1]), device=device)
        counter = torch.zeros((1, 1, mix.shape[1]), device=device)

    with torch.inference_mode():
        i = 0
        batch_data = []
        batch_locations = []
        while i < mix.shape[-1]:
            part = mix[:, i : i + CHUNK_SIZE]
            chunk_len = part.shape[-1]
            pad_mode = "reflect" if chunk_len > CHUNK_SIZE // 2 else "constant"
            part = torch.nn.functional.pad(part, (0, CHUNK_SIZE - chunk_len), mode=pad_mode, value=0)
            if use_half:
                part = part.to(torch.float16)
            batch_data.append(part)
            batch_locations.append((i, chunk_len))
            i += step
            if len(batch_data) >= BATCH_SIZE or i >= mix.shape[-1]:
                arr = torch.stack(batch_data, dim=0)
                x = model(arr)
                for j, (start, seg_len) in enumerate(batch_locations):
                    win = window[:seg_len].clone()
                    if start == 0:
                        win[:fade_size] = 1
                    elif i >= mix.shape[-1]:
                        win[-fade_size:] = 1
                    if num_stems == 1:
                        result[..., start : start + seg_len] += x[j, :, :seg_len].float() * win
                        counter[..., start : start + seg_len] += win
                    else:
                        result[:, :, start : start + seg_len] += x[j, :, :, :seg_len].float() * win
                        counter[:, :, start : start + seg_len] += win
                batch_data.clear()
                batch_locations.clear()
                on_progress(i / mix.shape[-1])

    est = result / counter.clamp(min=1e-8)
    if length_init > 2 * border and border > 0:
        est = est[..., border:-border]
    return est


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="vocals", choices=["vocals", "sw6"])
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    sys.path.insert(0, vendor)

    # fail fast before the checkpoint download on GPU-less machines
    if args.device == "cuda" and not torch.cuda.is_available():
        fail("GPU engine not available (no supported NVIDIA/AMD GPU, or the GPU build of torch is not installed)")

    is_sw6 = args.model == "sw6"
    ckpt_name = SW6_CKPT_NAME if is_sw6 else CKPT_NAME
    ckpt_url = SW6_CKPT_URL if is_sw6 else CKPT_URL
    label = "6-stem studio engine" if is_sw6 else "vocals engine"
    size_mb = 667 if is_sw6 else 913
    num_stems = 6 if is_sw6 else 1

    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        emit(type="progress", stage="separate", pct=0, message=f"Downloading {label} ({size_mb}MB, one time)")
        download_checkpoint(ckpt_path, url=ckpt_url, label=label)

    if args.device == "auto":
        device, use_half = resolve_device()
    else:
        device = args.device
        if device == "cuda" and torch.cuda.is_available():
            use_half = torch.cuda.get_device_capability(0)[0] >= 7
        else:
            use_half = device == "mps"
    emit(type="progress", stage="separate", pct=0, message=f"loading {label} on {device}")

    started = time.time()
    try:
        model = load_model(ckpt_path, device, use_half, variant=args.model)
    except Exception as e:
        fail(f"{label} load failed: {e}")

    audio, sr = load_wav(args.input)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    mix = torch.from_numpy(audio).to(device)
    del audio

    last_emit = 0.0

    def on_progress(frac):
        nonlocal last_emit
        now = time.time()
        if now - last_emit >= 0.5:
            emit(type="progress", stage="separate", pct=min(92, int(frac * 92)))
            last_emit = now

    emit(type="progress", stage="separate", pct=0, message="separating")
    try:
        est = separate(model, mix, device, use_half, on_progress, num_stems=num_stems)
    except Exception as e:
        if device == "mps":
            emit(type="progress", stage="separate", pct=0, message=f"gpu failed ({e}), falling back to cpu")
            model = None
            gc.collect()
            mix = mix.to("cpu")
            release_device_memory("mps")
            device = "cpu"
            use_half = False
            try:
                model = load_model(ckpt_path, device, use_half, variant=args.model)
            except Exception as e2:
                fail(f"{label} load failed on cpu: {e2}")
            try:
                est = separate(model, mix, device, use_half, on_progress, num_stems=num_stems)
            except Exception as e2:
                fail(f"{label} separation failed on cpu: {e2}")
        else:
            fail(f"separation failed: {e}")

    out_cpu = est.cpu().numpy()
    model = None
    mix = None
    est = None
    gc.collect()
    release_device_memory(device)
    if np.isnan(out_cpu).any():
        fail("separation produced invalid audio")
    os.makedirs(args.out, exist_ok=True)

    if not is_sw6:
        emit(type="progress", stage="separate", pct=96, message="Saving vocals stem")
        save_wav_f32(os.path.join(args.out, "vocals.wav"), out_cpu, sr)
        emit(type="stem", name="vocals")
        emit(type="progress", stage="separate", pct=99, message="Vocals ready")
        emit(
            type="done",
            stems=["vocals"],
            out_dir=args.out,
            seconds=round(time.time() - started, 1),
        )
    else:
        wanted = None
        if args.only:
            wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        target_names = [name for name in SW6_STEMS if wanted is None or name in wanted]
        total_targets = max(1, len(target_names))
        written = []
        for idx, name in enumerate(target_names):
            pct = 92 + int(((idx + 1) / total_targets) * 7)
            emit(type="progress", stage="separate", pct=min(99, pct), message=f"Saving {name} stem")
            i = SW6_STEMS.index(name)
            save_wav_f32(os.path.join(args.out, f"{name}.wav"), out_cpu[i], sr)
            written.append(name)
            emit(type="stem", name=name)
        emit(type="progress", stage="separate", pct=99, message="Finalizing stems")
        emit(
            type="done",
            stems=written,
            out_dir=args.out,
            seconds=round(time.time() - started, 1),
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(crash_message(e))
