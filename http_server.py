from flask import Flask, request, send_file
from TTS.api import TTS
from TTS.tts.utils.text.cleaners import (
    english_cleaners, 
    french_cleaners, 
    portuguese_cleaners, 
    chinese_mandarin_cleaners,
    basic_german_cleaners,
    basic_turkish_cleaners,
    multilingual_cleaners
)
import torch
import os
import io
import uuid
import re
import wave
from typing import List, Optional
from threading import Lock
from itertools import cycle

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build a small pool of TTS instances to enable parallel requests when possible.
# Strategy: Always prefer GPUs, only use CPU workers when all GPUs are busy.
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
cpu_workers = int(os.environ.get("TTS_CPU_WORKERS", "4"))

gpu_devices = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else []
cpu_devices = ["cpu"] * max(1, cpu_workers) if cpu_workers > 0 else []

TTS_INSTANCES = [TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev) for dev in gpu_devices + cpu_devices]
INSTANCE_LOCKS = [Lock() for _ in TTS_INSTANCES]  # guard per-instance use

def acquire_instance():
    # Try GPUs first (non-blocking check), fallback to any available instance
    for i in range(num_gpus):
        if INSTANCE_LOCKS[i].acquire(blocking=False):
            return TTS_INSTANCES[i], INSTANCE_LOCKS[i]
    
    # All GPUs busy, find any available instance (blocking)
    for i, lock in enumerate(INSTANCE_LOCKS):
        if lock.acquire(blocking=False):
            return TTS_INSTANCES[i], lock
    
    # All busy, wait for first GPU (prefer GPU over CPU even when waiting)
    if num_gpus > 0:
        INSTANCE_LOCKS[0].acquire()
        return TTS_INSTANCES[0], INSTANCE_LOCKS[0]
    else:
        # No GPUs, use first CPU instance
        INSTANCE_LOCKS[0].acquire()
        return TTS_INSTANCES[0], INSTANCE_LOCKS[0]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

MAX_CHARS = 250*4  # safe length per request after tokenizer limit increase

def clean_text_for_tts(text: str, language: str) -> str:
    """Apply appropriate text cleaner based on language to handle punctuation properly."""
    if not text:
        return text

    # Use language-specific cleaners that handle punctuation correctly
    if language == "en":
        return english_cleaners(text)
    elif language == "fr":
        return french_cleaners(text)
    elif language == "pt":
        cleaned = portuguese_cleaners(text)
        cleaned = re.sub(r'([.!?…:;])+([\s\n]|$)', "\n", cleaned)
        return re.sub(r"\n{2,}", "\n", cleaned).strip()
    elif language == "zh" or language == "zh-cn":
        return chinese_mandarin_cleaners(text)
    elif language == "de":
        return basic_german_cleaners(text)
    elif language == "tr":
        return basic_turkish_cleaners(text)
    else:
        # For multilingual models or unsupported languages, use multilingual_cleaners
        return multilingual_cleaners(text)

def split_sentences(text: str, max_len: int = MAX_CHARS) -> List[str]:
    """Split text into sentence-based chunks not exceeding *max_len* characters."""
    if len(text) <= max_len:
        return [text.strip()]

    parts = re.split(r"(?<=[.!?…:;])\s+", text)
    chunks, buf = [], ""
    for part in parts:
        if len(buf) + len(part) > max_len:
            if buf:
                chunks.append(buf.strip())
            buf = part
        else:
            buf += (" " if buf else "") + part
    if buf:
        chunks.append(buf.strip())
    return chunks or [text.strip()]


def synth_segment(text: str, language: str, speaker: Optional[str], tts_model: TTS, **kwargs) -> io.BytesIO:
    """Synthesize a single segment and return it as a BytesIO WAV buffer."""
    buf = io.BytesIO()
    # Clean text using appropriate language-specific cleaner to handle punctuation
    cleaned_text = clean_text_for_tts(text, language)
    tts_model.tts_to_file(text=cleaned_text, language=language, file_path=buf, speaker=speaker, **kwargs)
    buf.seek(0)
    return buf


def concat_wavs(buffers: List[io.BytesIO]) -> io.BytesIO:
    """Concatenate multiple WAV buffers into one and return a new buffer."""
    if len(buffers) == 1:
        return buffers[0]

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out_wav:
        params_set = False
        for buf in buffers:
            with wave.open(buf, "rb") as in_wav:
                if not params_set:
                    out_wav.setparams(in_wav.getparams())
                    params_set = True
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
    out_buf.seek(0)
    return out_buf


@app.route('/synthesize', methods=['POST'])
def synthesize():
    # Validate mandatory fields.
    if not request.form.get('text') or not request.form.get('language'):
        return "Missing text or language", 400

    audio_file = request.files.get('audio')  # Optional speaker reference.
    text = request.form['text']
    language = request.form['language']

    speaker_kwargs, tmp = {}, None
    if audio_file:
        tmp = f"{uuid.uuid4()}.wav"
        audio_file.save(tmp)
        speaker_kwargs['speaker_wav'] = tmp

    # Pin and lock a model instance for this request to keep segments consistent.
    tts_model, instance_lock = acquire_instance()

    # Use first available pre-defined speaker if model supports multiple speakers and no speaker_wav provided.
    default_speaker = None
    if not speaker_kwargs and getattr(tts_model, 'is_multi_speaker', False):
        spk_list = getattr(tts_model, 'speakers', None)
        if spk_list:
            default_speaker = spk_list[0]

    try:
        try:
            if len(text) <= MAX_CHARS:
                final_buf = synth_segment(text, language, default_speaker, tts_model, **speaker_kwargs)
            else:
                segments = split_sentences(text)
                buffers = [
                    synth_segment(seg, language, default_speaker, tts_model, **speaker_kwargs)
                    for seg in segments
                ]
                final_buf = concat_wavs(buffers)
        finally:
            # release the per-instance lock acquired in acquire_instance()
            instance_lock.release()

        return send_file(final_buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    except Exception as e:
        return str(e), 500
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

if __name__ == '__main__':
    # Run in production mode: disable debug and the automatic reloader which
    # would otherwise spawn a second Python process (duplicating GPU memory).
    app.run(host='0.0.0.0', port=10230, debug=False, use_reloader=False)

