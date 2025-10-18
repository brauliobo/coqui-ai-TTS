from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
from TTS.tts.utils.text.cleaners import *
import torch
import os
import io
import uuid
import re
import wave
import gc
from contextlib import contextmanager
import inspect
from cachetools import TTLCache
import traceback
from typing import List, Optional
from threading import Lock, Semaphore
import time

app = Flask(__name__)
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Performance knobs
torch.set_num_threads(int(os.environ.get("TTS_TORCH_THREADS", "1")))
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# Device and instance setup
num_gpus = torch.cuda.device_count()
cpu_workers = int(os.environ.get("TTS_CPU_WORKERS", "8"))
devices = [f"cuda:{i}" for i in range(num_gpus)] + ["cpu"] * max(1, cpu_workers)

MODEL_CACHE_TTL = int(os.environ.get("TTS_MODEL_CACHE_TTL_SECONDS", "600"))
MODEL_CACHE = TTLCache(maxsize=max(1, len(devices)), ttl=MODEL_CACHE_TTL)
MODEL_CACHE_LOCK = Lock()

def get_model_for_device(device: str) -> TTS:
    with MODEL_CACHE_LOCK:
        m = MODEL_CACHE.get(device)
        if m is not None:
            return m
    new_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    with MODEL_CACHE_LOCK:
        existing = MODEL_CACHE.get(device)
        if existing is not None:
            return existing
        MODEL_CACHE[device] = new_model
        return new_model

INSTANCE_LOCKS = [Lock() for _ in devices]
WORKER_BACKLOG = int(os.environ.get("TTS_WORKER_BACKLOG", "10"))
QUEUE_SEMAPHORES = [Semaphore(WORKER_BACKLOG) for _ in devices]
gc.collect()

def acquire_instance():
    """Acquire queue slot (not lock) preferring GPUs; return model and worker index."""
    total = len(devices)
    gpu_range = range(min(num_gpus, total))
    cpu_range = range(num_gpus, total)
    while True:
        # Rotate GPU preference to balance across devices
        g = list(gpu_range)
        s = (int(time.time() * 1000) % len(g)) if g else 0
        for i in (g[s:] + g[:s]):
            if QUEUE_SEMAPHORES[i].acquire(blocking=False):
                return get_model_for_device(devices[i]), i
        for i in cpu_range:
            if QUEUE_SEMAPHORES[i].acquire(blocking=False):
                return get_model_for_device(devices[i]), i
        time.sleep(0.005)

@contextmanager
def managed_buffer():
    """Context manager for BytesIO buffers to ensure proper cleanup."""
    buf = io.BytesIO()
    try:
        yield buf
    finally:
        buf.close()

MAX_CHARS = 1000

LANGUAGE_CLEANERS = {
    "en": english_cleaners,
    "fr": french_cleaners,
    "pt": portuguese_cleaners,
    "zh": chinese_mandarin_cleaners,
    "zh-cn": chinese_mandarin_cleaners,
    "de": basic_german_cleaners,
    "tr": basic_turkish_cleaners,
}

def clean_text_for_tts(text: str, language: str) -> str:
    """Apply appropriate text cleaner based on language."""
    return LANGUAGE_CLEANERS.get(language, multilingual_cleaners)(text) if text else text

def split_sentences(text: str, max_len: int = MAX_CHARS) -> List[str]:
    """Split text into sentence-based chunks not exceeding max_len characters."""
    if len(text) <= max_len:
        return [text.strip()]

    chunks, buf = [], ""
    for part in re.split(r"(?<=[.!?…:;])\s+", text):
        if len(buf) + len(part) > max_len and buf:
            chunks.append(buf.strip())
            buf = part
        else:
            buf += (" " if buf else "") + part
    return chunks + [buf.strip()] if buf else chunks or [text.strip()]


def synth_segment(text: str, language: str, worker_idx: int, tts_model: TTS, **kwargs) -> io.BytesIO:
    """Synthesize one segment; write to a temp file path to ensure compatibility, then load into memory."""
    lock = INSTANCE_LOCKS[worker_idx]
    clean_text = clean_text_for_tts(text, language)
    tmp_out = f"{uuid.uuid4()}.wav"
    lock.acquire()
    try:
        tts_model.tts_to_file(text=clean_text, language=language, file_path=tmp_out, **kwargs)
    finally:
        lock.release()
    try:
        with open(tmp_out, "rb") as f:
            data = f.read()
        buf = io.BytesIO(data)
        buf.seek(0)
        return buf
    finally:
        os.path.exists(tmp_out) and os.remove(tmp_out)


def concat_wavs(buffers: List[io.BytesIO]) -> io.BytesIO:
    """Concatenate multiple WAV buffers into one and return a new buffer."""
    if len(buffers) == 1:
        return buffers[0]

    out_buf = io.BytesIO()
    try:
        with wave.open(out_buf, "wb") as out_wav:
            params_set = False
            for buf in buffers:
                try:
                    with wave.open(buf, "rb") as in_wav:
                        if not params_set:
                            out_wav.setparams(in_wav.getparams())
                            params_set = True
                        out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
                finally:
                    # Close each input buffer after processing to free memory
                    buf.close()
        out_buf.seek(0)
        return out_buf
    except Exception:
        out_buf.close()
        raise


@app.route('/speakers', methods=['GET'])
def list_speakers():
    model = get_model_for_device(devices[0]) if devices else None
    speakers = list(getattr(model, 'speakers', []) or []) if model else []
    return jsonify([{'id': i, 'name': s} for i, s in enumerate(speakers)])


@app.route('/synthesize', methods=['POST'])
def synthesize():
    if not (text := request.form.get('text')) or not (language := request.form.get('language')):
        return "Missing text or language", 400

    # Handle optional speaker audio
    speaker_kwargs, tmp = {}, None
    if audio_file := request.files.get('audio'):
        tmp = f"{uuid.uuid4()}.wav"
        audio_file.save(tmp)
        speaker_kwargs['speaker_wav'] = tmp

    temperature = request.args.get('temperature', default=0.0, type=float)
    speaker_id = request.args.get('speaker', type=int)
    if speaker_id is None:
        speaker_id = request.form.get('speaker', default=31, type=int)

    final_buf, worker_idx = None, None
    try:
        tts_model, worker_idx = acquire_instance()

        speakers = list(getattr(tts_model, 'speakers', []) or [])
        default_speaker = None
        if not speaker_kwargs and getattr(tts_model, 'is_multi_speaker', False) and speakers:
            idx = speaker_id if 0 <= speaker_id < len(speakers) else (1 if len(speakers) > 1 else 0)
            default_speaker = speakers[idx]
            sig_params = set(inspect.signature(tts_model.tts_to_file).parameters.keys())
            if 'speaker' in sig_params:
                speaker_kwargs['speaker'] = default_speaker
            elif 'speaker_idx' in sig_params:
                speaker_kwargs['speaker_idx'] = idx
            elif 'speaker_id' in sig_params:
                speaker_kwargs['speaker_id'] = idx

        sig = inspect.signature(tts_model.tts_to_file)
        tts_kwargs = dict(speaker_kwargs)
        if 'temperature' in sig.parameters:
            tts_kwargs['temperature'] = temperature

        try:
            final_buf = (synth_segment(text, language, worker_idx, tts_model, **tts_kwargs)
                        if len(text) <= MAX_CHARS else
                        concat_wavs([synth_segment(seg, language, worker_idx, tts_model, **tts_kwargs)
                                   for seg in split_sentences(text)]))
            if len(text) > MAX_CHARS:
                gc.collect()
        finally:
            worker_idx is not None and QUEUE_SEMAPHORES[worker_idx].release()

        return send_file(final_buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    except Exception:
        final_buf and final_buf.close()
        if DEBUG_MODE:
            app.logger.exception("/synthesize failed")
            return traceback.format_exc(), 500
        return "Internal Server Error", 500
    finally:
        tmp and os.path.exists(tmp) and os.remove(tmp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10230, debug=DEBUG_MODE, use_reloader=False, threaded=True)

