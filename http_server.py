from flask import Flask, request, send_file
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
from typing import List, Optional
from threading import Lock, Semaphore
import time

app = Flask(__name__)

# Device and instance setup
num_gpus = torch.cuda.device_count()
cpu_workers = int(os.environ.get("TTS_CPU_WORKERS", "8"))
devices = [f"cuda:{i}" for i in range(num_gpus)] + ["cpu"] * max(1, cpu_workers)

TTS_INSTANCES = [TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev) for dev in devices]
INSTANCE_LOCKS = [Lock() for _ in TTS_INSTANCES]
WORKER_BACKLOG = int(os.environ.get("TTS_WORKER_BACKLOG", "2"))
QUEUE_SEMAPHORES = [Semaphore(WORKER_BACKLOG) for _ in TTS_INSTANCES]
gc.collect()

def acquire_instance():
    """Acquire instance with per-worker backlog; prefer GPUs, then CPUs."""
    total = len(TTS_INSTANCES)
    gpu_range = range(min(num_gpus, total))
    cpu_range = range(num_gpus, total)
    while True:
        for i in gpu_range:
            if QUEUE_SEMAPHORES[i].acquire(blocking=False):
                INSTANCE_LOCKS[i].acquire()
                return TTS_INSTANCES[i], INSTANCE_LOCKS[i], i
        for i in cpu_range:
            if QUEUE_SEMAPHORES[i].acquire(blocking=False):
                INSTANCE_LOCKS[i].acquire()
                return TTS_INSTANCES[i], INSTANCE_LOCKS[i], i
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


def synth_segment(text: str, language: str, speaker: Optional[str], tts_model: TTS, **kwargs) -> io.BytesIO:
    """Synthesize a single segment and return it as a BytesIO WAV buffer."""
    with managed_buffer() as buf:
        tts_model.tts_to_file(text=clean_text_for_tts(text, language), language=language, 
                              file_path=buf, speaker=speaker, **kwargs)
        buf.seek(0)
        return io.BytesIO(buf.getvalue())


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

    tts_model, instance_lock, worker_idx = acquire_instance()
    
    # Get default speaker if available
    default_speaker = (getattr(tts_model, 'speakers', [None]) or [None])[0] if (
        not speaker_kwargs and getattr(tts_model, 'is_multi_speaker', False)
    ) else None

    final_buf = None
    try:
        try:
            final_buf = (synth_segment(text, language, default_speaker, tts_model, **speaker_kwargs) 
                        if len(text) <= MAX_CHARS else 
                        concat_wavs([synth_segment(seg, language, default_speaker, tts_model, **speaker_kwargs) 
                                   for seg in split_sentences(text)]))
            if len(text) > MAX_CHARS:
                gc.collect()
        finally:
            instance_lock.release()
            QUEUE_SEMAPHORES[worker_idx].release()

        return send_file(final_buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    except Exception as e:
        final_buf and final_buf.close()
        return str(e), 500
    finally:
        tmp and os.path.exists(tmp) and os.remove(tmp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10230, debug=False, use_reloader=False)

