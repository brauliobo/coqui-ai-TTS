from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
from TTS.tts.utils.text.cleaners import *
import torch
import os
import io
import gc
import numpy as np
import inspect
from cachetools import TTLCache
import traceback
from threading import Lock, Semaphore
import time

app = Flask(__name__)
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"
MODEL_ID = os.environ.get("TTS_MODEL_ID", "tts_models/multilingual/multi-dataset/xtts_v2")

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

# Least-load selection tracking
INFLIGHT_LOCK = Lock()
INFLIGHT_COUNTS = []

MODEL_CACHE_TTL = int(os.environ.get("TTS_MODEL_CACHE_TTL_SECONDS", "600"))
MODEL_CACHE = TTLCache(maxsize=max(1, len(devices)), ttl=MODEL_CACHE_TTL)
MODEL_CACHE_LOCK = Lock()

def get_model_for_device(device: str) -> TTS:
    with MODEL_CACHE_LOCK:
        model = MODEL_CACHE.get(device)
        if model is not None:
            return model
        model = TTS(MODEL_ID).to(device)
        MODEL_CACHE[device] = model
        return model

INSTANCE_LOCKS = [Lock() for _ in devices]
WORKER_BACKLOG = int(os.environ.get("TTS_WORKER_BACKLOG", "10"))
QUEUE_SEMAPHORES = [Semaphore(WORKER_BACKLOG) for _ in devices]
INFLIGHT_COUNTS = [0 for _ in devices]
gc.collect()


@app.teardown_request
def _teardown_request_cleanup(exception):
    # Best-effort CPU memory cleanup after each request
    gc.collect()

def acquire_instance():
    """Acquire queue slot (not lock) preferring least-loaded device; return model and worker index."""
    total = len(devices)
    gpu_range = range(min(num_gpus, total))
    cpu_range = range(num_gpus, total)
    while True:
        # Order devices by fewest in-flight requests (GPUs first)
        with INFLIGHT_LOCK:
            gpu_order = sorted(list(gpu_range), key=lambda i: INFLIGHT_COUNTS[i])
            cpu_order = sorted(list(cpu_range), key=lambda i: INFLIGHT_COUNTS[i])
        ordered = gpu_order + cpu_order
        # Try non-blocking acquires in order
        for i in ordered:
            if QUEUE_SEMAPHORES[i].acquire(blocking=False):
                with INFLIGHT_LOCK:
                    INFLIGHT_COUNTS[i] += 1
                return get_model_for_device(devices[i]), i
        # Briefly wait on the currently least-loaded device
        if ordered:
            i = ordered[0]
            if QUEUE_SEMAPHORES[i].acquire(timeout=0.05):
                with INFLIGHT_LOCK:
                    INFLIGHT_COUNTS[i] += 1
                return get_model_for_device(devices[i]), i

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

def synth_segment(text: str, language: str, worker_idx: int, tts_model: TTS, **kwargs) -> np.ndarray:
    """Synthesize one segment and return waveform as a numpy array (no temp files)."""
    lock = INSTANCE_LOCKS[worker_idx]
    clean_text = clean_text_for_tts(text, language)
    with lock:
        with torch.inference_mode():
            wav = tts_model.tts(text=clean_text, language=language, **kwargs)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    return wav


def write_wav_with_coqui(tts_model: TTS, samples: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    tts_model.synthesizer.save_wav(wav=samples, path=buf)
    buf.seek(0)
    return buf


@app.route('/speakers', methods=['GET'])
def list_speakers():
    model = get_model_for_device(devices[0]) if devices else None
    speakers = list(getattr(model, 'speakers', []) or []) if model else []
    return jsonify([{'id': i, 'name': s} for i, s in enumerate(speakers)])


@app.route('/synthesize', methods=['POST'])
def synthesize():
    if not (text := request.form.get('text')) or not (language := request.form.get('language')):
        return "Missing text or language", 400

    # Speaker selection (no temp files)
    speaker_kwargs = {}

    temperature = request.args.get('temperature', default=0.0, type=float)
    speaker_id = request.args.get('speaker', type=int)
    if speaker_id is None:
        speaker_id = request.form.get('speaker', default=31, type=int)

    final_buf, worker_idx = None, None
    try:
        tts_model, worker_idx = acquire_instance()

        speakers = list(getattr(tts_model, 'speakers', []) or [])
        if not speaker_kwargs and getattr(tts_model, 'is_multi_speaker', False) and speakers:
            idx = speaker_id if 0 <= speaker_id < len(speakers) else (1 if len(speakers) > 1 else 0)
            default_speaker = speakers[idx]
            sig_params = set(inspect.signature(tts_model.tts).parameters.keys())
            if 'speaker' in sig_params:
                speaker_kwargs['speaker'] = default_speaker
            elif 'speaker_idx' in sig_params:
                speaker_kwargs['speaker_idx'] = idx
            elif 'speaker_id' in sig_params:
                speaker_kwargs['speaker_id'] = idx

        sig = inspect.signature(tts_model.tts)
        tts_kwargs = dict(speaker_kwargs)
        if 'temperature' in sig.parameters:
            tts_kwargs['temperature'] = temperature

        audio = None
        try:
            audio = synth_segment(text, language, worker_idx, tts_model, **tts_kwargs)
            final_buf = write_wav_with_coqui(tts_model, audio)
        finally:
            if worker_idx is not None:
                QUEUE_SEMAPHORES[worker_idx].release()
                with INFLIGHT_LOCK:
                    INFLIGHT_COUNTS[worker_idx] = max(0, INFLIGHT_COUNTS[worker_idx] - 1)

        resp = send_file(final_buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
        def _cleanup():
            try:
                final_buf.close()
            except Exception:
                pass
            try:
                # Clear cache only for the device used by this request
                dev = devices[worker_idx]
                if dev.startswith("cuda:"):
                    torch.cuda.set_device(int(dev.split(":")[1]))
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        resp.call_on_close(_cleanup)

        # Free CPU/GPU caches after generating
        try:
            del audio
        except Exception:
            pass
        gc.collect()

        return resp
    except Exception:
        'final_buf' in locals() and final_buf and final_buf.close()
        if DEBUG_MODE:
            app.logger.exception("/synthesize failed")
            return traceback.format_exc(), 500
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10230, debug=DEBUG_MODE, use_reloader=False, threaded=True)

