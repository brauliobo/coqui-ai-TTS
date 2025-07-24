from flask import Flask, request, send_file
from TTS.api import TTS
import torch
import os
import io
import uuid
import re
import wave
from typing import List, Optional

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

MAX_CHARS = 250*4  # safe length per request after tokenizer limit increase

def split_sentences(text: str, max_len: int = MAX_CHARS) -> List[str]:
    """Split text into sentence-based chunks not exceeding *max_len* characters."""
    if len(text) <= max_len:
        return [text.strip()]

    parts = re.split(r"(?<=[.!?])\s+", text)
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


def synth_segment(text: str, language: str, speaker: Optional[str], **kwargs) -> io.BytesIO:
    """Synthesize a single segment and return it as a BytesIO WAV buffer."""
    buf = io.BytesIO()
    tts.tts_to_file(text=text, language=language, file_path=buf, speaker=speaker, **kwargs)
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

    # Use first available pre-defined speaker if model supports multiple speakers and no speaker_wav provided.
    default_speaker = None
    if not speaker_kwargs and getattr(tts, 'is_multi_speaker', False):
        spk_list = getattr(tts, 'speakers', None)
        if spk_list:
            default_speaker = spk_list[0]

    try:
        if len(text) <= MAX_CHARS:
            final_buf = synth_segment(text, language, default_speaker, **speaker_kwargs)
        else:
            segments = split_sentences(text)
            buffers = [synth_segment(seg, language, default_speaker, **speaker_kwargs) for seg in segments]
            final_buf = concat_wavs(buffers)

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

