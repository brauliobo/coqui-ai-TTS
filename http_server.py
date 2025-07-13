from flask import Flask, request, send_file
from TTS.api import TTS
import torch
import os
import io
import uuid

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

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
        output_buffer = io.BytesIO()
        tts.tts_to_file(text=text, language=language, file_path=output_buffer, speaker=default_speaker, **speaker_kwargs)
        output_buffer.seek(0)
        return send_file(output_buffer, mimetype="audio/wav", as_attachment=True, download_name="output.wav")
    except Exception as e:
        return str(e), 500
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10230)

