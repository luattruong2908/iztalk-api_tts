import sys
import os
sys.path.append(os.path.abspath("src"))

import base64
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from infer.f5tts_wrapper import F5TTSWrapper
from fastapi.responses import FileResponse
import uvicorn

# ==== CONFIGURATION ====
REF_VOICE_DIR = "src/infer/ref_voices"
OUTPUT_DIR = "output"
VOCAB_PATH = "src/infer/model/vocab.txt"
CKPT_PATH = "src/infer/model/model_48000.safetensors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== TEXT NORMALIZATION (thay cho vinorm) ====
def normalize_text(text: str) -> str:
    return text.strip()

# ==== TTS INIT ====
tts = F5TTSWrapper(
    vocoder_name="vocos",
    ckpt_path=CKPT_PATH,
    vocab_file=VOCAB_PATH,
    use_ema=False,
)

# ==== FASTAPI ====
app = FastAPI()

class VoiceRef(BaseModel):
    id: str
    filename: str

@app.get("/voice_refs", response_model=List[VoiceRef])
def get_voice_refs():
    wav_files = glob.glob(os.path.join(REF_VOICE_DIR, "*.wav"))
    return [
        VoiceRef(
            id=os.path.splitext(os.path.basename(f))[0],
            filename=os.path.basename(f)
        )
        for f in wav_files
    ]

class SynthesizeRequest(BaseModel):
    voice_id: str
    text: str

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    ref_path = os.path.join(REF_VOICE_DIR, f"{req.voice_id}.wav")
    if not os.path.isfile(ref_path):
        raise HTTPException(status_code=404, detail="Voice reference not found")

    try:
        text_norm = normalize_text(req.text)

        # Chuẩn hóa và clip audio tham chiếu
        tts.preprocess_reference(ref_audio_path=ref_path, ref_text="", clip_short=True)

        # Sinh âm thanh
        output_path = os.path.join(OUTPUT_DIR, f"gen_{req.voice_id}.wav")
        tts.generate(
            text=text_norm,
            output_path=output_path,
            nfe_step=20,
            cfg_strength=2.0,
            speed=1.0,
            cross_fade_duration=0.15,
        )

        # Đọc và mã hóa base64
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "voice_id": req.voice_id,
            "text": req.text,
            "audio_base64": audio_base64,
            "mime_type": "audio/wav"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
