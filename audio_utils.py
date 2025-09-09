import os
import tempfile
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS

# Global playback control
play_lock = threading.Lock()
is_speaking = threading.Event()

# Load models
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")

TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
tts_engine = TTS(TTS_MODEL)
tts_engine.to("cuda")



def record_audio(silence_threshold=0.1, silence_duration=3.0, samplerate=16000) -> str:

    while is_speaking.is_set():
        time.sleep(0.1)

    q = queue.Queue()
    audio_data = []
    silent_chunks = 0
    silence_limit = int(silence_duration * samplerate / 1024) 
    max_chunks = int(30 * samplerate / 1024)  # 30 sec max

    def callback(indata, frames, time, status):
        if status:
            print("InputStream status:", status)
        q.put(indata.copy())

    print("Speak now... (auto-stop after silence)")
    
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=1024):
        chunk_count = 0
        while chunk_count < max_chunks:
            chunk = q.get()
            audio_data.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
                
            if silent_chunks >= silence_limit:
                break
                
            chunk_count += 1

    audio = np.concatenate(audio_data, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, samplerate)
    return tmp.name

def speech_to_text(audio_path: str) -> str:
    print("Transcribing...")
    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()
    print("Transcription:", text)
    return text

#tts
def text_to_speech(text: str, samplerate=22050) -> str:
    if not text:
        return ""
        
    fname = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav")
    print("Synthesizing speech...")
    tts_engine.tts_to_file(text=text, file_path=fname)

    def play_audio():
        try:
            with play_lock:  # 🔒 Only one playback at a time
                is_speaking.set()  # 🚨 Mark speaking
                data, sr = sf.read(fname, dtype="float32")
                sd.play(data, sr)
                sd.wait()
                is_speaking.clear()  # ✅ Done speaking
            try:
                os.remove(fname)
            except:
                pass
        except Exception as e:
            print("Playback error:", e)

    threading.Thread(target=play_audio).start()
    return fname