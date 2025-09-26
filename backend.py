# from typing import TypedDict, Annotated, Sequence
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
# from dotenv import load_dotenv
# import os
# import time
# from langchain_ollama import ChatOllama
# from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
# from fastapi.responses import FileResponse, JSONResponse
# from convince_boss import graph, AgentState
# import os
# import tempfile
# import threading
# import time
# import queue
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
# import whisper
# from TTS.api import TTS
# import re
# # from audio_utils import speech_to_text, text_to_speech
# import asyncio,aiofiles
# from concurrent.futures import ThreadPoolExecutor

# load_dotenv()
# app = FastAPI()

# play_lock = threading.Lock()
# is_speaking = threading.Event()

# # Load models
# WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")

# TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
# tts_engine = TTS(TTS_MODEL)
# tts_engine.to("cuda")


# llm = ChatOllama(model="mistral:instruct", temperature=0.6)

# DB_PATH = "checkpoints.sqlite"
# THREAD_ID = "test2" 
# config = {"configurable": {"thread_id": THREAD_ID}}





# # @app.post("/convince_boss")
# # async def convince_boss(audioFile: str):
# #     meeting_topic = "is AI a fad?"
# #     text=speech_to_text(audioFile)
# #     seed: AgentState = {
# #         "messages": [
# #             SystemMessage(content=f"""
# # You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
# # You are playing the role of the boss of the user. 
# # The user has to try to convince you to let them present in an important meeting. 
# # The topic of the meeting is '{meeting_topic}'.

# # Instructions for the roleplay:
# # - Your tone should adapt based on the user's performance (pass_meter value)
# # - Respond only with dialogue, as if you are speaking directly to the user.  
# # - Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
# # - Be firm and direct. Give constructive, actionable feedback in short sentences.
# # - Push back, challenge their arguments, and make them defend themselves.  
# # - Keep your responses very short and to the point — no more than 1-2 sentences.
# # - If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
# # - Use 1 or 2 short sentences only in your response.
# # """), HumanMessage(content=text),
# #         ],
# #         "posture_history": [],
# #         "evaluation_done": False,
# #         "start_time": time.time(),
# #         "pass_meter": 0
# #     }

# #     with SqliteSaver.from_conn_string(DB_PATH) as memory:
# #         graph_app = graph.compile(checkpointer=memory)

# #         snapshot = memory.get(config)
# #         if snapshot:
# #             # final_state = graph_app.invoke(
# #             #     snapshot["messages"].append(HumanMessage(content=text)),
# #             #     config
# #             # )
# #             previous_state = dict(snapshot) 
# #          # make a copy
# #             if "messages" not in previous_state:
# #                 previous_state["messages"] = []
# #             previous_state["messages"].append(HumanMessage(content=text))
# #             final_state = graph_app.invoke(previous_state, config)
# #         else:
# #             print("Starting new conversation...")
# #             final_state = graph_app.invoke(seed, config)
        
# #         boss_reply = final_state["messages"][-1].content

        
# #         audio_path = text_to_speech(boss_reply)

# #         final_pass_meter = final_state.get("pass_meter", 0)
# #         print(f"\n--- Final Result ---")
# #         print(f"Pass meter: {final_pass_meter}")
# #         if final_pass_meter >= 0:
# #             print("Overall: PASSED")
# #         else:
# #             print("Overall: FAILED")
# #         print("Conversation finished.")

# #         # Step 5: Return both text + audio (so frontend can choose)
# #         return {
# #             "text_response": boss_reply,
# #             "audio_response": audio_path,
# #             "pass_meter": final_pass_meter
# #         }

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import time
from langchain_ollama import ChatOllama
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS
import re
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import io
import base64
import tempfile
import shutil

load_dotenv()
app = FastAPI()

# Serve static files for audio
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global playback control
play_lock = threading.Lock()
is_speaking = threading.Event()

# Load models
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")

TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
tts_engine = TTS(TTS_MODEL)
tts_engine.to("cuda")

llm = ChatOllama(model="mistral:instruct", temperature=0.6)

DB_PATH = "checkpoints.sqlite"
THREAD_ID = "test3" 
config = {"configurable": {"thread_id": THREAD_ID}}

# Audio utility functions
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


def text_to_speech_streaming(text):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # write TTS to temp_file
        with open(temp_file.name, "wb") as f:
            f.write(generate_tts_bytes(text))  # your TTS function
        # read the bytes back
        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
        return audio_bytes
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            print("Warning: could not delete temp file:", e)

def text_to_speech_file(text: str, samplerate=22050) -> str:
    """Convert text to speech and return file path"""
    if not text:
        return ""
        
    fname = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav")
    print("Synthesizing speech...")
    tts_engine.tts_to_file(text=text, file_path=fname)
    return fname


import uuid
import json
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Globals to add near the top of your file
executor = ThreadPoolExecutor(max_workers=4)  # tune to your GPU/CPU
session_states: Dict[str, dict] = {}         # in-memory per-ws AgentState cache
session_locks: Dict[str, asyncio.Lock] = {}  # per-session async lock

# These will be initialized at startup
compiled_graph = None
persistent_memory = None  # SqliteSaver instance

@app.on_event("startup")
async def startup_event():
    global compiled_graph, persistent_memory
    # Create a persistent SqliteSaver (keep it open)
    # If SqliteSaver.from_conn_string returns a context-managed object,
    # call its constructor / factory once and keep the returned object.
    persistent_memory = SqliteSaver.from_conn_string(DB_PATH)
    # compile graph once, using the checkpointer/persistent_memory
    # (some langgraph APIs compile differently; adapt if your graph.compile requires other args)
    compiled_graph = graph.compile(checkpointer=persistent_memory)
    print("Startup: compiled graph and opened persistent DB saver.")


async def run_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: fn(*args, **kwargs))


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()

    # assign a session id for this websocket connection
    session_id = str(uuid.uuid4())
    session_locks[session_id] = asyncio.Lock()
    session_states[session_id] = None  # will populate after first invoke

    try:
        while True:
            data = await websocket.receive()

            # ----------------------------
            # 1) Binary audio from client
            # ----------------------------
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # 1a. Transcribe in executor (blocking Whisper) -> returns text
                transcription = await run_in_executor(_transcribe_bytes, audio_bytes)

                # Immediately send back the transcription (low-latency feedback)
                await websocket.send_text(f"TRANSCRIPTION:{transcription}")

                # 1b. Evaluate / invoke LangGraph & LLM in executor
                # use per-session lock to avoid concurrent invokes for same session
                async with session_locks[session_id]:
                    # pass the transcription into the graph and get LLM reply + updated state
                    boss_reply, new_state = await run_in_executor(
                        _process_turn_with_graph, session_id, transcription
                    )

                    # update in-memory state
                    session_states[session_id] = new_state

                # 1c. Speak the boss reply (TTS) in executor -> get bytes
                audio_buffer = await run_in_executor(text_to_speech_streaming, boss_reply)
                audio_bytes_out = audio_buffer.getvalue()

                # 1d. Send the LLM reply meta + audio in binary
                # Option A: send a short metadata text first
                await websocket.send_text(json.dumps({"type": "llm_reply", "text": boss_reply}))

                # Then send binary audio. Client must expect raw wav bytes here.
                await websocket.send_bytes(audio_bytes_out)

            # ----------------------------
            # 2) Text commands (TTS trigger etc)
            # ----------------------------
            elif "text" in data:
                message = data["text"]
                if message.startswith("TTS:"):
                    tts_text = message[4:]
                    audio_buffer = await run_in_executor(text_to_speech_streaming, tts_text)
                    await websocket.send_bytes(audio_buffer.getvalue())

                elif message.startswith("RESET_STATE"):
                    # allow client to reset conversation
                    session_states[session_id] = None
                    # also optionally clear DB snapshot
                    await run_in_executor(_clear_session_snapshot, session_id)
                    await websocket.send_text("SESSION_RESET")

                else:
                    # other control msgs
                    await websocket.send_text(f"UNKNOWN_COMMAND:{message}")

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        # on disconnect persist session state if you want
        if session_states.get(session_id):
            await run_in_executor(_checkpoint_session_state, session_id)
        # cleanup
        session_states.pop(session_id, None)
        session_locks.pop(session_id, None)


from pydub import AudioSegment
import io

def _transcribe_bytes(audio_bytes: bytes) -> str:
    """Convert any audio format to WAV for Whisper."""
    
    try:
        # Load audio from bytes using pydub (handles multiple formats)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to mono, 16kHz sample rate (Whisper's preferred format)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export as WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_data = wav_io.getvalue()
        
    except Exception as e:
        print(f"Audio conversion failed: {e}")
        # Fallback: try direct processing with WebM extension
        wav_data = audio_bytes
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_data)
        tmp_name = tmp_file.name
    
    try:
        text = speech_to_text(tmp_name)
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    return text


def _process_turn_with_graph(session_id: str, user_text: str):
    """Invoke compiled_graph with current session state, update pass_meter, return boss reply and new state object."""
    global compiled_graph, persistent_memory

    # build seed similar to your /convince_boss endpoint
    meeting_topic="who is the hottest bollywood actress?"
    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the boss of the user. 
The user has to try to convince you to let them present in an important meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value)
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be firm and direct. Give constructive, actionable feedback in short sentences.
- Push back, challenge their arguments, and make them defend themselves.  
- Keep your responses very short and to the point — no more than 1-2 sentences.
- If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
- Use 1 or 2 short sentences only in your response.
"""),
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0
    }

    # try to reuse existing session snapshot
    snapshot = None
    try:
        snapshot = persistent_memory.get({"configurable": {"thread_id": session_id}})
    except Exception:
        # if persistent saved snapshot unavailable, fallback to in-memory
        snapshot = None

    if snapshot:
        previous_state = dict(snapshot)
        if "messages" not in previous_state:
            previous_state["messages"] = []
        previous_state["messages"].append(HumanMessage(content=user_text))
        final_state = compiled_graph.invoke(previous_state, {"configurable": {"thread_id": session_id}})
    else:
        # if no snapshot, use seed
        final_state = compiled_graph.invoke(seed, {"configurable": {"thread_id": session_id}})

    # extract boss reply from final_state
    boss_reply = ""
    msgs = final_state.get("messages", [])
    if msgs:
        last = msgs[-1]
        boss_reply = getattr(last, "content", str(last))

    # checkpoint the new state into persistent_memory (non-blocking could be done later)
    try:
        persistent_memory.put({"configurable": {"thread_id": session_id}, **final_state})
    except Exception:
        pass

    return boss_reply, final_state


def _checkpoint_session_state(session_id: str):
    """Force persist in-memory session state into DB on disconnect."""
    global persistent_memory
    state = session_states.get(session_id)
    if state and persistent_memory:
        persistent_memory.put({"configurable": {"thread_id": session_id}, **state})


def _clear_session_snapshot(session_id: str):
    """Optional: delete DB snapshot for session (implementation depends on SqliteSaver API)"""
    # implement deletion if your SqliteSaver exposes it
    pass


# REST endpoint for audio file upload and processing
@app.post("/convince_boss")
async def convince_boss(audio_file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        audio_path = tmp_file.name
    
    meeting_topic = "is AI a fad?"
    text = speech_to_text(audio_path)
    
    # Clean up uploaded file
    os.unlink(audio_path)
    
    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the boss of the user. 
The user has to try to convince you to let them present in an important meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value)
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be firm and direct. Give constructive, actionable feedback in short sentences.
- Push back, challenge their arguments, and make them defend themselves.  
- Keep your responses very short and to the point — no more than 1-2 sentences.
- If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
- Use 1 or 2 short sentences only in your response.
"""), HumanMessage(content=text),
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0
    }

    with SqliteSaver.from_conn_string(DB_PATH) as memory:
        graph_app = graph.compile(checkpointer=memory)

        snapshot = memory.get(config)
        if snapshot:
            previous_state = dict(snapshot)
            if "messages" not in previous_state:
                previous_state["messages"] = []
            previous_state["messages"].append(HumanMessage(content=text))
            final_state = graph_app.invoke(previous_state, config)
        else:
            print("Starting new conversation...")
            final_state = graph_app.invoke(seed, config)
        
        boss_reply = final_state["messages"][-1].content
        final_pass_meter = final_state.get("pass_meter", 0)
        
        print(f"\n--- Final Result ---")
        print(f"Pass meter: {final_pass_meter}")
        if final_pass_meter >= 0:
            print("Overall: PASSED")
        else:
            print("Overall: FAILED")
        print("Conversation finished.")

        # Generate audio file
        audio_buffer = text_to_speech_streaming(boss_reply)
        
        # Return response with audio data as base64
        audio_data = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return {
            "text_response": boss_reply,
            "audio_data": audio_data,  # Base64 encoded audio
            "audio_format": "wav",
            "pass_meter": final_pass_meter
        }

# Streaming TTS endpoint
@app.post("/tts")
async def text_to_speech_endpoint(request: dict):
    text = request.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    
    audio_buffer = text_to_speech_streaming(text)
    
    return StreamingResponse(
        io.BytesIO(audio_buffer.getvalue()),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

# Download audio file endpoint
@app.get("/audio/{audio_id}")
async def get_audio_file(audio_id: str):
    # In a real implementation, you'd look up the file path by audio_id
    audio_path = text_to_speech_file("This is a test audio file.")
    
    return FileResponse(
        path=audio_path,
        media_type='audio/wav',
        filename='response.wav'
    )

# Health check endpoint

# module-level
_persistent_memory_ctx = None
persistent_memory = None
compiled_graph = None

@app.on_event("startup")
async def startup_event():
    global compiled_graph, persistent_memory, _persistent_memory_ctx
    _persistent_memory_ctx = SqliteSaver.from_conn_string(DB_PATH)
    persistent_memory = _persistent_memory_ctx.__enter__()   # ENSURE real saver object
    compiled_graph = graph.compile(checkpointer=persistent_memory)
    print("Startup: compiled graph and opened persistent DB saver.")
    
@app.on_event("shutdown")
async def shutdown_event():
    global _persistent_memory_ctx
    if _persistent_memory_ctx is not None:
        try:
            _persistent_memory_ctx.__exit__(None, None, None)
        except Exception as e:
            print("Error closing persistent saver:", e)


@app.get("/")
async def root():
    return {"message": "Audio Streaming API is running"}

# AgentState definition (from convince_boss.py)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    posture_history: list
    start_time: float
    evaluation_done: bool
    pass_meter: int

# Graph definition (simplified version from convince_boss.py)
def llm_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    current_pass_meter = state.get("pass_meter", 0)
    
    if messages and getattr(messages[-1], "content", "").strip().lower() == "exit":
        messages.append(SystemMessage(content="Goodbye!"))
        return {
            "messages": messages, 
            "posture_history": state.get("posture_history", []),
            "start_time": state.get("start_time", time.time()), 
            "evaluation_done": True,
            "pass_meter": current_pass_meter
        }

    pass_meter_context = f"\n\nCurrent user performance score: {current_pass_meter}. "
    if current_pass_meter <= -4:
        pass_meter_context += "The user is performing very poorly. Be extremely rude, dismissive, and impatient. Use short, harsh responses. Do not elaborate."
    elif current_pass_meter <= -2:
        pass_meter_context += "The user is performing poorly. Be critical and skeptical. Use short responses. Challenge every point."
    elif current_pass_meter <= 0:
        pass_meter_context += "The user is performing neutrally. Be professional but challenging. Use concise responses."
    else:
        pass_meter_context += "The user is performing well. Show grudging respect but still push back. Use concise responses."

    modified_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            modified_msg = SystemMessage(content=msg.content + pass_meter_context)
            modified_messages.append(modified_msg)
        else:
            modified_messages.append(msg)

    print("Sending messages to LLM (history length =", len(modified_messages), ")")
    try:
        resp = llm.invoke(messages=modified_messages)
    except TypeError:
        resp = llm.invoke(modified_messages)

    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate))
    
    if "Evaluation:" in assistant_text:
        assistant_text = assistant_text.split("Evaluation:")[0].strip()
    
    ai_msg = AIMessage(content=assistant_text)

    return {
        "messages": messages + [ai_msg], 
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()), 
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": current_pass_meter
    }

def evaluation_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    current_pass_meter = state.get("pass_meter", 0)

    conversation_text = "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in msgs if hasattr(msg, 'content') and msg.content
    )

    evaluation_prompt_text = f"""You are evaluating a public speaking performance in a training exercise.

Conversation history:
{conversation_text}

Evaluation criteria (focus on delivery, not content):
1. (25%) Based on the response, Did the user appear calm and confident?
2. (25%) Was the speech clear and easy to understand?
3. (25%) Was the grammar correct?
4. (25%) Did the user make a persuasive argument?
5. DO NOT focus on the actual content of the presentation but rather on the public speaking skills.

Scoring:
- PASS if the speaker demonstrates good public speaking skills (score ≥60%)
- FAIL if the speaker needs significant improvement (score <60%)

Return EXACTLY one JSON object with two fields:
{{"decision": "PASS" or "FAIL", "explanation": "brief explanation focusing on delivery skills"}}

Remember: You're evaluating public speaking skills, not the factual accuracy of the AI arguments.
"""

    human_msg = HumanMessage(content=evaluation_prompt_text)

    try:
        resp = llm.invoke(messages=[human_msg])
    except TypeError:
        resp = llm.invoke([human_msg])

    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate)).strip()

    decision = None
    explanation = None
    try:
        parsed = json.loads(assistant_text)
        decision = parsed.get("decision", "").strip().upper()
        explanation = parsed.get("explanation", "").strip()
    except Exception:
        m = re.search(r'\b(PASS|FAIL)\b[:\-\s]*(.*)', assistant_text, re.IGNORECASE)
        if m:
            decision = m.group(1).upper()
            explanation = m.group(2).strip()[:200] if m.group(2) else ""
        else:
            decision = "FAIL"
            explanation = assistant_text.replace("\n", " ")[:200]

    if decision == "PASS":
        new_pass_meter = current_pass_meter + 2
        print("✅ pass_meter increased to:", new_pass_meter)
    else:
        new_pass_meter = current_pass_meter - 2
        print("❌ pass_meter decreased to:", new_pass_meter)

    print(f"Evaluation: {decision}. {explanation}")

    return {
        "messages": msgs,  
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()), 
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": new_pass_meter
    }

# Create graph
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("evaluation", evaluation_node)

graph.add_edge(START, "llm")
graph.add_edge("llm", "evaluation")
graph.add_edge("evaluation", END)

def generate_tts_bytes(text: str) -> bytes:
    """
    Generate TTS audio bytes from text using TTS model.
    Returns WAV-formatted bytes suitable for streaming.
    """
    import io
    from TTS.utils.audio import AudioProcessor

    if not text:
        return b""

    # Use in-memory buffer
    buf = io.BytesIO()

    # Generate speech as numpy array
    wav = tts_engine.tts(text, speaker=None)  # returns numpy array at 22050 Hz by default

    # Convert numpy array to WAV in-memory
    import soundfile as sf
    sf.write(buf, wav, samplerate=tts_engine.synthesizer.output_sample_rate, format='WAV')
    buf.seek(0)
    return buf.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
