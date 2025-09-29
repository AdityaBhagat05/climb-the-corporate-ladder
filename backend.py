
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import time
from langchain_ollama import ChatOllama
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from convince_boss import graph, AgentState
from public_speaking import publicAgentState, public_graph
import tempfile
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_utils import speech_to_text, text_to_speech, speak
import asyncio, aiofiles
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
app = FastAPI()



DB_PATH_CONV = "checkpointsconv.sqlite"
THREAD_ID_CONV = "test14" 
config_conv = {"configurable": {"thread_id": THREAD_ID_CONV}}

DB_PATH_PUB = "checkpointspub.sqlite"
THREAD_ID_PUBLIC= "publictest8"
config_public={"configurable": {"thread_id": THREAD_ID_PUBLIC}}

DB_PATH_NPC = "checkpointsnpc.sqlite"
THREAD_ID_NPC = "npctest7" 
config_npc = {"configurable": {"thread_id": THREAD_ID_NPC}}



@app.post("/convince_boss")
async def convince_boss(audioFile: UploadFile = File(...)):
    meeting_topic = "is AI a fad?"

    contents = await audioFile.read()
    text = speech_to_text(contents)

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

    with SqliteSaver.from_conn_string(DB_PATH_CONV) as memory:
        graph_app = graph.compile(checkpointer=memory)
        snapshot = memory.get(config_conv)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))

            def _reconstruct_message(m):
                if isinstance(m, BaseMessage):
                    return m
                if isinstance(m, dict):
                    typ = m.get("type") or m.get("_type") or m.get("message_type")
                    content = m.get("content") or m.get("text") or m.get("body")
                    if typ:
                        typ = typ.lower()
                        if "system" in typ:
                            return SystemMessage(content=content or "")
                        if "human" in typ:
                            return HumanMessage(content=content or "")
                        if "ai" in typ or "assistant" in typ:
                            return AIMessage(content=content or "")
                return m

            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]
            msgs.append(HumanMessage(content=text))
            previous_state["messages"] = msgs

            start_time = previous_state.get("start_time", seed["start_time"])
            elapsed = time.time() - start_time
            if elapsed > 120: 
                final_pass_meter = previous_state.get("pass_meter", 0)
                timeout_msg = f"I'm out of time. Final pass meter: {final_pass_meter}"
                print("[convince_boss] TIMEOUT:", timeout_msg)

                try:
                    text_to_speech(timeout_msg)
                except Exception as e:
                    print("[convince_boss] TTS on timeout failed:", e)

                sucess = final_pass_meter >= 0
                return {
                    "text_response": timeout_msg,
                    "sucess": sucess,
                    "pass_meter": final_pass_meter
                }

            final_state = graph_app.invoke(previous_state, config_conv)
        else:
            print("Starting new conversation...")
            final_state = graph_app.invoke(seed, config_conv)

        boss_reply = final_state["messages"][-1].content

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Pass meter: {final_pass_meter}")
        sucess = final_pass_meter >= 0
        print("Overall:", "PASSED" if sucess else "FAILED")
        print("Conversation finished.")

        return {
            "text_response": boss_reply,
            "sucess": sucess,
            "pass_meter": final_pass_meter
        }
    




@app.post("/public_speaking")
async def public_speaking(audioFile: UploadFile = File(...)):
    meeting_topic = "is AI a fad?"
    max_interactions = 5  # <-- use 5 interactions

    contents = await audioFile.read()
    text = speech_to_text(contents)
    public_seed: publicAgentState = {
        "messages": [
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0,
        "turns": 0,
        "last_question": "",
        "last_user_response": "",
        "last_transcript": "",
        "max_turns": max_interactions,  # <-- reflect the 5-interaction limit
        "meeting_topic": meeting_topic,
    }

    with SqliteSaver.from_conn_string(DB_PATH_PUB) as memory:
        graph_app = public_graph.compile(checkpointer=memory)
        snapshot = memory.get(config_public)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))

            def _reconstruct_message(m):
                if isinstance(m, BaseMessage):
                    return m
                if isinstance(m, dict):
                    typ = m.get("type") or m.get("_type") or m.get("message_type")
                    content = m.get("content") or m.get("text") or m.get("body")
                    if typ:
                        typ = typ.lower()
                        if "system" in typ:
                            return SystemMessage(content=content or "")
                        if "human" in typ:
                            return HumanMessage(content=content or "")
                        if "ai" in typ or "assistant" in typ:
                            return AIMessage(content=content or "")
                return m

            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]
            msgs.append(HumanMessage(content=text))
            previous_state["messages"] = msgs

            # Use interactions (turns) instead of elapsed time
            current_turns = int(previous_state.get("turns", 0))
            if current_turns >= max_interactions:
                final_pass_meter = previous_state.get("pass_meter", 0)
                timeout_msg = f"Well thats it. Final pass meter: {final_pass_meter}"
                print("[public_speaking] MAX INTERACTIONS:", timeout_msg)

                try:
                    text_to_speech(timeout_msg)
                except Exception as e:
                    print("[public_speaking] TTS on max-interactions failed:", e)

                sucess = final_pass_meter >= 0
                return {
                    "text_response": timeout_msg,
                    "sucess": sucess,
                    "pass_meter": final_pass_meter
                }

            # still allowed to continue — invoke the graph
            final_state = graph_app.invoke(previous_state, config_public)
        else:
            print("Starting new conversation...")
            final_state = graph_app.invoke(public_seed, config_public)

        audience_reply = final_state["messages"][-1].content

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Excitement meter: {final_pass_meter}")
        sucess = final_pass_meter >= 0
        print("Overall:", "PASSED" if sucess else "FAILED")
        print("Conversation finished.")

        return {
            "text_response": audience_reply,
            "sucess": sucess,
            "Excitement_meter": final_pass_meter
        }

@app.post("/play_npc")
async def play_npc(audioFile: UploadFile = File(...)):
    meeting_topic = "is AI a fad?"

    contents = await audioFile.read()
    text = speech_to_text(contents)

    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the coworker of the user. 
you have to give hints to the user regarding an important upcoming meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value)
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be friendly and fun, but formal.
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

    with SqliteSaver.from_conn_string(DB_PATH_NPC) as memory:
        graph_app = graph.compile(checkpointer=memory)
        snapshot = memory.get(config_npc)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))

            def _reconstruct_message(m):
                if isinstance(m, BaseMessage):
                    return m
                if isinstance(m, dict):
                    typ = m.get("type") or m.get("_type") or m.get("message_type")
                    content = m.get("content") or m.get("text") or m.get("body")
                    if typ:
                        typ = typ.lower()
                        if "system" in typ:
                            return SystemMessage(content=content or "")
                        if "human" in typ:
                            return HumanMessage(content=content or "")
                        if "ai" in typ or "assistant" in typ:
                            return AIMessage(content=content or "")
                return m

            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]
            msgs.append(HumanMessage(content=text))
            previous_state["messages"] = msgs

            start_time = previous_state.get("start_time", seed["start_time"])
            elapsed = time.time() - start_time
            if elapsed > 120: 
                final_pass_meter = previous_state.get("pass_meter", 0)
                timeout_msg = f"I have to run. See you later!"
                print("[convince_boss] TIMEOUT:", timeout_msg)

                try:
                    text_to_speech(timeout_msg)
                except Exception as e:
                    print("[convince_boss] TTS on timeout failed:", e)

                sucess = final_pass_meter >= 0
                return {
                    "text_response": timeout_msg,
                    "sucess": sucess,
                    "pass_meter": final_pass_meter
                }

            final_state = graph_app.invoke(previous_state, config_npc)
        else:
            print("Starting new conversation...")
            final_state = graph_app.invoke(seed, config_npc)

        boss_reply = final_state["messages"][-1].content

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Pass meter: {final_pass_meter}")
        sucess = final_pass_meter >= 0

        return {
            "text_response": boss_reply,
            "sucess": sucess,
            "pass_meter": final_pass_meter
        }
    


