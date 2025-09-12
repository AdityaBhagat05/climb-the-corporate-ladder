

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import time
import json
import re

# local utilities
from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import detect_posture_and_confidence
from langchain_ollama import ChatOllama

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    posture_history: list
    start_time: float
    evaluation_done: bool
    pass_meter: int

# --- Globals ---
llm = ChatOllama(model="mistral:instruct", temperature=0.6)

# --- Nodes ---

def stt_node(state: AgentState) -> AgentState:
    audio_file = record_audio()
    text = speech_to_text(audio_file)
    try:
        os.remove(audio_file)
    except Exception:
        pass

    
    new_state = {
        "messages": state["messages"],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0),
    }

    if text and text.strip().lower() in ["exit", "quit", "stop"]:
        new_state["messages"] = list(new_state["messages"]) + [HumanMessage(content="exit")]
        return new_state

    if text:
        human_msg = HumanMessage(content=text)
        new_state["messages"] = list(new_state["messages"]) + [human_msg]

    return new_state


def llm_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    current_pass_meter = state.get("pass_meter", 0)
    
    if messages and getattr(messages[-1], "content", "").strip().lower() == "exit":
        # graceful shutdown message
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
        "pass_meter": current_pass_meter  # Pass meter unchanged in this node
    }


def tts_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not msgs:
        return state
    last = msgs[-1]
    text = getattr(last, "content", str(last))

    if not text:
        return state

    if text.strip().lower() in ["exit", "quit", "stop", "goodbye!"]:
        return state

    if "Evaluation:" in text:
        text = text.split("Evaluation:")[0].strip()

    try:
        text_to_speech(text)
    except Exception as e:
        print("TTS error:", e)

    return state


def continue_conv(state: AgentState) -> str:
    msgs = state["messages"]
    if msgs and getattr(msgs[-1], "content", "").strip().lower() == "exit":
        return "end"
    if isinstance(msgs[-1], SystemMessage) and "goodbye" in msgs[-1].content.lower():
        return "end"

    elapsed = time.time() - state.get("start_time", time.time())
    if elapsed >= 120:
        print("⏰ Timer ended: 2 minutes reached, moving to evaluation.")
        return "end"

    return "continue"


def posture_info_node(state: AgentState) -> AgentState:
    if "posture_history" not in state:
        state["posture_history"] = []

    try:
        data = detect_posture_and_confidence()
    except Exception as e:
        data = {"posture": "unknown", "gaze": "unknown", "confidence": "unknown", "arms": "unknown", "head_tilt": None}
        print("Posture detection error:", e)

    state["posture_history"].append(data)
    print(f"[Posture Info] {data}")
    return {
        "messages": state["messages"], 
        "posture_history": state["posture_history"],
        "start_time": state.get("start_time", time.time()), 
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0)
    }


def evaluation_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    posture_data = state.get("posture_history", [])
    current_pass_meter = state.get("pass_meter", 0)

    conversation_text = "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in msgs if hasattr(msg, 'content') and msg.content
    )

    summary = "Posture and Confidence History:\n"
    for entry in posture_data:
        summary += (
            f"- Posture: {entry.get('posture')}, Gaze: {entry.get('gaze')}, "
            f"Confidence: {entry.get('confidence')}, Arms: {entry.get('arms')}, "
            f"Head Tilt: {entry.get('head_tilt')}\n"
        )

    evaluation_prompt_text = f"""You are evaluating a public speaking performance in a training exercise.

Conversation history:
{conversation_text}


Evaluation criteria (focus on delivery, not content):

1. (25%)Based on the response,Did the user appear calm and confident?
2. (25%)Was the speech clear and easy to understand?
3. (25%)Was the grammar correct?
4. (25%)Did the user make a persuasive argument?
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
        "posture_history": posture_data,
        "start_time": state.get("start_time", time.time()), 
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": new_pass_meter
    }


def final_evaluation(state: AgentState) -> AgentState:
    current_pass_meter = state.get("pass_meter", 0)
    print(f"Pass meter final value: {current_pass_meter}")
    if current_pass_meter >= 0:
        print("Passed")
    else:
        print("Failed")
    return {
        "messages": state["messages"],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": True,
        "pass_meter": current_pass_meter
    }


# --- Graph ---

graph = StateGraph(AgentState)
graph.add_node("stt", stt_node)
graph.add_node("camera", posture_info_node)
graph.add_node("llm", llm_node)
graph.add_node("tts", tts_node)
graph.add_node("evaluation", evaluation_node)
graph.add_node("final_evaluation", final_evaluation)

# edges
graph.add_edge(START, "stt")
graph.add_edge("stt", "camera")
graph.add_edge("camera", "llm")
graph.add_edge("llm", "tts")
graph.add_edge("tts", "evaluation")

graph.add_conditional_edges(
    "evaluation",
    lambda state: ("continue" if continue_conv(state) == "continue" else "end"),
    {
        "continue": "stt",
        "end": "final_evaluation",
    },
)

graph.add_edge("final_evaluation", END)

app = graph.compile()

if __name__ == "__main__":
    meeting_topic = "is AI a fad?"

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
-Use 1 or 2 short sentences only in your response.
""")
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0
    }

    final_state = app.invoke(seed)

    final_pass_meter = final_state.get("pass_meter", 0)
    print(f"\n--- Final Result ---")
    print(f"Pass meter: {final_pass_meter}")
    if final_pass_meter >= 0:
        print("Overall: PASSED")
    else:
        print("Overall: FAILED")
    print("Conversation finished.")