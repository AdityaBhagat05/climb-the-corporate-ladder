


from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import detect_posture_and_confidence

load_dotenv()

# --- State ---
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     posture_history: list
#     evaluation_done: bool

import time

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    posture_history: list
    evaluation_done: bool
    start_time: float   # store when session started



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


# --- Nodes ---
def stt_node(state: AgentState) -> AgentState:
    audio_file = record_audio()
    text = speech_to_text(audio_file)
    try:
        os.remove(audio_file)
    except:
        pass

    if text.lower() in ["exit", "quit", "stop"]:
        return {"messages": state["messages"] + [HumanMessage(content="exit")],
                "posture_history": state.get("posture_history", []),
                "evaluation_done": False}

    human_msg = HumanMessage(content=text)
    return {"messages": state["messages"] + [human_msg],
            "posture_history": state.get("posture_history", []),
            "evaluation_done": False}


def llm_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if messages and messages[-1].content.lower() == "exit":
        return {"messages": messages + [SystemMessage(content="Goodbye!")],
                "posture_history": state.get("posture_history", []),
                "evaluation_done": False}

    print("Sending messages to LLM (history length =", len(messages), ")")
    try:
        resp = llm.invoke(messages=messages)
    except TypeError:
        resp = llm.invoke(messages)

    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate))
    ai_msg = AIMessage(content=assistant_text)

    return {"messages": messages + [ai_msg],
            "posture_history": state.get("posture_history", []),
            "evaluation_done": False}


def tts_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    if not msgs:
        return state
    last = msgs[-1]
    text = getattr(last, "content", str(last))

    if text.lower() in ["exit", "quit", "stop", "goodbye!"]:
        return state

    text_to_speech(text)
    return state


def continue_conv(state: AgentState) -> str:
    msgs = state["messages"]

    # End if user says exit/quit/stop
    if msgs and msgs[-1].content.lower() == "exit":
        return "end"
    if isinstance(msgs[-1], SystemMessage) and "goodbye" in msgs[-1].content.lower():
        return "end"

    # End if 2 minutes have passed
    elapsed = time.time() - state.get("start_time", time.time())
    if elapsed >= 60:  # 120 seconds = 2 minutes
        print("⏰ Timer ended: 2 minutes reached, moving to evaluation.")
        return "end"

    return "continue"



def posture_info_node(state: AgentState) -> AgentState:
    if "posture_history" not in state:
        state["posture_history"] = []

    data = detect_posture_and_confidence()
    state["posture_history"].append(data)

    print(f"[Posture Info] {data}")
    return {"messages": state["messages"], "posture_history": state["posture_history"], "evaluation_done": False}



def evaluation_node(state: AgentState) -> AgentState:
    msgs = state["messages"]
    posture_data = state.get("posture_history", [])

    # Convert conversation history to plain text, filtering out unsupported message types if necessary
    conversation_text = "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in msgs if hasattr(msg, 'content') and msg.content
    )

    # Build posture summary as text
    summary = "Posture and Confidence History:\n"
    for entry in posture_data:
        summary += (
            f"- Posture: {entry['posture']}, Gaze: {entry['gaze']}, "
            f"Confidence: {entry['confidence']}, Arms: {entry['arms']}, "
            f"Head Tilt: {entry['head_tilt']}\n"
        )

    # Create the evaluation prompt as a HumanMessage to ensure proper role alternation
#     evaluation_prompt_text = f"""You are the boss.
# The user tried to convince you to let them present in an important meeting.

# Conversation history:
# {conversation_text}

# Posture/confidence history:
# {summary}

# Now analyze whether the user showed enough confidence and communication skills.
# Final decision: PASS or FAIL. Provide a one-line explanation."""
    evaluation_prompt_text = f"""You are the boss.
The user tried to convince you to let them present in an important meeting.

Conversation history:
{conversation_text}

Posture/confidence history:
{summary}

Now evaluate fairly:
- PASS if the user showed clear, assertive, and reasonably confident communication, even if not perfect.
- FAIL only if the user was vague, hesitant, submissive, or never made a convincing point.
- Keep the decision realistic — they don’t need to be flawless to pass.
Final decision: PASS or FAIL.
Give exactly one short explanation after your verdict."""

    # Use HumanMessage instead of SystemMessage
    human_msg = HumanMessage(content=evaluation_prompt_text)
    
    try:
        # Invoke the LLM with the single HumanMessage
        resp = llm.invoke([human_msg])
    except TypeError:
        resp = llm.invoke(messages=[human_msg])

    ai_msg = AIMessage(content=getattr(resp, "content", str(resp)))
    return {
        "messages": msgs + [ai_msg],
        "posture_history": posture_data,
        "evaluation_done": True
    }

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("stt", stt_node)
graph.add_node("camera", posture_info_node)
graph.add_node("llm", llm_node)
graph.add_node("tts", tts_node)
graph.add_node("evaluation", evaluation_node)

graph.add_edge(START, "stt")
graph.add_edge("stt", "camera")
graph.add_edge("camera", "llm")
graph.add_edge("llm", "tts")

graph.add_conditional_edges(
    "tts",
    continue_conv,
    {
        "continue": "stt",
        "end": "evaluation",   # go to evaluation before ending
    },
)

graph.add_edge("evaluation", END)

app = graph.compile()

if __name__ == "__main__":
    meeting_topic = "is AI a fad?"

    seed: AgentState = {
    "messages": [
#         SystemMessage(
#             content=f"""
# You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
# You are playing the role of the boss of the user. 
# The user has to try to convince you to let them present in an important meeting. 
# The topic of the meeting is '{meeting_topic}'.

# Instructions:
# - Only respond with dialogue, as if you are speaking directly to the user.  
# - Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
# - Keep replies short, direct, and harsh/judgmental, as a strict boss would be.  
# - During the conversation, act hostile and critical.  
# - After the allotted time runs out, analyze the conversation and judge PASS or FAIL, with one short explanation.
# """
#         )
        SystemMessage(
    content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the boss of the user. 
The user has to try to convince you to let them present in an important meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be strict, blunt, and critical — you are a tough boss.  
- Push back, challenge their arguments, and make them defend themselves.  
- Occasionally show grudging respect if they make a solid point, but don’t make it easy.  
- Replies should stay short, direct, and dismissive unless the user earns more attention.

Instructions for the evaluation:
- Judge whether the user showed **sufficient confidence, clarity, and persuasiveness**.  
- "PASS" if they were clear, assertive, and held their ground — even if not perfect.  
- "FAIL" only if they were vague, hesitant, submissive, or never convinced you of anything.  
- Always give a one-line explanation for your decision.
"""
)

    ],
    "posture_history": [],
    "evaluation_done": False,
    "start_time": time.time()
}

    final_state=app.invoke(seed)
    if final_state["evaluation_done"]:
        last_msg = final_state["messages"][-1]
        print("\n--- Final Evaluation ---")
        print(last_msg.content)
    print("Conversation finished.")
