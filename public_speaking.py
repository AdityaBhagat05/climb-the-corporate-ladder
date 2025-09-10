from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os


load_dotenv()

from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import detect_posture_and_confidence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)