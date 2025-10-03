<!-- Badges Section -->
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/unity-2022%2B-black" alt="Unity Version">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
  <img src="https://img.shields.io/badge/AI-Google%20Gemini-orange" alt="Google Gemini">
</p>


# Climb the Corporate Ladder

Welcome to the **future of public speaking training**.  
Climb the Corporate Ladder is a **Gen AI–powered gamified public speaking training experience** with multiple levels, challenges, and mini-games — designed to turn even the most hesitant speaker into a confident professional.

---

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [Who Is It For](#who-is-it-for)
- [What Makes It Different](#what-makes-it-different)
- [Highlights](#highlights)
- [How It Works](#how-it-works)
- [User Experience](#user-experience)
- [Game Flow](#game-flow)
- [Feasibility](#feasibility)
- [Challenges & Risks](#challenges--risks)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## 🏢 About the Project
Climb the Corporate Ladder is a **gamified training program** that blends **AI-powered feedback, immersive environments, and mini-games** to help individuals improve their public speaking and communication skills.  

---

## 👥 Who Is It For
This project is designed for:  
- Young professionals entering the job market  
- Students preparing for interviews and networking  
- Anyone looking to build confidence in high-pressure communication  

---

## 🚀 What Makes It Different
Unlike traditional workshops and online courses, this project:  
- Adapts to individual needs  
- Provides **real-time AI feedback**  
- Gamifies the learning process with engaging scenarios  

---

## 🌟 Highlights
- **LangGraph AI Agents** for dynamic feedback  
- **AI confidence scoring** with measurable progress  
- Evaluates **tone, clarity, and assertiveness**  
- **Posture, body language, and facial expression analysis** via video integration  

---

## ⚙️ How It Works
1. Audio input from user → transcribed with **OpenAI Whisper**  
2. Camera & posture analysis with **MediaPipe**  
3. Text + body data → processed by **Google Gemini (LLM)**  
4. LLM replies in character → audio generated via **Coqui-TTS**  
5. Evaluation node checks responses → updates pass/fail meter  
6. Conversation continues until time limit → final evaluation  

---

## 🎮 User Experience
- Built in **Unity Game Engine**  
- Immersive environments: **Office, Presentation Hall, Outdoor**  
- Interactive NPCs with unique personalities & memory  
- Mini-games to improve English skills (e.g., Hangman)  

---

## 🎮 Demo Game Flow

```
Start 
   ↓
Posture Game 
   ↓
Office Games (Pronunciation, Convincing, Hangman, NPC Interactions) 
   ↓
Evaluation 
   ↓
Convince the Boss 
   ↓
Final Presentation 
   ↓
End
```
---

## ✅ Feasibility
- **Technically Feasible**: Uses mature open-source AI tools  
- **Cost-effective**: Free tiers & open-source options  
- **Market Ready**: Aligned with EdTech gamification demand  

---

## ⚠️ Challenges & Risks
- Privacy concerns with voice/video data  
- AI accuracy for diverse accents  
- High dev time for branching storylines  
- Dependency on external APIs  

---
## ▶️ How to Run Our Game

1. **Download the code**
   - Go to the GitHub repository and download the project as a `.zip` file.
   - Unzip the file and place it in a folder of your choice.

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On **Windows**:
     ```bash
     .venv/Scripts/activate
     ```
   - On **Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Get a Google Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/) and click **Get API Key**.
   - Follow the instructions to create a free API key.

6. **Create a `.env` file**
   - Inside the game folder, create a new file named `.env`.
   - Paste your API key in the following format:
     ```
     GOOGLE_API_KEY="YOUR_API_KEY_HERE"
     ```

7. **Install OpenAI Whisper**
   - Download from [OpenAI Whisper GitHub Repo](https://github.com/openai/whisper).

8. **Start the backend server**
   ```bash
   uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
   ```

9. **Run the game logic**
   - In another terminal, navigate to the project folder and type:
     ```bash
     python nre.py
     ```

10. **Launch the game**
    - Double-click the Unity `.exe` file to start the game.  
    - Enjoy your fun learning experience!


## 📂 File Structure
```
├── .gitignore
├── audio_utils.py
├── backend.py
├── camera_utils.py
├── conv_agent.py
├── convince_boss.py
├── nre.py
├── public_speaking.py
└── requirements.txt
```

---

## 🛠️ Tech Stack
- **Game Engine**: Unity  
- **LLM**: Google Gemini  
- **Speech-to-Text**: OpenAI Whisper  
- **TTS**: Coqui-TTS  
- **Video Analysis**: MediaPipe  
- **Backend**: FastAPI (Uvicorn)  
- **Language**: Python  

---

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.  

---
