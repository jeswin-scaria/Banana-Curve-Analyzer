# 🍌 Claude Context Package — Banana Geometry Lab

This folder contains the complete backend codebase of **CHILL ETHAKAAA** formatted for effortless uploading to **Claude (Anthropic)** or any other LLM.

---

## 📁 Files Included in this Folder

1. **[`CLAUDE_BACKEND_CONTEXT.md`](file:///d:/Banana-Curve-Analyzer/backend_context/CLAUDE_BACKEND_CONTEXT.md)**  
   - **Recommended for Claude chat / Claude Project Knowledge**.  
   - Contains the full mathematical explanation, computer-vision architecture diagram, formula definitions, API documentation, and the complete source code of all backend modules (`src/config.py`, `src/preprocessing.py`, `src/skeleton.py`, `src/analyzer.py`, `src/visualization.py`, `src/cli.py`, `requirements.txt`).
   - Drag-and-drop directly into a Claude chat or add as a Project Knowledge source file.

2. **[`backend_bundle.py`](file:///d:/Banana-Curve-Analyzer/backend_context/backend_bundle.py)**  
   - **Standalone Single-File Python Bundle**.  
   - All backend modules merged into one standalone, self-contained Python file without inter-module import friction.  
   - Fully executable (`python backend_context/backend_bundle.py --image samples/curved_banana.png --json`).
   - Ideal if you want Claude to review or run pure Python code directly.

---

## 🚀 How to Upload to Claude

### Option A: Upload Markdown Document (Best for Questions, Architecture & Reasoning)
1. Open [Claude.ai](https://claude.ai).
2. Start a new conversation (or open a Project).
3. Click the paperclip / attachment icon 📎.
4. Select `d:\Banana-Curve-Analyzer\backend_context\CLAUDE_BACKEND_CONTEXT.md`.
5. Ask your questions (e.g., *"Analyze this computer vision pipeline and help me optimize the spline fitting"*).

### Option B: Upload Python Code Bundle (Best for Code Refactoring / Porting)
1. In Claude, upload `d:\Banana-Curve-Analyzer\backend_context\backend_bundle.py`.
2. Claude will have all classes, functions, and imports in a single file.
