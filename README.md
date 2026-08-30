# Multi-Agent Medical Image Classification System with Grad-CAM Explainability and LLM-Driven Automated Reporting

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3.17-blueviolet.svg)

## 💡 The Core Idea
MedVision AI is a cutting-edge **Multi-Agent Artificial Intelligence System** designed to bridge the gap between black-box computer vision models and clinical reliability. 

Traditional AI in healthcare often provides a prediction without explanation, leading to mistrust among medical professionals. MedVision AI solves this by employing an orchestra of specialized AI agents to not only predict diseases using Vision Transformers (ViT) but also visually explain the prediction (Grad-CAM), draft a human-readable clinical report using an LLM, and **automatically audit its own report** for medical accuracy and hallucinations before presenting it to the user.

---

## 🏗️ System Architecture & Multi-Agent Flow

The system is divided into two major phases: **The Training Pipeline** (Agents 1-5) and **The Inference & Audit Pipeline** (Agents 6-8).

### 1. The Multi-Agent Architecture Graph

```mermaid
graph TD
    subgraph Phase 1: Training & Data Pipeline
    A1[Agent 1: Data Cleaner] --> A2[Agent 2: Preprocessor]
    A2 --> A3[Agent 3: Model Selector]
    A3 --> A4[Agent 4: Trainer]
    A4 --> A5[Agent 5: Evaluator]
    end

    subgraph Phase 2: Inference & Audit Pipeline
    U((User / Clinical Staff)) -->|Uploads Image| UI[Streamlit Frontend]
    UI -->|API Request| A6[Agent 6: Vision Classifier & Grad-CAM]
    A5 -.->|Provides trained weights| A6
    A6 -->|Visual Findings & Confidence| A7[Agent 7: Report Draft Generator]
    A7 -->|Draft Medical Report| A8[Agent 8: LLM-as-a-Judge]
    
    A8 -->|Fails Audit| A7
    A8 -->|Passes Audit| UI
    end
    
    style A1 fill:#1f77b4,color:#fff
    style A2 fill:#1f77b4,color:#fff
    style A3 fill:#1f77b4,color:#fff
    style A4 fill:#1f77b4,color:#fff
    style A5 fill:#1f77b4,color:#fff
    
    style A6 fill:#ff7f0e,color:#fff
    style A7 fill:#2ca02c,color:#fff
    style A8 fill:#d62728,color:#fff
```

### 2. Inference Data Flowchart

This sequence diagram illustrates exactly what happens when a user uploads a medical image to the Streamlit UI.

```mermaid
sequenceDiagram
    participant User
    participant Streamlit (UI)
    participant Agent 6 (Vision)
    participant Agent 7 (Drafter)
    participant Agent 8 (Judge)

    User->>Streamlit (UI): Uploads Medical Scan
    Streamlit (UI)->>Agent 6 (Vision): Send image for inference
    Agent 6 (Vision)->>Agent 6 (Vision): Extract ViT Features
    Agent 6 (Vision)->>Agent 6 (Vision): Generate Grad-CAM Heatmap
    Agent 6 (Vision)-->>Agent 7 (Drafter): Send Predictions & Visuals
    
    rect rgb(200, 220, 240)
        Note right of Agent 7 (Drafter): The Self-Correction Loop
        Agent 7 (Drafter)->>Agent 8 (Judge): Drafts initial clinical report
        Agent 8 (Judge)->>Agent 8 (Judge): Scores report against 10-pt clinical criteria
        alt Score < Threshold
            Agent 8 (Judge)-->>Agent 7 (Drafter): Reject & provide actionable feedback
            Agent 7 (Drafter)->>Agent 8 (Judge): Generate revised report
        end
    end
    
    Agent 8 (Judge)-->>Streamlit (UI): Return validated report & quality scores
    Streamlit (UI)-->>User: Display Beautiful, Hallucination-Free Dashboard
```

---

## 🔬 Core Components Detailed

### 👁️ Explainable AI (Grad-CAM)
Deep learning models are notoriously opaque. MedVision uses **Gradient-weighted Class Activation Mapping (Grad-CAM)**. When the Vision Transformer predicts a condition, Agent 6 extracts the gradients flowing into the final attention block to produce a heatmap. This highlights exactly which pixels (e.g., a specific tumor region in an MRI) led to the AI's decision.

### ⚖️ The "LLM-as-a-Judge" (Agent 8)
Large Language Models are prone to hallucinations—inventing patient data or diagnosing conditions not present in the image. 
Agent 8 acts as a strict clinical auditor. It takes the draft from Agent 7 and evaluates it on a **point-wise clinical checklist**:
1. Are there contradictions with the Vision Model?
2. Did it hallucinate patient age/gender?
3. Is the tone professional?

If the report scores below the acceptable threshold, Agent 8 forces Agent 7 to rewrite it.

---

## 📁 Directory Structure
```text
Major_project/
├── agents/             # The logic for Agents 1 through 5 (Training Pipeline)
├── backend/            # FastAPI backend orchestrating Agents 6, 7, and 8
├── config/             # Environment and system configurations (LLM thresholds, etc.)
├── datasets/           # Raw and cleaned medical image datasets
├── explainability/     # Grad-CAM heatmap generation algorithms
├── frontend/           # The Streamlit interactive user interface
├── llm/                # Prompts, parsers, and LangChain wrappers for Agents 7 & 8
├── models/             # PyTorch model architectures and saved weights
└── outputs/            # Generated heatmaps, logs, and validated JSON medical reports
```

---

## 🚀 Execution Guide

Follow these commands to run the project. You can either use the **Quick Start** to jump straight to the UI, or run the full training pipeline.

### ⚡ Quick Start (Inference & UI Only)
If you don't want to train the model from scratch, you can download our pre-trained 343MB Vision Transformer weights and jump straight into the application.

1. **Download the pre-trained weights:**
   ```bash
   python download_weights.py
   ```
2. **Start the FastAPI Backend:**
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. **Start the Streamlit Frontend (in a new terminal):**
   ```bash
   streamlit run frontend/Home.py
   ```
*Open your browser to `http://localhost:8501` to use the system!*

---

### 🏗️ Advanced: Full Training & Data Pipeline
If you want to train the model yourself, run these sequentially to prepare data and train the vision model.
```bash
# 1. Agent 1 (Data Cleaning): Automatically cleans and validates raw dataset
python agents/agent1_data_cleaning.py

# 2. Agent 2 (Data Preprocessing): Resizes and normalizes images for the ViT
python agents/agent2_preprocessing.py

# 3. Agent 3 (Model Selection): Initializes architecture and hyperparameters
python agents/agent3_model_selection.py

# 4. Agent 4 (Model Training): Trains the Vision Transformer
python agents/agent4_training.py

# 5. Agent 5 (Model Evaluation): Tests model against validation data
python agents/agent5_evaluation.py
```

## 🛠️ Technology Stack
- **Backend Orchestration:** FastAPI, Uvicorn, LangChain
- **Computer Vision:** PyTorch, torchvision, OpenCV, Grad-CAM
- **Language Models:** Groq (Llama-3/GPT integration)
- **Frontend / UI:** Streamlit, Custom HTML/CSS
- **Data Management:** Pandas, Pillow, NumPy