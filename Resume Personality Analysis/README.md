# 🧠 NeuroHire: AI-Powered Psychometric Profiling Engine

**NeuroHire** is an end-to-end AI system that predicts a candidate’s **Big Five Personality Traits**  
(**Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism**) directly from their resume.

Unlike simple keyword-based systems, NeuroHire leverages a **Multimodal Reinforcement Learning Agent** that fuses:
- **Semantic understanding** (BERT embeddings)
- **Structural layout analysis** (resume organization and formatting)

to generate **holistic, explainable HR insights**.

---

## 🚀 Key Features

### 🔍 Multimodal Analysis
- Combines **semantic text embeddings** (what you wrote)
- With **structural meta-features** (how you organized it)
- Produces a complete psychometric profile

### 🧑‍🏫 Knowledge Distillation
- Implements a **Teacher–Student architecture**
- A Large Language Model (Gemini / GPT-4 / DeepSeek) generates synthetic psychological labels
- A lightweight neural network is trained for **offline inference**

### 🧠 Reinforcement Learning Agent
- Custom **PyTorch RL agent**
- Trained to minimize loss against expert-level psychological profiling

### 🖥️ Interactive Dashboard
- Cyberpunk-styled **Streamlit UI**
- Real-time PDF parsing
- Dynamic **Plotly Radar Charts** for personality visualization

---

## 🛠️ Tech Stack

### Core AI
- PyTorch
- Transformers (BERT)
- Scikit-Learn

### Data Pipeline
- OpenAI API / Google Gemini API (label generation)

### Visualization
- Streamlit
- Plotly Express

### Parsing
- pypdf
- python-docx

---

## 🏗️ System Architecture

The system follows a **3-stage pipeline**:

### 1️⃣ Data Generation (The Teacher)
- **Input:** Raw resumes (CSV / PDF)
- **Process:**  
  LLM (Gemini / DeepSeek) acts as a *psychometrician* to generate personality labels
- **Output:**  
  `final_labeled_dataset.json`

---

### 2️⃣ Model Training (The Student)
- **Input:** Labeled JSON dataset
- **Process:**  
  PyTorch agent learns to map:
  - BERT vectors  
  - Structural meta-features (length, bullet density, formatting)
- **Output:**  
  `hiring_agent_model.pth` (trained weights)

---

### 3️⃣ Inference (The Application)
- **Input:** User-uploaded PDF resume
- **Process:**  
  Parsing → BERT encoding → RL agent prediction
- **Output:**  
  Radar chart + personality insights

---

## 💻 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/a7madv4d2/Resume-Personality-Analysis.git
cd Resume-Personality-Analysis
```

### 2️⃣ Install Dependencies
```bash
pip install torch transformers streamlit plotly pandas pypdf python-docx openai google-generativeai tqdm
```


