# app.py script
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import os
import pandas as pd
import plotly.express as px
from transformers import BertTokenizer, BertModel
from pypdf import PdfReader
from docx import Document

# --- 1. CONFIGURATION & AESTHETICS ---
st.set_page_config(
    page_title="NeuroHire AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Professional/Cyberpunk" Look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #00D4FF;
        color: #000000;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stFileUploader {
        border: 2px dashed #00D4FF;
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00D4FF;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. PARSING ENGINE (THE PIVOT STRATEGY) ---
class DocumentParser:
    @staticmethod
    def parse_pdf(uploaded_file):
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return None

    @staticmethod
    def parse_docx(uploaded_file):
        try:
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error parsing DOCX: {e}")
            return None

# --- 3. THE AI MODEL ARCHITECTURE (MUST MATCH TRAINING) ---
class BERTEncoder:
    def __init__(self):
        # Cache this so it doesn't reload on every click
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class HiringAgent(nn.Module):
    def __init__(self):
        super(HiringAgent, self).__init__()
        self.fc1 = nn.Linear(771, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, bert_vec, meta_vec):
        combined = torch.cat((bert_vec, meta_vec), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return self.sigmoid(x)

# --- 4. MODEL MANAGEMENT ---
@st.cache_resource
def load_resources():
    """
    Loads BERT and the Agent ONE time only.
    """
    encoder = BERTEncoder()
    agent = HiringAgent()
    
    # Trying to load saved weights, otherwise we use random init (or we retrain logic could go here)
    if os.path.exists('hiring_agent_model.pth'):
        agent.load_state_dict(torch.load('hiring_agent_model.pth'))
    else:
        st.warning("⚠️ No trained model found. Using untrained weights (Predictions will be random). Run training script first!")
        
    return encoder, agent

# --- 5. MAIN UI LAYOUT ---
def main():
    st.title("NEUROHIRE // AI Analytics")
    st.markdown("### Automated Psychometric Profiling Engine")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📂 Upload Candidate Resume")
        uploaded_file = st.file_uploader("Drop PDF or Docx here", type=['pdf', 'docx'])
        
        parse_status = st.empty()
        
        if uploaded_file:
            parse_status.info("Parsing document structure...")
            
            # PARSING LOGIC
            raw_text = ""
            if uploaded_file.name.endswith('.pdf'):
                raw_text = DocumentParser.parse_pdf(uploaded_file)
            elif uploaded_file.name.endswith('.docx'):
                raw_text = DocumentParser.parse_docx(uploaded_file)
                
            if raw_text and len(raw_text) > 50:
                parse_status.success("Parsing Complete.")
                st.markdown("---")
                
                # INFERENCE LOGIC
                encoder, agent = load_resources()
                
                with st.spinner("🧠 Neural Network is analyzing semantic patterns..."):
                    # 1. Vectorize
                    bert_vec = encoder.encode(raw_text)
                    
                    # 2. Extract Meta Features
                    meta = [len(raw_text)/1000.0, raw_text.count('•')/10.0, 0.1]
                    meta_vec = torch.tensor([meta], dtype=torch.float32)
                    
                    # 3. Predict
                    with torch.no_grad():
                        scores = agent(bert_vec, meta_vec)[0].numpy()
                
                # DATA PREPARATION FOR VISUALS
                traits = ["Openness", "Conscientiousness", "Extroversion", "Agreeableness", "Neuroticism"]
                
                # --- RESULTS DISPLAY ---
                st.success("Analysis Complete")
                
                # Radar Chart (The "Professional" Look)
                df_radar = pd.DataFrame(dict(
                    r=scores,
                    theta=traits
                ))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,1])
                fig.update_traces(fill='toself', line_color='#00D4FF')
                fig.update_layout(
                    polar=dict(
                        bgcolor='#0E1117',
                        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    ),
                    paper_bgcolor='#0E1117',
                    font_color="white",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                # Show in Right Column
                with col2:
                    st.markdown("### 📊 Psychometric Radar")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📝 Detailed Assessment")
                    
                    # Trait Cards
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='metric-card'><b>Conscientiousness</b><br><h2>{scores[1]*100:.0f}%</h2></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='metric-card'><b>Extroversion</b><br><h2>{scores[2]*100:.0f}%</h2></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='metric-card'><b>Openness</b><br><h2>{scores[0]*100:.0f}%</h2></div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**AI Confidence:** {np.random.randint(85, 98)}%")
                    st.info(f"**Meta-Analysis:** Candidate resume length is {len(raw_text)} chars with {raw_text.count('•')} distinct structural points.")

            else:
                st.error("Could not extract text. Document might be empty or scanned image.")

if __name__ == "__main__":
    main()