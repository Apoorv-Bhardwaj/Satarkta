import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from satarkta_ml import SatarktaModel
from data_generator import generate_live_batch
import json

# Set up page config
st.set_page_config(
    page_title="FinGuard Accelerate | Satarkta",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;'>
        made with ❤️ by <a href='https://github.com/Apoorv-Bhardwaj' target='_blank'>Apoorv</a>
    </div>
    """, 
    unsafe_allow_html=True
)

# Initialize ML Engine
@st.cache_resource
def load_ml_engine():
    try:
        return SatarktaModel('model.json')
    except FileNotFoundError:
        st.error("Model artifact not found. Please run `python train_model.py` first.")
        st.stop()

ml_engine = load_ml_engine()

# Initialize Gemini AI
def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.warning("GEMINI_API_KEY environment variable not found. AI Co-Pilot disabled.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash', 
                                      system_instruction="You are a Senior Fraud Operations Assistant named Satarkta. You analyze financial transaction data to identify potential fraud. The data uses PCA-transformed features (V1-V28). Be concise, professional, and do not use emojis.")
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini: {str(e)}")
        return None

ai_model = setup_gemini()

# State Management
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

if 'live_df' not in st.session_state:
    st.session_state.live_df = pd.DataFrame()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    if st.button("Fetch Next Transaction Batch", type="primary"):
        with st.spinner("Fetching and scoring..."):
            raw_batch = generate_live_batch(batch_size=15, current_step=st.session_state.current_step)
            # Remove ground truth label for inference realism if it exists
            if 'Class' in raw_batch.columns:
                raw_batch = raw_batch.drop(columns=['Class'])
            
            scored_batch = ml_engine.predict(raw_batch)
            st.session_state.live_df = scored_batch
            st.session_state.current_step += 15

# Main Dashboard
st.title("FinGuard Accelerate Dashboard")

if not st.session_state.live_df.empty:
    df = st.session_state.live_df
    
    # Operational Metrics
    total_tx = len(df)
    total_blocked = len(df[df['Action'] == 'BLOCKED'])
    total_flagged = len(df[df['Action'] == 'FLAGGED'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions (Current Batch)", total_tx)
    with col2:
        st.metric("Total Blocked", total_blocked)
    with col3:
        st.metric("Total Flagged", total_flagged)
        
    st.subheader("Live Transaction Stream")
    
    # DataFrame Styling
    def highlight_actions(row):
        action = row['Action']
        if action == 'BLOCKED':
            return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
        elif action == 'FLAGGED':
            return ['background-color: rgba(255, 255, 0, 0.2)'] * len(row)
        else:
            return [''] * len(row)
            
    # Hide V5 through V28 for cleaner UI
    display_cols = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'Risk_Score', 'Action']
    display_df = df[display_cols]
    
    styled_df = display_df.style.apply(highlight_actions, axis=1).format({
        'Amount': "{:.2f}",
        'Risk_Score': "{:.4f}",
        'V1': "{:.3f}", 'V2': "{:.3f}", 'V3': "{:.3f}", 'V4': "{:.3f}"
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    st.caption("Note: PCA features V5-V28 are hidden from the visual dashboard for clarity but are still used for ML scoring.")
else:
    st.info("Click 'Fetch Next Transaction Batch' in the sidebar to start the live stream.")

# AI Co-Pilot Interface
st.divider()
st.subheader("AI Co-Pilot")

if ai_model is None:
    st.warning("AI Co-Pilot is currently unavailable due to missing API key.")
else:
    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat Input
    prompt = st.chat_input("Ask a question about the current transaction batch...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    if not st.session_state.live_df.empty:
                        # Send full dataframe context including hidden columns to AI
                        context_json = st.session_state.live_df.to_json(orient="records")
                        full_prompt = f"Context (Live DataFrame State as JSON): {context_json}\n\nUser Question: {prompt}"
                    else:
                        full_prompt = f"Context: No transactions have been fetched yet.\n\nUser Question: {prompt}"
                        
                    response = ai_model.generate_content(full_prompt)
                    st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error communicating with AI: {str(e)}")
