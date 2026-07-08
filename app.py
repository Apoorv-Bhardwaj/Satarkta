import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types
from satarkta_ml import SatarktaModel
from data_generator import generate_live_batch

# Set up page config
st.set_page_config(
    page_title="Satarkta Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom footer
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown(
    """
    <div style='position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray; z-index: 999;'>
        made with ❤️ by <a href='https://github.com/Apoorv-Bhardwaj' target='_blank' style='text-decoration: none;'>Apoorv</a>
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

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    st.markdown("### Configuration")
    user_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to enable the Satarkta Bot.")
    st.divider()
    
    fetch_regular = st.button("Fetch Next Transaction Batch", type="primary", use_container_width=True)
    fetch_malicious = st.button("Fetch Malicious Transaction", use_container_width=True)

# Initialize Gemini AI
def setup_gemini():
    api_key = user_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.sidebar.warning("API key not provided. Satarkta Bot disabled.")
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini: {str(e)}")
        return None

ai_client = setup_gemini()

# State Management
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

if 'live_df' not in st.session_state:
    st.session_state.live_df = pd.DataFrame()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

import time

if fetch_regular or fetch_malicious:
    with st.spinner("Fetching and scoring..."):
        time.sleep(0.5)
        
        raw_batch = generate_live_batch(
            batch_size=15, 
            current_step=st.session_state.current_step,
            force_malicious=bool(fetch_malicious)
        )
        
        if 'isFraud' in raw_batch.columns:
            raw_batch = raw_batch.drop(columns=['isFraud'])
        
        scored_batch = ml_engine.predict(raw_batch)
        st.session_state.live_df = scored_batch
        st.session_state.current_step += 15

# Main Dashboard
st.title("Satarkta Pipeline (PaySim Dataset)")

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
            
    display_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'Risk_Score', 'Action']
    display_df = df[display_cols]
    
    styled_df = display_df.style.apply(highlight_actions, axis=1).format({
        'amount': "{:.2f}",
        'oldbalanceOrg': "{:.2f}",
        'newbalanceOrig': "{:.2f}",
        'oldbalanceDest': "{:.2f}",
        'newbalanceDest': "{:.2f}",
        'Risk_Score': "{:.4f}"
    })
    
    st.dataframe(styled_df, width="stretch", height=400)
else:
    st.info("Click 'Fetch Next Transaction Batch' in the sidebar to start the live stream.")

# Satarkta Bot Interface
st.divider()
st.subheader("Satarkta Bot")

if ai_client is None:
    st.warning("Satarkta Bot is currently unavailable due to missing API key. Please enter your Gemini API key in the sidebar.")
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
                        context_json = st.session_state.live_df.to_json(orient="records")
                        schema_desc = (
                            "Data Schema (PaySim Mobile Money Fraud):\n"
                            "- step: 1 step represents 1 hour of time.\n"
                            "- type: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).\n"
                            "- amount: Transaction amount in local currency.\n"
                            "- oldbalanceOrg / newbalanceOrig: Sender's account balance before and after the transaction.\n"
                            "- oldbalanceDest / newbalanceDest: Recipient's account balance before and after the transaction.\n"
                            "- Risk_Score: The machine learning model's predicted probability of fraud (0.0 to 1.0).\n"
                            "- Action: 'FLAGGED' or 'BLOCKED' if the Risk_Score exceeds certain thresholds.\n"
                            "Fraud often happens when transfers or cash-outs rapidly drain an account to zero (e.g., amount == oldbalanceOrg)."
                        )
                        full_prompt = f"{schema_desc}\n\nContext (Live DataFrame State as JSON): {context_json}\n\nUser Question: {prompt}"
                    else:
                        full_prompt = f"Context: No transactions have been fetched yet.\n\nUser Question: {prompt}"
                        
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a Senior Fraud Operations Assistant named Satarkta. You analyze financial transaction data to identify potential fraud. Explain anomalies clearly using the provided schema. Be concise, professional, and do not use emojis."
                        )
                    )
                    st.markdown(response.text)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Error communicating with AI: {error_msg}")
