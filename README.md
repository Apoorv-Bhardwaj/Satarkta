---
title: Satarkta Pipeline
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Satarkta Pipeline

**Real-Time Financial Fraud Detection & AI Investigation Co-Pilot**

Project Satarkta is a state-of-the-art financial security dashboard designed to intercept fraudulent transactions in real-time. It pairs a lightning-fast XGBoost inference engine with a Gemini-powered AI Co-Pilot to not only block fraud, but to clearly explain *why* it was blocked using interpretable transaction features.

---

##  Key Features

* **High-Speed Inference Engine:** Utilizes an optimized XGBoost classifier to score incoming transactions in milliseconds.
* **Interpretable Fraud Detection:** Trained on the industry-standard **PaySim** dataset (Mobile Money Fraud). The model analyzes understandable metrics like transaction type, transfer amount, and account balances before and after the transaction.
* **Live Streaming Simulation:** The dashboard continuously fetches and scores new batches of transactions, acting as a real-time monitor for fraud operations teams.
* **AI Investigation Co-Pilot (Satarkta):** Powered by **Google's Gemini 1.5 Flash**. Fraud analysts can chat directly with the dashboard. The AI interprets the live transaction data and explains anomalies (e.g., "This transaction was flagged because it was a massive TRANSFER that drained the user's origin balance down to zero").
* **BYOK (Bring Your Own Key):** Secure integration that requires users to provide their own Gemini API Key directly in the UI, preventing abuse of shared open keys on public deployments.
* **Automated CI/CD:** Fully integrated GitHub Actions workflow that automatically syncs and deploys the latest codebase to Hugging Face Spaces on every push without overwriting manually hosted data.

---

##  Architecture & Tech Stack

- **Frontend & UI:** Streamlit (Python)
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
- **Generative AI:** Google Gemini API (`google-generativeai`)
- **Data Pipeline:** Hugging Face Datasets (`datasets`)
- **Deployment & Hosting:** Hugging Face Spaces
- **CI/CD:** GitHub Actions

---

##  Local Setup & Execution

### 1. Prerequisites
Ensure you have Python 3 installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. The Dataset (PaySim)
The application handles data automatically! By using the `datasets` library, it seamlessly pulls the standard PaySim dataset (`theman10/paysim`) from the Hugging Face Hub. You no longer need to manually download or manage large CSV files.

### 3. Launch the Dashboard
Run the Streamlit server locally:
```bash
streamlit run app.py
```

### 4. API Keys
When the dashboard loads, navigate to the sidebar's **Control Panel** and securely enter your Gemini API Key in the "Configuration" section to enable the AI Co-Pilot.

---

## ☁️ Deployment (GitHub to Hugging Face Spaces)

This project features a zero-touch deployment pipeline using a custom Hugging Face Hub sync script. 

1. Create a **Docker / Streamlit** Space on Hugging Face.
2. In your GitHub repository **Settings > Secrets and variables > Actions**, add your Hugging Face Access Token as `HF_TOKEN`.
3. Update `.github/workflows/sync_to_hf.yml` with your HF username and Space name if necessary.
4. Push to the `main` branch. GitHub Actions will automatically push the code to Hugging Face, building and deploying the app instantly!

---

made with ❤️ by [Apoorv](https://github.com/Apoorv-Bhardwaj)
