# Satarkta Pipeline

**Real-Time Financial Fraud Detection & AI Investigation Co-Pilot**

Project Satarkta is a state-of-the-art financial security dashboard designed to intercept fraudulent transactions in real-time. Built to handle highly imbalanced datasets, it pairs a lightning-fast XGBoost inference engine with a Gemini-powered AI Co-Pilot to not only block fraud, but to explain *why* it was blocked.

---

##  Key Features

* **High-Speed Inference Engine:** Utilizes an optimized XGBoost classifier trained on Kaggle's dual GPUs (`tree_method='hist'`, `device='cuda'`) to score incoming transactions in milliseconds.
* **Tackling Extreme Imbalance:** Trained specifically on the notoriously difficult MLG-ULB Credit Card Fraud dataset (where only 0.17% of transactions are fraudulent). Evaluated using AUPRC (Area Under the Precision-Recall Curve) to ensure genuine robustness against false positives.
* **Live Streaming Simulation:** The dashboard continuously fetches and scores new batches of transactions, acting as a real-time monitor for fraud operations teams.
* **AI Investigation Co-Pilot (Satarkta):** Powered by **Google's Gemini 1.5 Flash**. Fraud analysts can chat directly with the dashboard. The AI is silently fed the live JSON state of the transaction table, allowing it to interpret the complex PCA-transformed variables (V1-V28) and explain exactly why a specific transaction was flagged.
* **Automated CI/CD:** Fully integrated GitHub Actions workflow that automatically syncs and deploys the latest codebase to Hugging Face Spaces on every push.

---

##  Architecture & Tech Stack

- **Frontend & UI:** Streamlit (Python)
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
- **Generative AI:** Google Gemini API (`google-generativeai`)
- **Training Environment:** Kaggle Notebooks (GPU P100 / T4 x2)
- **Deployment & Hosting:** Hugging Face Spaces
- **CI/CD:** GitHub Actions

---

##  Local Setup & Execution

### 1. Prerequisites
Ensure you have Python 3 installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. The Dataset (MLG-ULB)
To simulate the live data stream locally, download the **Credit Card Fraud Detection** dataset by the Machine Learning Group - ULB from Kaggle. 
Place the `creditcard.csv` file directly in the root directory. *(Note: If this file is missing, the application will automatically generate synthetic random PCA noise so the UI can still be tested without crashing).*

### 3. API Keys
Set your Gemini API key in your terminal environment:
- **Windows:** `$env:GEMINI_API_KEY="your_api_key_here"`
- **Mac/Linux:** `export GEMINI_API_KEY="your_api_key_here"`

### 4. Launch the Dashboard
Run the Streamlit server:
```bash
streamlit run app.py
```

---

## ☁️ Deployment (GitHub to Hugging Face Spaces)

This project features a zero-touch deployment pipeline. 

1. Create a **Streamlit** Space on Hugging Face.
2. In your Space **Settings > Variables and secrets**, add your `GEMINI_API_KEY`.
3. In your GitHub repository **Settings > Secrets and variables > Actions**, add your Hugging Face Access Token as `HF_TOKEN`.
4. Update `.github/workflows/sync_to_hf.yml` with your HF username and Space name.
5. Push to the `main` branch. GitHub Actions will automatically push the code and the `model.json` artifact to Hugging Face, building and deploying the app instantly!

---

made with ❤️ by [Apoorv](https://github.com/Apoorv-Bhardwaj)
