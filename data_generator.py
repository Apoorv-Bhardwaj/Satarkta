import pandas as pd
import numpy as np
import os

def _generate_synthetic_pca(n_samples: int) -> pd.DataFrame:
    """Generates synthetic random data matching the MLG-ULB Credit Card Fraud schema."""
    np.random.seed(None)
    
    data = {
        'Time': np.random.uniform(0, 172792, size=n_samples).round(0),
        'Amount': np.random.exponential(scale=88, size=n_samples).round(2),
        'Class': np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01]) # 1% fraud for testing
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, size=n_samples)
        
    df = pd.DataFrame(data)
    # Sort by time to simulate a stream
    df = df.sort_values('Time').reset_index(drop=True)
    return df

def generate_live_batch(batch_size: int = 15, file_path: str = 'creditcard.csv', current_step: int = 0) -> pd.DataFrame:
    """
    Attempts to read a batch from the real creditcard.csv if it exists locally.
    Otherwise, generates synthetic PCA data for testing the UI.
    """
    if os.path.exists(file_path):
        # Read a chunk from the real CSV
        try:
            # Skip rows to simulate streaming
            df = pd.read_csv(file_path, skiprows=range(1, current_step + 1), nrows=batch_size)
        except Exception:
            df = pd.read_csv(file_path, nrows=batch_size)
    else:
        df = _generate_synthetic_pca(batch_size)
        
    return df

if __name__ == "__main__":
    print("Generating synthetic PCA data for testing...")
    df_test = _generate_synthetic_pca(100)
    print(df_test.head())
