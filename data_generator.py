import pandas as pd
import numpy as np
import os

_cached_normal = None
_cached_fraud = None

def _generate_synthetic_pca(n_samples: int) -> pd.DataFrame:
    """Generates synthetic random data matching the MLG-ULB Credit Card Fraud schema."""
    np.random.seed(None)
    
    data = {
        'Time': np.random.uniform(0, 172792, size=n_samples).round(0),
        'Amount': np.random.exponential(scale=88, size=n_samples).round(2),
        'Class': np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03]) # 3% fraud for testing
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, size=n_samples)
        
    df = pd.DataFrame(data)
    # Sort by time to simulate a stream
    df = df.sort_values('Time').reset_index(drop=True)
    return df

def get_cached_data(file_path):
    global _cached_normal, _cached_fraud
    if _cached_normal is None or _cached_fraud is None:
        print(f"Loading full {file_path} into memory for balanced sampling...")
        df = pd.read_csv(file_path)
        _cached_normal = df[df['Class'] == 0]
        _cached_fraud = df[df['Class'] == 1]
    return _cached_normal, _cached_fraud

def generate_live_batch(batch_size: int = 15, file_path: str = 'creditcard.csv', current_step: int = 0) -> pd.DataFrame:
    """
    Reads from the real creditcard.csv using a cached memory strategy.
    Forces a ~30:1 legitimate-to-fraud ratio so the dashboard shows interesting alerts.
    """
    if os.path.exists(file_path):
        normal_df, fraud_df = get_cached_data(file_path)
        
        # Force exactly 1 or 2 fraud cases per batch (approx 15:1 to 7:1 ratio) to guarantee an exciting demo
        num_fraud = np.random.choice([1, 2])
            
        num_normal = batch_size - num_fraud
        
        batch = pd.concat([
            normal_df.sample(num_normal),
            fraud_df.sample(num_fraud)
        ])
        
        # Shuffle the batch so fraud isn't always at the bottom
        batch = batch.sample(frac=1).reset_index(drop=True)
        return batch
    else:
        df = _generate_synthetic_pca(batch_size)
        return df

if __name__ == "__main__":
    print("Generating synthetic PCA data for testing...")
    df_test = _generate_synthetic_pca(100)
    print(df_test.head())
