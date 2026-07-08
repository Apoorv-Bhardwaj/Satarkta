import pandas as pd
import numpy as np
from datasets import load_dataset
import os

_cached_normal = None
_cached_fraud = None

def _load_paysim_data():
    """Loads the PaySim dataset from Hugging Face Hub (cached automatically)."""
    global _cached_normal, _cached_fraud
    if _cached_normal is None or _cached_fraud is None:
        print("Loading PaySim dataset from Hugging Face Hub...")
        # Load the dataset
        ds = load_dataset('theman10/paysim', split='train')
        df = ds.to_pandas()
        
        # We need standard features
        df = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                 'oldbalanceDest', 'newbalanceDest', 'isFraud']]
        
        _cached_normal = df[df['isFraud'] == 0]
        _cached_fraud = df[df['isFraud'] == 1]
    return _cached_normal, _cached_fraud

def generate_live_batch(batch_size: int = 15, current_step: int = 0, force_malicious: bool = False) -> pd.DataFrame:
    """Generates a batch by sampling from the PaySim dataset."""
    normal_df, fraud_df = _load_paysim_data()
    
    if force_malicious:
        num_fraud = np.random.choice([1, 2, 3])
    else:
        num_fraud = 0
        
    num_normal = batch_size - num_fraud
    
    batch = pd.concat([
        normal_df.sample(num_normal),
        fraud_df.sample(num_fraud) if num_fraud > 0 else pd.DataFrame()
    ])
    
    batch = batch.sample(frac=1).reset_index(drop=True)
    return batch

if __name__ == "__main__":
    print("Testing PaySim data loader...")
    df_test = generate_live_batch(5, force_malicious=True)
    print(df_test)
