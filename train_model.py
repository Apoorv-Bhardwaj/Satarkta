import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from data_generator import _generate_synthetic_pca
import os

def train_and_save_model(data_path: str = 'creditcard.csv', model_path: str = 'model.json'):
    """Trains an XGBoost model on the credit card dataset and saves the artifact."""
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Generating synthetic random noise for testing...")
        df = _generate_synthetic_pca(n_samples=5000)
        print("NOTE: This model is trained on random noise. Use Kaggle to train on the real dataset.")
    else:
        df = pd.read_csv(data_path)

    features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost Classifier...")
    # NOTE for Kaggle users: 
    # To utilize Kaggle's parallel GPU, set device='cuda' below.
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',
        scale_pos_weight=99, # Handle extreme imbalance
        objective='binary:logistic',
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    # Predict probabilities for AUPRC
    y_scores = clf.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_scores)
    
    print(f"Model trained. Test AUPRC (Average Precision): {auprc:.4f}")

    clf.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
