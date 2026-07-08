import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from data_generator import _load_paysim_data

def train_and_save_model(model_path: str = 'model.json'):
    """Trains an XGBoost model on the PaySim dataset and saves the artifact."""
    
    print("Fetching PaySim dataset for training...")
    normal_df, fraud_df = _load_paysim_data()
    
    # Take a sample to keep training fast (e.g. 50k normal, all fraud)
    df = pd.concat([
        normal_df.sample(50000, random_state=42),
        fraud_df
    ]).reset_index(drop=True)
    
    # Convert 'type' to categorical for XGBoost
    df['type'] = df['type'].astype('category')
    
    features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                'oldbalanceDest', 'newbalanceDest']
    X = df[features]
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost Classifier on PaySim features...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',
        scale_pos_weight=50, # Handle imbalance
        objective='binary:logistic',
        enable_categorical=True,
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
