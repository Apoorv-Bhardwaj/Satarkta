import pandas as pd
import xgboost as xgb
import os

class SatarktaModel:
    def __init__(self, model_path: str = 'model.json'):
        self.model_path = model_path
        self.model = xgb.XGBClassifier()
        self.features = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        self.load_model()
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact {self.model_path} not found. Please train the model first.")
        self.model.load_model(self.model_path)
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame of raw transactions, runs them through XGBoost,
        and returns a DataFrame with Risk_Score and Action appended.
        """
        X = df[self.features]
        
        # Get raw probabilities for the positive class (Class = 1)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        df['Risk_Score'] = probabilities
        
        # Apply thresholds for actions
        def assign_action(score):
            if score > 0.85:
                return "BLOCKED"
            elif score > 0.60:
                return "FLAGGED"
            else:
                return "APPROVED"
                
        df['Action'] = df['Risk_Score'].apply(assign_action)
        return df
