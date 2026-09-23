import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def build_model():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    return model

def train_and_save(model_path: str):
    # Dummy training data
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"])
    y_train = np.random.randint(0, 2, 100)
    
    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
