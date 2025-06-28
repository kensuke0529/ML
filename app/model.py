import joblib
import numpy as np

model = joblib.load('fraud-detection/xgbclassifier.joblib')


def predict_fraud(features: np.ndarray) -> int:
    """Take in a 1D feature vector and return prediction (0 or 1)"""
    prediction = model.predict(features.reshape(1, -1))
    return int(prediction[0])
