from fastapi import FastAPI
from app.schema import TransactionInput
from app.model import predict_fraud, best_parameter
from fastapi.encoders import jsonable_encoder

app = FastAPI()


@app.post("/predict")
def predict_transaction(data: TransactionInput):
    features = data.to_numpy()  # define this in schema.py
    pred = predict_fraud(features)
    return {"prediction": int(pred)}
