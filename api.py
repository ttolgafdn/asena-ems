from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Modeli yüklüyoruz
model = joblib.load('echo_ai_model.pkl')

app = FastAPI()

# 1. ARKADAŞININ GÖNDERDİĞİ FORMAT (Bu kısmı arkadaşının hatasındaki verilere göre güncelledim)
class SnifferData(BaseModel):
    mac: str
    rssi: float
    sensorLat: float
    sensorLon: float
    timestamp: int

@app.post("/predict_location")
def predict_location(data: SnifferData):
    # AI Girdi Hazırlığı
    input_df = pd.DataFrame([{
        'Senaryo_Tipi': 1,
        'Sniffer_X': data.sensorLat,
        'Sniffer_Y': data.sensorLon,
        'Sniffer_Z': 0.0,
        'RSSI_Anlik': data.rssi,
        'RSSI_Mean_5s': data.rssi,
        'RSSI_Std_5s': 0.0
    }])
    
    # Tahmin
    prediction = model.predict(input_df)[0]
    
    # --- KRİTİK DÜZELTME BURASI ---
    # Arkadaşının PredictionDTO sınıfı muhtemelen direkt şu alanları bekliyor:
    return {
        "gercekX": round(prediction[0], 4),
        "gercekY": round(prediction[1], 4),
        "gercekZ": round(prediction[2], 2),
        "gercekDerinlik": round(prediction[3], 2),
        "status": "OK"
    }