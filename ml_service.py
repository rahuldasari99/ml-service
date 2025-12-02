
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import math
import traceback

# ----------------------------------------------
# Load ETA Model
# ----------------------------------------------
ETA_MODEL_PATH = "best_eta_model.joblib"

def safe_load(path):
    try:
        model = joblib.load(path)
        print(f"[ML] Loaded model: {path}")
        return model
    except Exception as e:
        print(f"[ML] Failed to load {path}: {e}")
        return None

eta_model = safe_load(ETA_MODEL_PATH)

# ----------------------------------------------
# Load FINAL PRICE MODEL
# ----------------------------------------------
FINAL_MODEL_PATH = "final_price_model.joblib"

try:
    final_price_model = joblib.load(FINAL_MODEL_PATH)
    print("[ML] Loaded final_price_model.joblib successfully")
except Exception as e:
    print("[ML] ERROR loading final_price_model.joblib:", e)
    final_price_model = None


# ----------------------------------------------
# FastAPI setup
# ----------------------------------------------
app = FastAPI(title="FixRoute ML Service")

from fastapi.middleware.cors import CORSMiddleware

# ENABLE CORS FOR FRONTEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- NEW FINAL PRICE ENDPOINT ----------
class BillingPayload(BaseModel):
    Service_Name: str
    User_Lat: float
    User_Lng: float
    Tech_Lat: float
    Tech_Lng: float
    Base_Charge: float
    Spare_Part_Price: float = 0.0

@app.post("/predict_price")
def predict_price(req: BillingPayload):
    try:
        # DISTANCE
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_km = haversine(
            req.User_Lat, req.User_Lng,
            req.Tech_Lat, req.Tech_Lng
        )
        distance_km = round(distance_km, 2)

        # ETA using your existing model
        df = pd.DataFrame([{
            "distance_km": distance_km,
            "base_cost": req.Base_Charge,
            "rating": 4.5,   # fallback
            "technician_charges": req.Base_Charge,
            "technician_rating": 4.5,
            "service_type": req.Service_Name,
            "vehicle_type": "mechanic",
        }])

        eta_pred = eta_model.predict(df)[0]
        eta_pred = round(float(eta_pred), 2)

        # Surge = ETA / 2 (example)
        surge = round(eta_pred * 0.5, 2)

        # FINAL PRICE
        final_price = round(req.Base_Charge + req.Spare_Part_Price + surge, 2)

        return {
            "status": "success",
            "Distance_KM": distance_km,
            "ETA_Minutes": eta_pred,
            "Handling_Surge": surge,
            "Base_Charge": req.Base_Charge,
            "Spare_Part_Price": req.Spare_Part_Price,
            "Final_Price": final_price
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ----------------------------------------------
# SCHEMAS
# ----------------------------------------------
class Serviceman(BaseModel):
    id: str
    full_name: Optional[str] = None
    base_cost: Optional[float] = 0.0
    rating: Optional[float] = 0.0
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None

class PredictRequest(BaseModel):
    user_lat: float
    user_lng: float
    service_type: Optional[str] = ""
    servicemen: List[Serviceman]

class PriceRequest(BaseModel):
    Service_Name: str
    User_Lat: float
    User_Lng: float
    Tech_Lat: float
    Tech_Lng: float
    Base_Charge: float
    Spare_Part_Price: Optional[float] = 0.0


# ----------------------------------------------
# Haversine distance
# ----------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    if lat2 is None or lon2 is None:
        return 9999.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ----------------------------------------------
# ETA Prediction Endpoint
# ----------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        user_lat = float(req.user_lat)
        user_lng = float(req.user_lng)
        service = req.service_type or ""

        rows = []

        for s in req.servicemen:
            dist = haversine_km(user_lat, user_lng, s.location_lat, s.location_lng)

            rows.append({
                "distance_km": dist,
                "base_cost": float(s.base_cost or 0.0),
                "rating": float(s.rating or 0.0),
                "technician_charges": float(s.base_cost or 0.0),
                "technician_rating": float(s.rating or 0.0),
                "service_type": service,
                "vehicle_type": "mechanic",
                "id": s.id,
                "full_name": s.full_name or ""
            })

        df = pd.DataFrame(rows)

        preds = eta_model.predict(df)
        df["eta_predicted"] = preds
        df = df.sort_values("eta_predicted")

        results = []
        for _, r in df.iterrows():
            results.append({
                "id": r["id"],
                "full_name": r["full_name"],
                "distance_km": float(r["distance_km"]),
                "base_cost": float(r["base_cost"]),
                "rating": float(r["rating"]),
                "eta_predicted": float(r["eta_predicted"])
            })

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        return {"results": []}


# ----------------------------------------------
# FINAL PRICE PREDICTION ENDPOINT
# ----------------------------------------------
@app.post("/predict_price")
def predict_price(req: PriceRequest):
    try:
        # 1. Distance
        dist = haversine_km(req.User_Lat, req.User_Lng, req.Tech_Lat, req.Tech_Lng)

        # 2. ETA estimation
        avg_speed_km_h = 30
        eta_minutes = round((dist / avg_speed_km_h) * 60, 2)

        # 3. ML model input
        df = pd.DataFrame([{
            "Service_Name": req.Service_Name,
            "User_Lat": req.User_Lat,
            "User_Lng": req.User_Lng,
            "Tech_Lat": req.Tech_Lat,
            "Tech_Lng": req.Tech_Lng,
            "Final_Distance_KM": dist,
            "Final_ETA_Minutes": eta_minutes
        }])

        # 4. Predict surge using your final model
        handling_surge = float(final_price_model.predict(df)[0])

        # 5. Compute final price
        final_price = req.Base_Charge + handling_surge + float(req.Spare_Part_Price or 0)

        return {
            "status": "success",
            "Distance_KM": round(dist, 2),
            "ETA_Minutes": eta_minutes,
            "Handling_Surge": round(handling_surge, 2),
            "Base_Charge": req.Base_Charge,
            "Spare_Part_Price": req.Spare_Part_Price,
            "Final_Price": round(final_price, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    


# ----------------------------------------------
# Run server
# ----------------------------------------------
if __name__ == "__main__":
    import uvicorn, os

    port = int(os.environ.get("PORT", 8000))   # Render gives PORT dynamically

    uvicorn.run(app, host="0.0.0.0", port=port)
