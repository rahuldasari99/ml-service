from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import math
import traceback

# =====================================================
#  Load ETA MODEL ONLY
# =====================================================
ETA_MODEL_PATH = "best_eta_model.joblib"

def safe_load(path):
    try:
        model = joblib.load(path)
        print(f"[ML] Loaded model: {path}")
        return model
    except Exception as e:
        print(f"[ML] Failed to load ETA Model: {e}")
        return None

eta_model = safe_load(ETA_MODEL_PATH)

# =====================================================
#  No Final Price Model (Removed)
# =====================================================
final_price_model = None  # Always fallback surge logic


# =====================================================
#  FastAPI setup
# =====================================================
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="FixRoute ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
#   SCHEMAS
# =====================================================
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


# =====================================================
#  Utility Function
# =====================================================
def haversine_km(lat1, lon1, lat2, lon2):
    if lat2 is None or lon2 is None:
        return 9999.0

    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# =====================================================
#  ETA Prediction Endpoint /predict  (UNCHANGED)
# =====================================================
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
                "full_name": s.full_name or "",
                "location_lat": float(s.location_lat),
                "location_lng": float(s.location_lng)
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
                "eta_predicted": float(r["eta_predicted"]),
                "location_lat": float(r["location_lat"]),
                "location_lng": float(r["location_lng"])
            })

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        return {"results": []}


# =====================================================
#  FINAL PRICE Endpoint (ONLY FALLBACK SURGE LOGIC)
# =====================================================
@app.post("/predict_price")
def calculate_price(req: PriceRequest):
    try:
        # 1. Distance
        dist = haversine_km(req.User_Lat, req.User_Lng, req.Tech_Lat, req.Tech_Lng)

        # 2. ETA estimation
        eta_minutes = round((dist / 30) * 60, 2)

        # 3. Prepare input (kept same for compatibility)
        df = pd.DataFrame([{
            "Service_Name": req.Service_Name,
            "User_Lat": req.User_Lat,
            "User_Lng": req.User_Lng,
            "Tech_Lat": req.Tech_Lat,
            "Tech_Lng": req.Tech_Lng,
            "Final_Distance_KM": dist,
            "Final_ETA_Minutes": eta_minutes
        }])

        # 4. FALLBACK surge logic ONLY (No ML)
        handling_surge = round(eta_minutes * 0.5, 2)

        # 5. Final Price
        final_price = req.Base_Charge + handling_surge + float(req.Spare_Part_Price or 0)

        return {
            "status": "success",
            "Distance_KM": round(dist, 2),
            "ETA_Minutes": eta_minutes,
            "Handling_Surge": handling_surge,
            "Base_Charge": req.Base_Charge,
            "Spare_Part_Price": req.Spare_Part_Price,
            "Final_Price": round(final_price, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# =====================================================
# Root homepage
# =====================================================
@app.get("/")
def home():
    return {"service": "FixRoute ML Service", "status": "running"}


# =====================================================
# Run Server
# =====================================================
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
