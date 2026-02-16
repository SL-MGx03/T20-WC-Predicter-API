from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from Win_Pred_Model import Win_Pred_Model  # IMPORTANT: must be your embedding model class


# -----------------------------
# Config paths (Render-friendly)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "T20-WC_model.pth"
PREP_PATH = BASE_DIR / "T20-WC_model_preprocess.joblib"


# -----------------------------
# Helpers
# -----------------------------
def _clean_city(city: str) -> str:
    city = "Unknown" if city is None else str(city)
    city = city.strip().lower()
    city = " ".join(city.split())  # collapse multiple spaces

    alias = {
        "bengaluru": "bangalore",
        "benagluru": "bangalore",
        "mumbai (wankhede)": "mumbai",
    }
    return alias.get(city, city)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")
    if not PREP_PATH.exists():
        raise RuntimeError(f"Missing preprocess file: {PREP_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = joblib.load(PREP_PATH)

    # Build model with the same sizes used in training
    # If your Win_Pred_Model __init__ needs more args (emb dims/hidden), add them to the pack and pass here.
    model = Win_Pred_Model(
        n_teams=pack["n_teams"],
        n_cities=pack["n_cities"],
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, pack, device


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="T20 Win Predictor API", version="1.0.0")

MODEL = None
PACK = None
DEVICE = None


@app.on_event("startup")
def _startup():
    global MODEL, PACK, DEVICE
    MODEL, PACK, DEVICE = load_artifacts()


# -----------------------------
# Request/Response models
# -----------------------------
class PredictRequest(BaseModel):
    team_a: str = Field(..., description="Team A name (must match training set spelling)")
    team_b: str = Field(..., description="Team B name (must match training set spelling)")
    city: str = Field(..., description="City name")
    first_inning_runs: float = Field(..., ge=0)
    first_inning_wkts: float = Field(..., ge=0, le=10)


class PredictResponse(BaseModel):
    team_a: str
    team_b: str
    city: str
    team_a_win_prob: float
    team_b_win_prob: float


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    """
    Returns the valid team/city values seen during training.
    Useful for dropdowns in your future website.
    """
    team_le = PACK["team_le"]
    city_le = PACK["city_le"]
    return {
        "n_teams": int(PACK["n_teams"]),
        "n_cities": int(PACK["n_cities"]),
        "teams": team_le.classes_.tolist(),
        "cities": city_le.classes_.tolist(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    team_le = PACK["team_le"]
    city_le = PACK["city_le"]
    scaler = PACK["scaler"]

    team_a = str(req.team_a).strip()
    team_b = str(req.team_b).strip()
    city = _clean_city(req.city)

    # Encode categories (error if unknown)
    try:
        team_a_id = int(team_le.transform([team_a])[0])
        team_b_id = int(team_le.transform([team_b])[0])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Unknown team name. Use GET /metadata to see allowed teams.",
        )

    try:
        city_id = int(city_le.transform([city])[0])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Unknown city name. Use GET /metadata to see allowed cities.",
        )

    # Scale numeric features
    num = np.array([[float(req.first_inning_runs), float(req.first_inning_wkts)]], dtype=np.float32)
    num_scaled = scaler.transform(num).astype(np.float32)

    # Torch tensors
    team_a_t = torch.tensor([team_a_id], dtype=torch.long, device=DEVICE)
    team_b_t = torch.tensor([team_b_id], dtype=torch.long, device=DEVICE)
    city_t = torch.tensor([city_id], dtype=torch.long, device=DEVICE)
    num_t = torch.tensor(num_scaled, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        logit = MODEL(team_a_t, team_b_t, city_t, num_t)
        prob_a = torch.sigmoid(logit).item()

    return PredictResponse(
        team_a=team_a,
        team_b=team_b,
        city=city,
        team_a_win_prob=float(prob_a),
        team_b_win_prob=float(1.0 - prob_a),
    )
