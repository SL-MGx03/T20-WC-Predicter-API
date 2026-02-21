from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from Win_Pred_Model import Win_Pred_Model


# -----------------------------
# Paths (Render-friendly)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "T20-WC_model.pth"
PREP_PATH = BASE_DIR / "T20-WC_model_preprocess.joblib"


# -----------------------------
# Cleaning helpers
# -----------------------------
def _clean_team(name: str) -> str:
    return str(name).strip()


def _clean_city(city: str) -> str:
    city = "Unknown" if city is None else str(city)
    city = city.strip().lower()
    city = " ".join(city.split())
    alias = {
        "bengaluru": "bangalore",
        "benagluru": "bangalore",
        "mumbai (wankhede)": "mumbai",
    }
    return alias.get(city, city)


# -----------------------------
# Load artifacts
# -----------------------------
def _torch_load_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a state_dict safely.
    Uses weights_only=True if supported by the installed torch.
    """
    try:
        # torch >= 2.4
        return torch.load(path, map_location=device, weights_only=True)  
    except TypeError:
        # older torch
        return torch.load(path, map_location=device)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model file: {MODEL_PATH}. "
            f"Make sure T20-WC_model.pth is committed to the repo and present on Render."
        )
    if not PREP_PATH.exists():
        raise RuntimeError(
            f"Missing preprocess file: {PREP_PATH}. "
            f"Make sure T20-WC_model_preprocess.joblib is committed to the repo and present on Render."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = joblib.load(PREP_PATH)

    # Rebuild model using saved dims if present (fallback to defaults)
    model = Win_Pred_Model(
        n_teams=int(pack["n_teams"]),
        n_cities=int(pack["n_cities"]),
        team_emb_dim=int(pack.get("team_emb_dim") or 8),
        city_emb_dim=int(pack.get("city_emb_dim") or 6),
    ).to(device)

    state = _torch_load_state_dict(MODEL_PATH, device)
    model.load_state_dict(state)
    model.eval()

    return model, pack, device


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="T20 Win Predictor API", version="1.1.0")

MODEL = None
PACK = None
DEVICE = None
LOAD_ERROR: Optional[str] = None


@app.on_event("startup")
def _startup():
    global MODEL, PACK, DEVICE, LOAD_ERROR
    try:
        MODEL, PACK, DEVICE = load_artifacts()
        LOAD_ERROR = None
    except Exception as e:
        # Keep service alive so /health shows the reason
        MODEL, PACK, DEVICE = None, None, None
        LOAD_ERROR = str(e)


# -----------------------------
# Pydantic models
# -----------------------------
class PredictRequest(BaseModel):
    team_a: str = Field(..., description="Team A (must match training spelling)")
    team_b: str = Field(..., description="Team B (must match training spelling)")
    city: str = Field(..., description="City")
    first_inning_runs: float = Field(..., ge=0)
    first_inning_wkts: float = Field(..., ge=0, le=10)


class PredictResponse(BaseModel):
    team_a: str
    team_b: str
    city: str
    team_a_win_prob: float
    team_b_win_prob: float
    predicted_winner: str


# -----------------------------
# Core prediction function
# -----------------------------
def _predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None or PACK is None or DEVICE is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Error: {LOAD_ERROR}")

    team_le = PACK["team_le"]
    city_le = PACK["city_le"]
    scaler = PACK["scaler"]

    team_a = _clean_team(req.team_a)
    team_b = _clean_team(req.team_b)
    city = _clean_city(req.city)

    # Encode
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

    # Tensors
    team_a_t = torch.tensor([team_a_id], dtype=torch.long, device=DEVICE)
    team_b_t = torch.tensor([team_b_id], dtype=torch.long, device=DEVICE)
    city_t = torch.tensor([city_id], dtype=torch.long, device=DEVICE)
    num_t = torch.tensor(num_scaled, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        logit = MODEL(team_a_t, team_b_t, city_t, num_t)
        prob_a = float(torch.sigmoid(logit).item())

    predicted = team_a if prob_a >= 0.5 else team_b

    return PredictResponse(
        team_a=team_a,
        team_b=team_b,
        city=city,
        team_a_win_prob=prob_a,
        team_b_win_prob=float(1.0 - prob_a),
        predicted_winner=predicted,
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if LOAD_ERROR is None else "error",
        "model_loaded": LOAD_ERROR is None,
        "error": LOAD_ERROR,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "model_path_exists": MODEL_PATH.exists(),
        "prep_path_exists": PREP_PATH.exists(),
    }


@app.get("/metadata")
def metadata():
    if PACK is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded. Error: {LOAD_ERROR}")
    return {
        "n_teams": int(PACK["n_teams"]),
        "n_cities": int(PACK["n_cities"]),
        "teams": PACK["team_le"].classes_.tolist(),
        "cities": PACK["city_le"].classes_.tolist(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _predict(req)


# -----------------------------
# Minimal web page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>T20 Win Predictor - Test</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; }
    label { display:block; margin-top: 10px; font-weight: 600; }
    input { width: 100%; padding: 10px; margin-top: 6px; }
    button { margin-top: 14px; padding: 10px 14px; cursor: pointer; }
    .small { color:#555; font-size: 0.95em; }
  </style>
</head>
<body>
  <h1>T20 Win Predictor (API Test Page)</h1>
  <p class="small">
    Use this page to test your deployed model quickly.
    For valid teams/cities, open <a href="/metadata" target="_blank">/metadata</a>.
    Health: <a href="/health" target="_blank">/health</a>.
  </p>

  <div class="card">
    <form method="post" action="/predict_form">
      <label>Team A</label>
      <input name="team_a" placeholder="India" required />

      <label>Team B</label>
      <input name="team_b" placeholder="Pakistan" required />

      <label>City</label>
      <input name="city" placeholder="Bangalore" required />

      <label>First innings runs</label>
      <input name="first_inning_runs" type="number" step="1" min="0" placeholder="160" required />

      <label>First innings wickets</label>
      <input name="first_inning_wkts" type="number" step="1" min="0" max="10" placeholder="6" required />

      <button type="submit">Predict</button>
    </form>
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    team_a: str = Form(...),
    team_b: str = Form(...),
    city: str = Form(...),
    first_inning_runs: float = Form(...),
    first_inning_wkts: float = Form(...),
):
    req = PredictRequest(
        team_a=team_a,
        team_b=team_b,
        city=city,
        first_inning_runs=first_inning_runs,
        first_inning_wkts=first_inning_wkts,
    )

    try:
        result = _predict(req)
        body = f"""
        <h2>Prediction Result</h2>
        <pre>{result.model_dump_json(indent=2)}</pre>
        <p><a href="/">Back</a></p>
        """
        return HTMLResponse(body)
    except HTTPException as e:
        body = f"""
        <h2>Error</h2>
        <pre>Status: {e.status_code}\nDetail: {e.detail}</pre>
        <p><a href="/">Back</a></p>
        """
        return HTMLResponse(body, status_code=e.status_code)
