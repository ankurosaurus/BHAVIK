"""
BHAVIK REST API Endpoints
Provides system diagnostics, model metadata, dataset collection, and settings.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.config import EXCLUDED_CLASSES, SUPPORTED_CLASSES
from backend.src.data_collector import DataCollector
from backend.src.predictor import ASLPredictor

router = APIRouter(prefix="/api", tags=["API"])

# Shared singletons (injected or initialized)
collector = DataCollector()
predictor = ASLPredictor()


class LandmarkItem(BaseModel):
    x: float
    y: float
    z: float = 0.0


class CollectSampleRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=1)
    landmarks: List[LandmarkItem] = Field(..., min_length=21, max_length=21)
    session_id: str = "web_client"
    is_left_hand: bool = False


class SettingsUpdateRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(None, ge=0.1, le=1.0)
    buffer_size: Optional[int] = Field(None, ge=3, le=30)
    stability_ratio: Optional[float] = Field(None, ge=0.5, le=1.0)
    cooldown_seconds: Optional[float] = Field(None, ge=0.2, le=5.0)


@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Returns server and ML engine health status."""
    return {
        "status": "online",
        "model_loaded": predictor.is_loaded,
        "supported_classes_count": len(SUPPORTED_CLASSES),
        "total_dataset_samples": collector.get_total_sample_count(),
    }


@router.get("/metadata")
def get_metadata() -> Dict[str, Any]:
    """Returns model metadata, evaluation metrics, and class definitions."""
    meta = predictor.get_metadata()
    return {
        "model_loaded": predictor.is_loaded,
        "supported_classes": SUPPORTED_CLASSES,
        "excluded_classes": EXCLUDED_CLASSES,
        "metrics": meta.get("metrics", {}),
        "trained_at": meta.get("trained_at"),
        "model_type": meta.get("model_type", "RandomForestClassifier"),
    }


@router.get("/dataset/counts")
def get_dataset_counts() -> Dict[str, Any]:
    """Returns sample counts per letter in the dataset."""
    return {
        "total_samples": collector.get_total_sample_count(),
        "per_class": collector.get_sample_counts(),
    }


@router.post("/dataset/collect")
def collect_sample(payload: CollectSampleRequest) -> Dict[str, Any]:
    """Appends a new user-contributed sample from the web interface."""
    dict_landmarks = [lm.model_dump() for lm in payload.landmarks]
    success = collector.add_sample(
        label=payload.label,
        landmarks=dict_landmarks,
        session_id=payload.session_id,
        is_left_hand=payload.is_left_hand,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Invalid sample or unsupported label.")

    return {
        "success": True,
        "label": payload.label.upper(),
        "new_count": collector.get_sample_counts().get(payload.label.upper(), 0),
        "total_samples": collector.get_total_sample_count(),
    }
