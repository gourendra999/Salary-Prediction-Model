"""Shared, deterministic feature engineering for salary model training and inference."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Experience (Years)", "Education Degree", "Specialization", "Education Level",
    "Job Sector", "Job Role", "Company Size", "City Tier",
]
NUMERIC_FEATURES = ["Experience (Years)", "Experience Squared"]
CATEGORICAL_FEATURES = [
    "Education Degree", "Specialization", "Education Level", "Job Sector",
    "Job Role", "Company Size", "City Tier", "Career Stage", "Education-Sector",
    "Sector-Role", "Company-Market",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _clean_category(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unknown"
    # Keep this identical in training and inference so category labels never drift.
    return str(value).strip().title()


def city_to_tier(city) -> str:
    """Map common Indian city-name variants to the app's tier labels."""
    if pd.isna(city) or not str(city).strip():
        return "Tier 3 (Smaller Cities)"
    key = re.sub(r"[^a-z0-9 ]", " ", str(city).casefold())
    key = re.sub(r"\s+", " ", key).strip()
    tier1 = {"bengaluru", "bangalore", "mumbai", "delhi", "delhi ncr", "new delhi", "hyderabad", "chennai", "pune"}
    tier2 = {"noida", "gurgaon", "gurugram", "ahmedabad", "kolkata"}
    if key in tier1:
        return "Tier 1 (Metro)"
    if key in tier2:
        return "Tier 2 (Mid-Sized)"
    return "Tier 3 (Smaller Cities)"


def normalize_raw_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return clean raw model features, preserving expected input column names."""
    df = frame.copy()
    for col in RAW_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown" if col != "Experience (Years)" else 0.0
    df["Experience (Years)"] = pd.to_numeric(df["Experience (Years)"], errors="coerce").fillna(0.0).clip(lower=0)
    categorical = [c for c in RAW_FEATURES if c != "Experience (Years)"]
    for col in categorical:
        df[col] = df[col].map(_clean_category)
    return df[RAW_FEATURES]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add non-target-derived numerical, career-stage and interaction features."""
    df = normalize_raw_features(frame)
    exp = df["Experience (Years)"].astype(float)
    df["Experience Squared"] = np.square(exp).astype(float)
    stage = pd.cut(
        exp,
        bins=[-np.inf, 1, 3, 7, 12, np.inf],
        labels=["Early Career", "Developing", "Mid-Career", "Senior", "Expert/Leadership"],
        right=True,
        include_lowest=True,
    )
    df["Career Stage"] = stage.astype(object).where(stage.notna(), "Unknown")
    df["Education-Sector"] = df["Education Level"].astype(str) + " | " + df["Job Sector"].astype(str)
    df["Sector-Role"] = df["Job Sector"].astype(str) + " | " + df["Job Role"].astype(str)
    df["Company-Market"] = df["Company Size"].astype(str) + " | " + df["City Tier"].astype(str)
    return df[MODEL_FEATURES]
