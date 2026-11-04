"""
Generate a reproducible, physically motivated landslide classification dataset.

This is simulated data, not a substitute for field observations. The target is
generated from a nonlinear susceptibility score with modest label noise. This
makes the classification pattern learnable while retaining realistic overlap.

With the matching Streamlit app's Random Forest, the default seed should
typically produce >90% held-out accuracy. Actual performance can vary with
model settings and train/test split. Report results as simulated-data results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 2026
SAMPLE_COUNT = 2000


def generate_dataset(sample_count=SAMPLE_COUNT, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    # Environmental predictors
    rainfall = np.clip(
        rng.gamma(shape=5.0, scale=35.0, size=sample_count) + 35, 40, 450
    )
    slope = np.clip(
        rng.normal(27, 13, sample_count)
        + 0.025 * (rainfall - 210),
        3, 70,
    )
    soil_saturation = np.clip(
        0.22 + 0.00115 * rainfall + rng.normal(0, 0.12, sample_count),
        0.01, 1.0,
    )
    vegetation_cover = np.clip(
        0.82 - 0.005 * slope - 0.0007 * rainfall
        + rng.normal(0, 0.13, sample_count),
        0.05, 1.0,
    )
    earthquake_activity = np.clip(
        rng.lognormal(mean=0.65, sigma=0.65, size=sample_count), 0.01, 12
    )
    proximity_to_water = np.clip(
        rng.gamma(shape=2.2, scale=0.75, size=sample_count), 0.02, 6
    )

    soil_type = rng.choice(
        ["Gravel", "Sand", "Silt"],
        size=sample_count,
        p=[0.30, 0.40, 0.30],
    )
    soil_effect = np.select(
        [soil_type == "Sand", soil_type == "Silt"],
        [0.28, 0.12],
        default=-0.08,
    )

    # Standardized contributions to a nonlinear susceptibility score
    rain_score = (rainfall - 170) / 75
    slope_score = (slope - 28) / 14
    saturation_score = (soil_saturation - 0.55) / 0.22
    vegetation_score = (vegetation_cover - 0.50) / 0.22
    earthquake_score = (earthquake_activity - 2.5) / 2.0
    water_score = (proximity_to_water - 1.4) / 1.1

    risk_score = (
        0.75 * rain_score
        + 0.80 * slope_score
        + 0.85 * saturation_score
        - 0.62 * vegetation_score
        + 0.28 * earthquake_score
        + 0.22 * water_score
        + 0.48 * rain_score * saturation_score
        + 0.38 * slope_score * saturation_score
        + soil_effect
    )

    # A thresholded noisy score creates a learnable binary target.
    # Noise is retained to prevent a perfectly deterministic toy dataset.
    threshold = np.median(risk_score)
    observed_score = risk_score + rng.normal(0, 0.22, sample_count)
    landslide = (observed_score > threshold).astype(int)

    data = pd.DataFrame(
        {
            "Rainfall_mm": rainfall,
            "Slope_Angle": slope,
            "Soil_Saturation": soil_saturation,
            "Vegetation_Cover": vegetation_cover,
            "Earthquake_Activity": earthquake_activity,
            "Proximity_to_Water": proximity_to_water,
            "Landslide": landslide,
            "Soil_Type_Gravel": (soil_type == "Gravel").astype(int),
            "Soil_Type_Sand": (soil_type == "Sand").astype(int),
            "Soil_Type_Silt": (soil_type == "Silt").astype(int),
        }
    )

    # Introduce 1% missing values in each feature column, not in the target.
    feature_columns = [column for column in data.columns if column != "Landslide"]
    missing_count = int(round(sample_count * 0.01))
    for column in feature_columns:
        missing_indices = rng.choice(sample_count, size=missing_count, replace=False)
        data.loc[missing_indices, column] = np.nan

    return data


if __name__ == "__main__":
    output_path = Path(__file__).with_name("generated_landslide_dataset.csv")
    dataset = generate_dataset()
    dataset.to_csv(output_path, index=False)

    print(f"Wrote {len(dataset):,} rows to {output_path}")
    print("\nTarget distribution:")
    print(dataset["Landslide"].value_counts(normalize=True).sort_index().round(3))
