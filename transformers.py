import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class BiomedicalAttrAdder(BaseEstimator, TransformerMixin):
    """Engineer domain-specific ICU physiological and electrophysiological indices."""
    def __init__(self, add_stress_ratio=True):
        self.add_stress_ratio = add_stress_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Work on a copy so the caller's input frame remains unchanged.
        
        X = X.copy()
        # 1. Modified Shock Index (Hemodynamic instability)
        X["shock_index"] = X["d1_heartrate_max"] / (X["d1_sysbp_min"] + 1e-5)
        
        # 2. Pulse Pressure (Stroke volume & vascular compliance)
        X["pulse_pressure"] = X["d1_sysbp_min"] - X["d1_diasbp_min"]
        
        # 3. Combined Glasgow Coma Scale (Neurological responsiveness)
        X["gcs_eye_motor"] = X["gcs_eyes_apache"] + X["gcs_motor_apache"]
        
        # 4. Glycemic Variability (Metabolic stress swing)
        X["glucose_swing"] = X["d1_glucose_max"] - X["d1_glucose_min"]
        
        # 5. Potassium Swing (Electrolyte / myocardial excitability shift)
        X["potassium_swing"] = X["d1_potassium_max"] - X["d1_potassium_min"]
        
        if self.add_stress_ratio:
            X["cardio_resp_stress"] = (
                (X["d1_heartrate_max"] * X["resprate_apache"]) / (X["d1_spo2_min"] + 1.0)
            )
        return X

    
    def get_feature_names_out(self, input_features=None):
        """Return original and engineered feature names."""

        if input_features is None:
            input_features = self.feature_names_in_

        feature_names = list(input_features)
        feature_names.append("shock_index")
        feature_names.append("pulse_pressure")
        feature_names.append("gcs_eye_motor")
        feature_names.append("glucose_swing")
        feature_names.append("potassium_swing")

        if self.add_stress_ratio:
            feature_names.append("cardio_resp_stress")

        return np.asarray(feature_names, dtype=object)
