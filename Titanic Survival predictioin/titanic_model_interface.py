"""
TitanicModelInterface: lightweight wrapper to load a model and predict survival.

Usage example:
    from titanic_model_interface import TitanicModelInterface
    iface = TitanicModelInterface()
    iface.load_model(r"C:\path\to\model.pkl")
    res = iface.predict({
        "Pclass":3, "Sex":"male", "Age":22, "SibSp":1, "Parch":0, "Fare":7.25, "Embarked":"S"
    })
    print(res)

The interface accepts a scikit-learn-like estimator (pickle/joblib), Keras .h5 models, or a custom preprocess function.
"""

import os
import pickle
from typing import Callable, Optional, Union, List, Dict, Any

import joblib
import pandas as pd
import numpy as np

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None


class TitanicModelInterface:
    """
    Lightweight interface around a Titanic survival model.
    - model: any estimator with predict or predict_proba, or a Keras model
    - preprocess_fn: optional callable to transform input dict(s) into a DataFrame matching the model's features
    - impute_values: dict of default median values used by the default preprocessor
    """

    def __init__(self, model: Any = None, preprocess_fn: Optional[Callable] = None, impute_values: Optional[Dict[str, float]] = None):
        self.model = model
        self.preprocess_fn = preprocess_fn or self.default_preprocess
        self.impute_values = impute_values or {"Age": 28.0, "Fare": 14.45}
        self.feature_order = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    def load_model(self, path: str, model_type: str = "auto"):
        """Load a model from disk. Supported types: 'sklearn' (pickle/joblib), 'keras', or 'auto'."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        mt = model_type.lower()
        if mt == "auto":
            if path.endswith(".h5") or path.endswith(".keras"):
                mt = "keras"
            else:
                mt = "sklearn"
        if mt == "sklearn":
            try:
                self.model = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
        elif mt == "keras":
            if keras_load_model is None:
                raise RuntimeError("Keras not available in this environment.")
            self.model = keras_load_model(path)
        else:
            raise ValueError("Unsupported model_type: " + model_type)
        return self.model

    def default_preprocess(self, X: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """Basic preprocessing that mirrors common Titanic pipelines:
        - maps Sex -> {male:0, female:1}
        - maps Embarked -> {S:0, C:1, Q:2}
        - fills missing Age/Fare with medians (from impute_values)
        - ensures columns present and in feature_order
        If your trained model used a different pipeline, pass a custom preprocess_fn.
        """
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("Input must be dict, list of dicts, or DataFrame")

        for c in self.feature_order:
            if c not in df.columns:
                df[c] = np.nan

        def map_sex(v):
            if pd.isnull(v):
                return np.nan
            if isinstance(v, str):
                v2 = v.strip().lower()
                if v2 in ("male", "m", "0"):
                    return 0
                if v2 in ("female", "f", "1"):
                    return 1
            try:
                return int(v)
            except Exception:
                return np.nan

        df["Sex"] = df["Sex"].apply(map_sex)

        emb_map = {"S": 0, "C": 1, "Q": 2}

        def map_emb(v):
            if pd.isnull(v):
                return np.nan
            if isinstance(v, str):
                return emb_map.get(v.strip().upper(), np.nan)
            try:
                return int(v)
            except Exception:
                return np.nan

        df["Embarked"] = df["Embarked"].apply(map_emb)

        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(self.impute_values.get("Age", 28.0)).astype(float)
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce").fillna(self.impute_values.get("Fare", 14.45)).astype(float)
        df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce").fillna(3).astype(int)
        df["SibSp"] = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0).astype(int)
        df["Parch"] = pd.to_numeric(df["Parch"], errors="coerce").fillna(0).astype(int)

        return df[self.feature_order]

    def predict(self, X_input: Union[Dict, List[Dict], pd.DataFrame], threshold: float = 0.5) -> Union[Dict, List[Dict]]:
        """Predict survival for one or more inputs.
        Returns a dict (single) or list of dicts (batch) with keys:
        - class: 0 or 1
        - probability: float (if available, else None)
        - raw: raw model output
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Pass model to constructor or call load_model().")

        X_df = self.preprocess_fn(X_input)
        single = False
        if isinstance(X_input, dict):
            single = True

        probs = None
        preds = None
        model = self.model

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_df)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    probs = probs[:, 1]
            elif hasattr(model, "predict") and not hasattr(model, "classes_"):
                raw = model.predict(X_df)
                raw = np.asarray(raw).reshape(-1)
                if np.all((raw >= 0) & (raw <= 1)):
                    probs = raw
                else:
                    preds = raw.astype(int)
            else:
                preds = model.predict(X_df)
                preds = np.asarray(preds).reshape(-1)
        except Exception:
            try:
                preds = np.asarray(model.predict(X_df)).reshape(-1)
            except Exception as e:
                raise RuntimeError(f"Model prediction failed: {e}")

        results = []
        n = X_df.shape[0]
        for i in range(n):
            p = float(probs[i]) if (probs is not None) else None
            c = int((p >= threshold)) if p is not None else (int(preds[i]) if preds is not None else None)
            results.append({"class": c, "probability": p, "raw": (float(p) if p is not None else (int(preds[i]) if preds is not None else None))})

        if single:
            return results[0]
        return results

    def predict_batch(self, X_list: List[Dict], threshold: float = 0.5) -> List[Dict]:
        return self.predict(X_list, threshold=threshold)

    def set_preprocessor(self, preprocess_fn: Callable):
        self.preprocess_fn = preprocess_fn


__all__ = ["TitanicModelInterface"]
