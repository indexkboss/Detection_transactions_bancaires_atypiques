"""Scoring des transactions : features, modèle non supervisé, risque, explications.

Utilisé par le notebook (export) ET par app.py (import d'un fichier brut).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

REFERENCE_DATE = pd.Timestamp("2017-12-01")

NUMERIC = [
    "TransactionAmt", "LogTransactionAmt", "addr1", "addr2", "dist1", "dist2",
    "RelativeHour", "HourSin", "HourCos", "HasIdentity",
]
CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
]
FEATURES = NUMERIC + CATEGORICAL

RAW_COLUMNS = [
    "TransactionAmt", "TransactionDT", "ProductCD", "card4", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    "DeviceType",
]

DESCRIPTIONS = {
    "TransactionAmt": "Montant de la transaction",
    "ProductCD": "Catégorie de transaction",
    "card4": "Réseau de carte bancaire",
    "card6": "Type de carte bancaire",
    "addr1": "Adresse principale",
    "addr2": "Adresse secondaire",
    "dist1": "Distance associée",
    "dist2": "Distance secondaire",
    "P_emaildomain": "Domaine email acheteur",
    "R_emaildomain": "Domaine email destinataire",
    "RelativeHour": "Heure de la transaction",
    "HasIdentity": "Informations d'identité disponibles",
    "DeviceType": "Type d'appareil",
}
# Variables dérivées/redondantes : exclues des explications affichées
EXPLAIN_FEATURES = list(DESCRIPTIONS)

ARTIFACT_FILES = {
    "preprocessor": "preprocessor.joblib",
    "xgb_model": "xgb_model.joblib",
    "isolation_forest": "isolation_forest.joblib",
    "anomaly_scaler": "anomaly_scaler.joblib",
}


# ---------------------------------------------------------------- features
def build_features(df):
    """Reconstruit les variables du notebook à partir d'un fichier de transactions."""
    out = df.copy()
    if "TransactionDT" not in out.columns and "TransactionDate" in out.columns:
        out["TransactionDT"] = (
            pd.to_datetime(out["TransactionDate"], errors="coerce") - REFERENCE_DATE
        ).dt.total_seconds()
    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    for col in ["TransactionAmt", "TransactionDT", "addr1", "addr2", "dist1", "dist2"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["LogTransactionAmt"] = np.log1p(out["TransactionAmt"].clip(lower=0))
    out["ReconstructedDate"] = REFERENCE_DATE + pd.to_timedelta(out["TransactionDT"], unit="s")
    out["RelativeHour"] = (out["TransactionDT"] % 86400) // 3600
    out["HourSin"] = np.sin(2 * np.pi * out["RelativeHour"] / 24)
    out["HourCos"] = np.cos(2 * np.pi * out["RelativeHour"] / 24)
    if "HasIdentity" not in out.columns:  # approximation : DeviceType renseigné
        out["HasIdentity"] = out["DeviceType"].notna().astype(int)
    for col in CATEGORICAL:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    return out


# ------------------------------------------------- mode SANS étiquettes
def build_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), CATEGORICAL),
    ])


def fit_unsupervised(df_raw, n_estimators=200, seed=42):
    """Entraîne preprocessing + Isolation Forest sans aucune étiquette isFraud."""
    X = build_features(df_raw)[FEATURES]
    pre = build_preprocessor()
    Xp = pre.fit_transform(X)
    iso = IsolationForest(
        n_estimators=n_estimators, max_samples=min(Xp.shape[0], 2048),
        random_state=seed, n_jobs=-1,
    ).fit(Xp)
    scaler = MinMaxScaler().fit((-iso.decision_function(Xp)).reshape(-1, 1))
    return {"preprocessor": pre, "isolation_forest": iso, "anomaly_scaler": scaler}


# ------------------------------------------------------------ persistance
def save_artifacts(artifacts, models_dir):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in ARTIFACT_FILES.items():
        if artifacts.get(key) is not None:
            joblib.dump(artifacts[key], models_dir / filename)


def load_artifacts(models_dir):
    """Retourne None si les modèles minimaux (preprocessor + IF + scaler) sont absents."""
    models_dir = Path(models_dir)
    artifacts = {}
    for key, filename in ARTIFACT_FILES.items():
        path = models_dir / filename
        if path.exists():
            artifacts[key] = joblib.load(path)
    needed = {"preprocessor", "isolation_forest", "anomaly_scaler"}
    return artifacts if needed.issubset(artifacts) else None


# ---------------------------------------------------------------- scoring
def risk_level(score):
    if score < 30:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def score_transactions(df_raw, artifacts, w_xgb=0.7):
    feats = build_features(df_raw)
    Xp = artifacts["preprocessor"].transform(feats[FEATURES])

    raw = -artifacts["isolation_forest"].decision_function(Xp)
    anomaly = np.clip(
        artifacts["anomaly_scaler"].transform(raw.reshape(-1, 1)).ravel(), 0, 1
    )

    keep = [c for c in ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD"]
            if c in df_raw.columns]
    res = df_raw[keep].copy()
    res["ReconstructedDate"] = feats["ReconstructedDate"]
    res["AnomalyScore"] = -raw                 # convention sklearn : bas = anormal
    res["AnomalyNormalized"] = anomaly

    xgb = artifacts.get("xgb_model")
    if xgb is not None:
        proba = xgb.predict_proba(Xp)[:, 1]
        res["FraudProbability"] = proba
        res["RiskScore"] = (w_xgb * proba + (1 - w_xgb) * anomaly) * 100
        res["RiskLevel"] = res["RiskScore"].apply(risk_level)
        res["ScoringMode"] = "Hybride (XGBoost + Isolation Forest)"
    else:
        # Sans modèle supervisé, les seuils 30/70 n'ont pas de sens : niveaux par rang
        res["RiskScore"] = anomaly * 100
        pct = res["RiskScore"].rank(pct=True)
        res["RiskLevel"] = np.select([pct > 0.95, pct > 0.80], ["High", "Medium"], "Low")
        res["ScoringMode"] = "Non supervisé (Isolation Forest)"
    return res


# ----------------------------------------------------------- explications
def _origin(feature_name):
    name = feature_name.split("__", 1)[-1]
    for col in sorted(CATEGORICAL, key=len, reverse=True):
        if name.startswith(col + "_"):
            return col
    return name


def _fmt(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "manquant" if pd.isna(value) else f"{value:g}"
    return "manquant" if pd.isna(value) else str(value)


def explain(df_raw, artifacts, top_k=3, reference=None):
    """Deux explications par ligne, séparées par ' | ' :
    - TopRiskFactors : contributions SHAP (XGBoost, log-odds) regroupées par variable
    - AnomalyReasons : variables les plus atypiques par rapport au fichier scoré
    À appeler sur un sous-ensemble (ex. top 1000), pas sur 500 000 lignes.
    """
    feats = build_features(df_raw)
    n = len(feats)
    out = pd.DataFrame(index=df_raw.index, columns=["TopRiskFactors", "AnomalyReasons"], data="")

    # --- SHAP groupé par variable d'origine
    xgb = artifacts.get("xgb_model")
    if xgb is not None:
        import shap
        Xp = artifacts["preprocessor"].transform(feats[FEATURES])
        sv = np.asarray(shap.TreeExplainer(xgb).shap_values(Xp))
        origin = np.array([_origin(f) for f in artifacts["preprocessor"].get_feature_names_out()])
        cols = [c for c in EXPLAIN_FEATURES if (origin == c).any()]
        G = np.column_stack([sv[:, origin == c].sum(axis=1) for c in cols])
        order = np.argsort(-G, axis=1)[:, :top_k]
        texts = []
        for i in range(n):
            parts = []
            for j in order[i]:
                if G[i, j] > 0:
                    c = cols[j]
                    parts.append(f"{DESCRIPTIONS[c]} = {_fmt(feats[c].iloc[i])} (+{G[i, j]:.2f})")
            texts.append(" | ".join(parts))
        out["TopRiskFactors"] = texts

    # --- Atypicité par rapport au fichier
    ref = feats if reference is None else build_features(reference)
    U, meta = [], []
    for c in EXPLAIN_FEATURES:
        s, r = feats[c], ref[c]
        if c in CATEGORICAL:
            freq = s.map(r.value_counts(normalize=True))
            u = (1 - freq).fillna(0).to_numpy()
            meta.append(("cat", c, freq.to_numpy(), None))
        else:
            med = r.median()
            iqr = r.quantile(0.75) - r.quantile(0.25)
            diff = (s - med).abs()
            u = (np.minimum(diff / (iqr + 1e-8), 3) / 3 if iqr > 0 else (diff > 0).astype(float))
            u = u.fillna(0).to_numpy()
            meta.append(("num", c, None, med))
        U.append(u)
    U = np.column_stack(U)
    order = np.argsort(-U, axis=1)[:, :top_k]
    texts = []
    for i in range(n):
        parts = []
        for j in order[i]:
            if U[i, j] <= 0:
                continue
            kind, c, freq, med = meta[j]
            val = _fmt(feats[c].iloc[i])
            if kind == "cat":
                parts.append(f"{DESCRIPTIONS[c]} = {val} (présent dans {freq[i]:.1%} des transactions)")
            else:
                parts.append(f"{DESCRIPTIONS[c]} = {val} (médiane du fichier : {med:g})")
        texts.append(" | ".join(parts))
    out["AnomalyReasons"] = texts
    return out