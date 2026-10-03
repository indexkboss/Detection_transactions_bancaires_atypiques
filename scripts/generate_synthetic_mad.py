"""Génère des transactions SYNTHÉTIQUES en dirhams (MAD) avec anomalies injectées.

Usage (depuis la racine du projet) :
    python scripts/generate_synthetic_mad.py
    python scripts/generate_synthetic_mad.py --n 10000 --rate 0.03 --seed 7 --evaluate

Fichiers produits dans data/synthetic/ :
    synthetic_mad.csv         -> à importer dans le dashboard (aucune étiquette)
    synthetic_mad_labels.csv  -> vérité terrain, réservée à l'évaluation
                                 (ne jamais la donner à l'analyste)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "synthetic"

# Profil horaire d'un site marchand : peu d'activité la nuit, pic en journée/soirée
HOUR_WEIGHTS = np.array([
    1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9,
    9, 9, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2,
], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()

# Catégories alignées sur IEEE-CIS (W, C, H, R, S) : le filtre du dashboard les reconnaît
CATEGORIES = ["W", "C", "H", "R", "S"]
CATEGORY_P = [0.55, 0.15, 0.12, 0.10, 0.08]
AMOUNT_SCALE = {"W": 1.0, "C": 1.4, "H": 0.8, "R": 1.1, "S": 0.6}   # panier moyen relatif

ANOMALY_TYPES = ["montant_extreme", "nuit_montant_eleve", "profil_inhabituel"]


def generate(n=5000, anomaly_rate=0.02, seed=42, start="2025-01-01", days=60):
    rng = np.random.default_rng(seed)

    # --- transactions normales -------------------------------------------------
    day = rng.integers(0, days, n)
    hour = rng.choice(24, size=n, p=HOUR_WEIGHTS)
    seconds = rng.integers(0, 3600, n)
    dates = (pd.Timestamp(start) + pd.to_timedelta(day, unit="D")
             + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(seconds, unit="s"))

    product = rng.choice(CATEGORIES, size=n, p=CATEGORY_P)
    scale = pd.Series(product).map(AMOUNT_SCALE).to_numpy()
    amount = rng.lognormal(mean=np.log(350), sigma=0.7, size=n) * scale

    df = pd.DataFrame({
        "TransactionID": np.arange(9_000_001, 9_000_001 + n),
        "TransactionDate": dates,
        "TransactionAmt": amount,
        "ProductCD": product,
        "card4": rng.choice(["visa", "mastercard", "american express"], size=n, p=[0.62, 0.35, 0.03]),
        "card6": rng.choice(["debit", "credit"], size=n, p=[0.70, 0.30]),
        "P_emaildomain": rng.choice(
            ["gmail.com", "hotmail.com", "yahoo.fr", "outlook.com", "menara.ma"],
            size=n, p=[0.55, 0.15, 0.12, 0.10, 0.08]),
        "DeviceType": rng.choice(["mobile", "desktop", None], size=n, p=[0.60, 0.25, 0.15]),
    })

    # --- anomalies injectées ---------------------------------------------------
    n_anom = max(1, int(round(n * anomaly_rate)))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    kinds = rng.choice(ANOMALY_TYPES, size=n_anom)
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""

    for i, kind in zip(anom_idx, kinds):
        if kind == "montant_extreme":            # montant 8 à 25 fois trop élevé
            df.loc[i, "TransactionAmt"] *= rng.uniform(8, 25)
        elif kind == "nuit_montant_eleve":       # achat conséquent entre 0h et 4h
            d = df.loc[i, "TransactionDate"].normalize()
            df.loc[i, "TransactionDate"] = (d + pd.Timedelta(hours=int(rng.integers(0, 5)),
                                                             minutes=int(rng.integers(0, 60))))
            df.loc[i, "TransactionAmt"] *= rng.uniform(3, 6)
        else:                                    # profil rare : domaine email, carte, appareil
            df.loc[i, "P_emaildomain"] = rng.choice(["protonmail.com", "tutanota.com"])
            df.loc[i, "card4"] = "discover"
            df.loc[i, "DeviceType"] = "desktop"
            df.loc[i, "TransactionAmt"] *= rng.uniform(2, 4)
        df.loc[i, "is_anomaly"] = 1
        df.loc[i, "anomaly_type"] = kind

    df["TransactionAmt"] = df["TransactionAmt"].clip(lower=5).round(2)
    df["Currency"] = "MAD"
    return df.sort_values("TransactionDate").reset_index(drop=True)


def evaluate(df):
    """Mode NON SUPERVISÉ : l'Isolation Forest est entraîné sans étiquettes,
    puis comparé à la vérité terrain (précision, rappel, PR-AUC, faux positifs)."""
    sys.path.insert(0, str(BASE_DIR))
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.scoring import fit_unsupervised, score_transactions

    raw = df.drop(columns=["is_anomaly", "anomaly_type"])
    scored = score_transactions(raw, fit_unsupervised(raw))
    y = df["is_anomaly"].to_numpy()
    s = scored["RiskScore"].to_numpy()

    print("\n=== Évaluation non supervisée sur données synthétiques ===")
    print(f"Transactions : {len(df)} | anomalies injectées : {y.sum()} ({y.mean():.1%})")
    print(f"PR-AUC  : {average_precision_score(y, s):.3f}")
    print(f"ROC-AUC : {roc_auc_score(y, s):.3f}")

    high = (scored["RiskLevel"] == "High").to_numpy()
    tp, fp = int(((y == 1) & high).sum()), int(((y == 0) & high).sum())
    print(f"Niveau High : {high.sum()} alertes | précision={tp / max(high.sum(), 1):.3f} "
          f"| rappel={tp / y.sum():.3f} | taux de faux positifs={fp / (y == 0).sum():.4f}")

    top_k = np.argsort(-s)[: int(y.sum())]
    print(f"Précision@{int(y.sum())} (autant d'alertes que d'anomalies) : {y[top_k].mean():.3f}")
    for kind in ANOMALY_TYPES:
        m = (df["anomaly_type"] == kind).to_numpy()
        print(f"  rappel dans le niveau High - {kind:<20}: {(high & m).sum() / max(m.sum(), 1):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5000, help="nombre de transactions")
    p.add_argument("--rate", type=float, default=0.02, help="proportion d'anomalies injectées")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate", action="store_true", help="évalue le mode non supervisé")
    args = p.parse_args()

    df = generate(args.n, args.rate, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["is_anomaly", "anomaly_type"]).to_csv(OUT_DIR / "synthetic_mad.csv", index=False)
    df[["TransactionID", "is_anomaly", "anomaly_type"]].to_csv(OUT_DIR / "synthetic_mad_labels.csv", index=False)
    print(f"{len(df)} transactions écrites dans {OUT_DIR}")
    print(f"Montant médian : {df['TransactionAmt'].median():.2f} MAD | max : {df['TransactionAmt'].max():.2f} MAD")

    if args.evaluate:
        evaluate(df)


if __name__ == "__main__":
    main()