"""Historique des décisions analyste (SQLite, journal en ajout seul).

Chaque décision est une NOUVELLE ligne : rien n'est écrasé, ce qui garde la trace
des changements d'avis sur une même transaction (audit).
"""
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Noms de colonnes identiques à l'ancien decision_history.csv (+ Analyst, Source)
COLUMNS = ["TransactionID", "AnalystDecision", "AnalystComment", "ReviewDate",
           "PreviousRiskLevel", "RiskScore", "Analyst", "Source"]

_DB_TO_APP = {
    "transaction_id": "TransactionID", "analyst_decision": "AnalystDecision",
    "analyst_comment": "AnalystComment", "review_date": "ReviewDate",
    "previous_risk_level": "PreviousRiskLevel", "risk_score": "RiskScore",
    "analyst": "Analyst", "source": "Source",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id      TEXT NOT NULL,
    analyst_decision    TEXT NOT NULL,
    analyst_comment     TEXT,
    review_date         TEXT NOT NULL,
    previous_risk_level TEXT,
    risk_score          REAL,
    analyst             TEXT,
    source              TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_tx ON decisions (transaction_id);
"""


def _connect(db_path):
    return closing(sqlite3.connect(str(db_path)))


def init_db(db_path, legacy_csv=None):
    """Crée la base si besoin. Si elle est vide et qu'un ancien CSV existe, l'importe une fois."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn, conn:
        conn.executescript(_SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0

    if empty and legacy_csv is not None and Path(legacy_csv).exists():
        try:
            old = pd.read_csv(legacy_csv)
        except (pd.errors.EmptyDataError, OSError):
            return
        for _, r in old.iterrows():
            add_decision(
                db_path,
                transaction_id=r.get("TransactionID"),
                decision=r.get("AnalystDecision"),
                comment=r.get("AnalystComment"),
                previous_risk_level=r.get("PreviousRiskLevel"),
                risk_score=r.get("RiskScore"),
                review_date=r.get("ReviewDate"),
                source="import_csv",
            )


def _clean_text(value):
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def add_decision(db_path, transaction_id, decision, comment="", previous_risk_level="",
                 risk_score=None, analyst="", source="", review_date=None):
    """Ajoute une décision. Les types numpy sont convertis (SQLite ne les accepte pas)."""
    score = None if risk_score is None or pd.isna(risk_score) else float(risk_score)
    review_date = review_date or datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn, conn:
        cur = conn.execute(
            "INSERT INTO decisions (transaction_id, analyst_decision, analyst_comment, review_date,"
            " previous_risk_level, risk_score, analyst, source) VALUES (?,?,?,?,?,?,?,?)",
            (_clean_text(transaction_id), _clean_text(decision), _clean_text(comment),
             _clean_text(review_date), _clean_text(previous_risk_level), score,
             _clean_text(analyst), _clean_text(source)),
        )
        return cur.lastrowid


def load_history(db_path):
    """Toutes les décisions, de la plus récente à la plus ancienne."""
    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", conn)
    return df.drop(columns=["id"]).rename(columns=_DB_TO_APP)[COLUMNS]


def get_transaction_history(db_path, transaction_id):
    """Décisions déjà prises sur UNE transaction (la plus récente en premier)."""
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM decisions WHERE transaction_id = ? ORDER BY id DESC",
            conn, params=(str(transaction_id),),
        )
    return df.drop(columns=["id"]).rename(columns=_DB_TO_APP)[COLUMNS]