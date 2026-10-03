'''Dashboard analyste — détection de transactions atypiques'''
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.scoring import explain, fit_unsupervised, load_artifacts, score_transactions
from src.history import add_decision, get_transaction_history, init_db, load_history

# 1. CONFIGURATION
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

QUEUE_PATH = DATA_DIR / "analyst_queue.csv"

# HISTORY_PATH = DATA_DIR / "decision_history.csv"
# DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = DATA_DIR / "decision_history.csv"      # ancien format (migré une fois)
DB_PATH = DATA_DIR / "decisions.db"                    # historique des décisions (SQLite)


DATA_DIR.mkdir(parents=True, exist_ok=True)
init_db(DB_PATH, legacy_csv=HISTORY_PATH)

REQUIRED_COLUMNS = ["TransactionID", "TransactionAmt", "RiskScore", "RiskLevel"]
HISTORY_COLUMNS = ["TransactionID", "AnalystDecision", "AnalystComment",
                   "ReviewDate", "PreviousRiskLevel", "RiskScore"]
TEXT_COLUMNS = ["ReviewStatus", "AnalystDecision", "AnalystComment", "ReviewDate"]
EXPLAIN_TOP_N = 1000   # nombre de transactions expliquées pour un fichier importé


RISK_COLORS = {
    "High": "#FF0000",    
    "Medium": "#FFA500",  
    "Low": "#0C9D59"     
}

# 2. FONCTIONS
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_csv(df, path):
    df.to_csv(path, index=False)


def initialize_queue(df):
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne(s) obligatoire(s) absente(s) : {missing}")

    df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    df["RiskScore"] = pd.to_numeric(df["RiskScore"], errors="coerce")

    defaults = {"ReviewStatus": "Pending", "AnalystDecision": "Not reviewed",
                "AnalystComment": "", "ReviewDate": ""}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    # Un CSV relit les colonnes vides en float : on force le texte
    # (sinon l'écriture d'une décision lève une erreur avec pandas 3)
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)
    return df


def prepare_upload(raw, force_unsupervised=False):
    """Fichier déjà scoré -> utilisé tel quel. Fichier brut -> scoré ici."""
    if all(c in raw.columns for c in REQUIRED_COLUMNS):
        return raw.drop(columns=["isFraud"], errors="ignore")
    missing = [c for c in ("TransactionID", "TransactionAmt") if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Fichier non reconnu : colonnes {missing} absentes. "
            "Fournir un fichier scoré (RiskScore, RiskLevel) ou un fichier brut de transactions."
        )

    artifacts = None if force_unsupervised else load_artifacts(MODELS_DIR)
    if artifacts is None:
        artifacts = fit_unsupervised(raw)
        st.sidebar.info("Mode non supervisé : Isolation Forest entraîné sur ce fichier, "
                        "sans étiquettes.")
    # scored = score_transactions(raw, artifacts)

    if "Currency" in raw.columns:          # conserve la devise du fichier (ex. MAD)
        scored["Currency"] = raw["Currency"]
    scored = score_transactions(raw, artifacts)

    top = scored.sort_values("RiskScore", ascending=False).head(EXPLAIN_TOP_N)
    scored = scored.join(explain(raw.loc[top.index], artifacts, reference=raw))
    for col in ("TopRiskFactors", "AnomalyReasons"):
        scored[col] = scored[col].fillna("")
    return raw.drop(columns=["isFraud"], errors="ignore")


def get_queue(uploaded_file, force_unsupervised=False):
    """Fichier importé : conservé en session (les décisions survivent au rerun)."""
    if uploaded_file is not None:
        key = f"{uploaded_file.name}-{uploaded_file.size}-{force_unsupervised}"
        if st.session_state.get("upload_key") != key:
            with st.spinner("Analyse du fichier importé..."):
                raw = pd.read_csv(uploaded_file)
                st.session_state["queue_df"] = initialize_queue(prepare_upload(raw, force_unsupervised))
            st.session_state["upload_key"] = key
        return st.session_state["queue_df"]
    st.session_state.pop("upload_key", None)
    q = load_csv(QUEUE_PATH)
    return initialize_queue(q) if not q.empty else q


def risk_color(level):
    return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(level, "⚪")


# 3. CHARGEMENT
col1, col2 = st.columns([1, 8])
with col1:
    logo = ASSETS_DIR / "detection.png"
    if logo.exists():
        st.image(str(logo), width=100)
with col2:
    st.title("Fraud Detection — Analyst Dashboard")

st.markdown(
    "Interface de surveillance des transactions atypiques. "
    "Les alertes sont classées par score de risque pour faciliter l'examen par un analyste."
)

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier de transactions (CSV, brut ou déjà scoré)", type=["csv"]
)
unsupervised_only = st.sidebar.checkbox(
    "Mode non supervisé uniquement (sans XGBoost)", value=False,
    help="À cocher pour des données qui ne ressemblent pas à IEEE-CIS (ex. données synthétiques en MAD).",
)
analyst_name = st.sidebar.text_input("Nom de l'analyste (pour l'historique)")
currency_choice = st.sidebar.selectbox(
    "Devise affichée (libellé uniquement, sans conversion)", ["USD", "MAD"]
)

try:
    df = get_queue(uploaded_file, unsupervised_only)
    if df.empty:
        st.warning("Aucune donnée trouvée. Exécute d'abord la cellule d'export du notebook.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

currency = df["Currency"].iloc[0] if "Currency" in df.columns and df["Currency"].notna().any() else currency_choice
if uploaded_file is not None and "ScoringMode" in df.columns:
    st.sidebar.caption(f"Mode de scoring : {df['ScoringMode'].iloc[0]}")

# 4. FILTRES
st.sidebar.header("Filtres")

amounts = df["TransactionAmt"].dropna()
min_amount = float(amounts.min()) if not amounts.empty else 0.0
max_amount = float(amounts.max()) if not amounts.empty else 0.0

st.sidebar.markdown(f"**Montant ({currency})**")
amt_col1, amt_col2 = st.sidebar.columns(2)
min_input = amt_col1.number_input("Min", min_value=0.0, value=min_amount, step=10.0, format="%.2f")
max_input = amt_col2.number_input("Max", min_value=0.0, value=max_amount, step=10.0, format="%.2f")
if min_input > max_input:
    st.sidebar.error("Le montant min doit être inférieur au max.")
    amount_range = (min_amount, max_amount)
else:
    amount_range = (min_input, max_input)

risk_options = ["Low", "Medium", "High"]
selected_risks = st.sidebar.multiselect("Niveau de risque", risk_options, default=risk_options)

status_options = ["Pending", "Reviewed"]
selected_status = st.sidebar.multiselect("Statut de vérification", status_options, default=status_options)

if "ProductCD" in df.columns:
    # Liste fixe des catégories IEEE-CIS + celles éventuellement présentes dans le fichier
    category_options = sorted(set(["C", "H", "R", "S", "W"]) | set(df["ProductCD"].dropna().astype(str)))
    selected_categories = st.sidebar.multiselect(
        "Catégorie de transaction", category_options, default=category_options
    )
else:
    selected_categories = None

date_column = next(
    (c for c in ["ReconstructedDate", "TransactionDate", "Date"] if c in df.columns), None
)
selected_dates = None
if date_column:
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = df[date_column].dropna()
    if not valid_dates.empty:
        selected_dates = st.sidebar.date_input(
            "Période",
            value=(valid_dates.min().date(), valid_dates.max().date()),
            min_value=valid_dates.min().date(),
            max_value=valid_dates.max().date(),
        )
else:
    st.sidebar.caption("Pas de colonne de date dans ce fichier.")

# 5. APPLICATION DES FILTRES
mask = (
    df["TransactionAmt"].between(amount_range[0], amount_range[1])
    & df["RiskLevel"].astype(str).isin(selected_risks)
    & df["ReviewStatus"].astype(str).isin(selected_status)
)
if selected_categories is not None:
    mask &= df["ProductCD"].astype(str).isin(selected_categories)
if date_column and isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask &= df[date_column].dt.date.between(start_date, end_date)
filtered_df = df[mask].copy()

# 6. INDICATEURS
st.subheader("Vue d'ensemble")
total_transactions = len(filtered_df)
high_risk_count = int((filtered_df["RiskLevel"] == "High").sum())
pending_count = int((filtered_df["ReviewStatus"] == "Pending").sum())
average_risk = filtered_df["RiskScore"].mean() if total_transactions else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions affichées", f"{total_transactions:,}")
m2.metric("Risque élevé", f"{high_risk_count:,}")
m3.metric("En attente", f"{pending_count:,}")
m4.metric("Score de risque moyen", f"{average_risk:.2f}/100")

# 7. GRAPHIQUES
# st.subheader("Analyse des risques")
# chart_col1, chart_col2 = st.columns(2)

# with chart_col1:
#     risk_distribution = (
#         filtered_df["RiskLevel"].value_counts().rename_axis("RiskLevel").reset_index(name="Count")
#     )
#     if not risk_distribution.empty:
#         st.plotly_chart(
#             px.pie(risk_distribution, names="RiskLevel", values="Count",
#                    title="Répartition par niveau de risque", hole=0.4),
#             width="stretch",
#         )

# with chart_col2:
#     if not filtered_df.empty:
#         st.plotly_chart(
#             px.histogram(filtered_df, x="TransactionAmt", color="RiskLevel",
#                          title=f"Distribution des montants ({currency})", nbins=30),
#             width="stretch",
#         )


# # 7. GRAPHIQUES

# st.subheader("Analyse des risques")
# chart_col1, chart_col2 = st.columns(2)

# with chart_col1:
#     risk_distribution = (
#     filtered_df["RiskLevel"]
#     .value_counts()
#     .rename_axis("RiskLevel")
#     .reset_index(name="Count")
# )


# if not risk_distribution.empty:
#     fig_pie = px.pie(
#         risk_distribution,
#         names="RiskLevel",
#         values="Count",
#         title="Répartition par niveau de risque",
#         hole=0.4,
#         color="RiskLevel",
#         color_discrete_map=RISK_COLORS
#     )

#     st.plotly_chart(fig_pie, width="stretch")


# with chart_col2:
#     if not filtered_df.empty:
#         fig_hist = px.histogram(
#         filtered_df,
#         x="TransactionAmt",
#         color="RiskLevel",
#         title=f"Distribution des montants ({currency})",
#         nbins=30,
#         color_discrete_map=RISK_COLORS,
#         category_orders={"RiskLevel": ["Low", "Medium", "High"]}
# )

# st.plotly_chart(fig_hist, width="stretch")

# 7. GRAPHIQUES
st.subheader("Analyse des risques")
if filtered_df.empty:
    st.info("Aucune donnée à représenter avec ces filtres.")
else:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        risk_distribution = (filtered_df["RiskLevel"].value_counts()
                             .rename_axis("RiskLevel").reset_index(name="Count"))
        fig_pie = px.pie(risk_distribution, names="RiskLevel", values="Count",
                         title="Répartition par niveau de risque", hole=0.4,
                         color="RiskLevel", color_discrete_map=RISK_COLORS)
        st.plotly_chart(fig_pie, width="stretch")
    with chart_col2:
        fig_hist = px.histogram(filtered_df, x="TransactionAmt", color="RiskLevel",
                                title=f"Distribution des montants ({currency})", nbins=30,
                                color_discrete_map=RISK_COLORS,
                                category_orders={"RiskLevel": ["Low", "Medium", "High"]})
        st.plotly_chart(fig_hist, width="stretch")

# 8. TABLEAU DES ALERTES
st.subheader("Transactions à examiner")
display_columns = [
    c for c in ["TransactionID", date_column, "TransactionAmt", "ProductCD", "FraudProbability",
                "AnomalyScore", "RiskScore", "RiskLevel", "ReviewStatus"]
    if c and c in filtered_df.columns
]
st.dataframe(
    filtered_df[display_columns].sort_values("RiskScore", ascending=False),
    width="stretch", hide_index=True,
)

# 9 + 10. EXAMEN ET DÉCISION (tout est dans le else : pas d'erreur si aucun résultat)
st.subheader("Examen d'une transaction")

if filtered_df.empty:
    st.info("Aucune transaction ne correspond aux filtres.")
else:
    selected_id = st.selectbox("Sélectionner une transaction",
                               filtered_df["TransactionID"].astype(str).tolist())
    selected_transaction = filtered_df[
        filtered_df["TransactionID"].astype(str) == selected_id
    ].iloc[0]

    d1, d2, d3 = st.columns(3)
    d1.metric("Montant", f"{selected_transaction['TransactionAmt']:,.2f} {currency}")
    d2.metric("Risk Score", f"{selected_transaction['RiskScore']:.2f}/100")
    d3.metric("Niveau", f"{risk_color(selected_transaction['RiskLevel'])} {selected_transaction['RiskLevel']}")

    # Explications (SHAP / atypicité)
    explanations = [
        ("TopRiskFactors", "Facteurs ayant le plus augmenté le score (XGBoost / SHAP)"),
        ("AnomalyReasons", "Caractéristiques les plus atypiques (Isolation Forest)"),
    ]
    shown = False
    for col, title in explanations:
        text = str(selected_transaction.get(col, "")).strip()
        if text and text.lower() != "nan":
            st.markdown(f"**{title}**")
            for item in text.split(" | "):
                st.write(f"• {item}")
            shown = True
    if not shown:
        st.caption("Aucune explication disponible pour cette transaction.")

    with st.expander("Toutes les informations de la transaction"):
        st.json({k: (None if pd.isna(v) else str(v)) for k, v in selected_transaction.to_dict().items()})

    if "FraudProbability" in selected_transaction.index and pd.notna(selected_transaction["FraudProbability"]):
        st.write("Probabilité estimée par XGBoost :", f"{selected_transaction['FraudProbability']:.4f}")
    if "AnomalyScore" in selected_transaction.index:
        st.write("Score brut Isolation Forest (bas = plus atypique) :", selected_transaction["AnomalyScore"])
    st.caption("Ces scores sont des indicateurs du modèle. Ils ne constituent pas une preuve de fraude.")

    st.subheader("Décision de l'analyste")
    previous = get_transaction_history(DB_PATH, selected_id)
    if not previous.empty:
        st.caption(f"Cette transaction a déjà {len(previous)} décision(s) enregistrée(s) :")
        st.dataframe(previous, width="stretch", hide_index=True)
    with st.form("analyst_review_form"):
        analyst_decision = st.selectbox(
            "Décision", ["Fraud suspected", "Legitimate", "Escalate", "Need more information"]
        )
        analyst_comment = st.text_area("Commentaire de l'analyste")
        submitted = st.form_submit_button("Enregistrer la décision")

    if submitted:
        row_mask = df["TransactionID"].astype(str) == selected_id
        review_time = datetime.now().isoformat(timespec="seconds")

        df.loc[row_mask, "AnalystDecision"] = analyst_decision
        df.loc[row_mask, "AnalystComment"] = analyst_comment
        df.loc[row_mask, "ReviewStatus"] = "Reviewed"
        df.loc[row_mask, "ReviewDate"] = review_time

        if uploaded_file is None:
            save_csv(df, QUEUE_PATH)
        else:
            st.session_state["queue_df"] = df

        # history = load_csv(HISTORY_PATH)
        # if history.empty:
        #     history = pd.DataFrame(columns=HISTORY_COLUMNS)
        # new_decision = pd.DataFrame([{
        #     "TransactionID": selected_transaction["TransactionID"],
        #     "AnalystDecision": analyst_decision,
        #     "AnalystComment": analyst_comment,
        #     "ReviewDate": review_time,
        #     "PreviousRiskLevel": selected_transaction["RiskLevel"],
        #     "RiskScore": selected_transaction["RiskScore"],
        # }])
        # save_csv(pd.concat([history, new_decision], ignore_index=True), HISTORY_PATH)
        add_decision(
            DB_PATH,
            transaction_id=selected_id,
            decision=analyst_decision,
            comment=analyst_comment,
            previous_risk_level=selected_transaction["RiskLevel"],
            risk_score=selected_transaction["RiskScore"],
            analyst=analyst_name,
            source=uploaded_file.name if uploaded_file is not None else "analyst_queue",
            review_date=review_time,
        )
        st.success("Décision enregistrée dans l'historique.")
        st.rerun()

# 11. HISTORIQUE
st.subheader("Historique des décisions")
# history = load_csv(HISTORY_PATH)
history = load_history(DB_PATH)

if history.empty:
    st.info("Aucune décision enregistrée pour le moment.")
else:
    st.dataframe(history.sort_values("ReviewDate", ascending=False), width="stretch", hide_index=True)
    st.download_button("Télécharger l'historique CSV",
                       data=history.to_csv(index=False).encode("utf-8"),
                       file_name="decision_history.csv", mime="text/csv")

# 12. EXPORT
st.subheader("Export")
st.download_button("Télécharger les transactions filtrées",
                   data=filtered_df.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_transactions.csv", mime="text/csv")

st.caption("Prototype de dashboard analyste — les décisions humaines sont enregistrées localement en CSV.")