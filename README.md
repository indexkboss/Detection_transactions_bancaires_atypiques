# 🛡️ Détection des transactions atypiques

Application d'aide à l'analyste pour **repérer rapidement les opérations financières inhabituelles** dans un fichier de transactions anonymisées, **même sans exemples de fraude étiquetés**.

Le projet combine un modèle **non supervisé** (Isolation Forest), un modèle **supervisé** (XGBoost, quand des étiquettes sont disponibles), un **score de risque** de 0 à 100 et des **explications lisibles** de chaque alerte, le tout dans un dashboard avec validation humaine et historique des décisions.

> ⚠️ Les scores sont des indicateurs statistiques d'aide à la décision. Ils ne constituent **pas** une preuve de fraude : la décision finale revient toujours à l'analyste.

---

## Sommaire

1. [Fonctionnalités](#1-fonctionnalités)
2. [Fonctionnement](#2-fonctionnement)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Données](#5-données)
6. [Utilisation](#6-utilisation)
7. [Méthodologie](#7-méthodologie)
8. [Résultats](#8-résultats)
9. [Limites et choix techniques](#9-limites-et-choix-techniques)
10. [Pistes d'amélioration](#10-pistes-damélioration)

---

## 1. Fonctionnalités

| Besoin du MVP | Réalisation |
|---|---|
| Import d'un fichier de transactions | Téléversement CSV, fichier **brut** (scoré à l'import) ou **déjà scoré** |
| Dashboard analyste | Indicateurs clés, répartition des niveaux de risque, distribution des montants, tableau d'alertes trié par risque |
| Score de risque | Score 0–100 et niveau `Low` / `Medium` / `High` |
| Filtres | Montant (min/max), période, catégorie de transaction, niveau de risque, statut de vérification |
| Explication de l'anomalie | Facteurs SHAP (XGBoost) et caractéristiques les plus atypiques (Isolation Forest), en langage métier |
| Validation humaine | Décision (`Fraud suspected`, `Legitimate`, `Escalate`, `Need more information`) + commentaire |
| Historique des décisions | Journal **SQLite en ajout seul** (rien n'est écrasé), nom de l'analyste, source du fichier, export CSV |
| Mode sans étiquettes | Isolation Forest entraîné directement sur le fichier importé, sans aucune étiquette |

---

## 2. Fonctionnement

```
                        ┌──────────────────────────┐
  Fichier CSV  ───────► │  build_features()        │  montant (log), heure (sin/cos),
  de transactions       │  (src/scoring.py)        │  catégories, présence d'identité…
                        └────────────┬─────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        Isolation Forest                           XGBoost (si modèle
        (sans étiquettes)                          entraîné disponible)
                  │                                     │
          AnomalyNormalized                       FraudProbability
                  └──────────────────┬──────────────────┘
                                     ▼
                  RiskScore = 0,7 × XGBoost + 0,3 × anomalie   (mode hybride)
                  RiskScore = anomalie                          (mode non supervisé)
                                     ▼
                       Low / Medium / High  +  explications
                                     ▼
                  Dashboard  →  décision analyste  →  historique (SQLite)
```

---

## 3. Structure du projet

```
.
├── app.py                      # Dashboard Streamlit
├── requirements.txt
├── README.md
├── src/
│   ├── scoring.py              # features, scoring, explications (partagé notebook + app)
│   └── history.py              # historique des décisions (SQLite)
├── notebooks/
│   └── 1_data_understanding.ipynb   # EDA, modélisation, évaluation, export
├── scripts/
│   └── generate_synthetic_mad.py    # données synthétiques en dirhams
├── models/                     # *.joblib générés par le notebook
├── data/
│   ├── raw/                    # fichiers IEEE-CIS (non versionnés)
│   ├── processed/              # analyst_queue.csv, test_results.csv, decisions.db
│   └── synthetic/              # données synthétiques MAD générées
└── assets/                     # logo du dashboard
```

---

## 4. Installation

Prérequis : **Python 3.13** (version utilisée pendant le développement) et `pip`.

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

# 2. Installer les dépendances
pip install -r requirements.txt
```

Bibliothèques principales : `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `streamlit`, `plotly`, `matplotlib`, `seaborn`, `jupyterlab`.

> 💡 Les fichiers `models/*.joblib` dépendent des versions de `scikit-learn` et `xgboost`. Fige-les avec `pip freeze > requirements.txt` et réexécute le notebook si tu changes de version.

---

## 5. Données

### IEEE-CIS Fraud Detection (entraînement et évaluation)
Téléchargement sur [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data), puis dépôt dans `data/raw/` de `train_transaction.csv` et `train_identity.csv`.

- 590 540 transactions, 3,5 % de fraudes (jeu très déséquilibré)
- Les fichiers `test_*.csv` de Kaggle n'ont pas d'étiquette `isFraud` : ils ne sont pas utilisés. L'évaluation se fait sur un **split temporel** du fichier d'entraînement.

### Données synthétiques en dirhams (démonstration)
Générées par un script, avec des anomalies injectées (montants extrêmes, achats de nuit élevés, profils rares) :

```bash
python scripts/generate_synthetic_mad.py --evaluate
```

Deux fichiers sont créés dans `data/synthetic/` :
- `synthetic_mad.csv` : à importer dans le dashboard (aucune étiquette, devise `MAD`) ;
- `synthetic_mad_labels.csv` : vérité terrain, **réservée à l'évaluation**.

Options : `--n` (nombre de transactions), `--rate` (proportion d'anomalies), `--seed`.

### Credit Card Fraud Detection (non utilisé)
Ce jeu ne contient que des composantes PCA anonymisées (`V1`–`V28`), le temps et le montant. Sans variable métier (catégorie, domaine email, type de carte…), il ne permet pas de produire des explications lisibles par un analyste, ce qui est au cœur du projet.

---

### Format du fichier à importer

Colonnes **obligatoires** : `TransactionID`, `TransactionAmt`.

Colonnes **optionnelles** (les absentes sont traitées comme valeurs manquantes) :
`TransactionDT` (secondes depuis le 2017-12-01) **ou** `TransactionDate`, `ProductCD`, `card4`, `card6`, `addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `Currency`.

Un fichier déjà scoré (avec `RiskScore` et `RiskLevel`) est utilisé tel quel. Une colonne `isFraud` éventuelle est **toujours ignorée** pour ne pas exposer l'étiquette à l'analyste.

---

## 6. Utilisation

### Étape 1 : entraîner et exporter (une seule fois)
Ouvre `notebooks/1_data_understanding.ipynb` et exécute toutes les cellules. La dernière cellule produit :
- `models/*.joblib` (préprocesseur, XGBoost, Isolation Forest, normalisateur) ;
- `data/processed/analyst_queue.csv` : 1 000 transactions à examiner (les 500 plus risquées + 500 tirées au hasard dans le reste, avec leurs explications, **sans** `isFraud`) ;
- `data/processed/test_results.csv` : résultats complets du jeu de test (usage interne, contient `isFraud`).

### Étape 2 : lancer le dashboard
```bash
streamlit run app.py
```

### Étape 3 : travailler comme analyste
1. Saisis ton nom dans la barre latérale (il sera enregistré avec chaque décision).
2. Sans import, le dashboard affiche `analyst_queue.csv`. Tu peux aussi **importer un fichier CSV**.
3. Filtre les alertes (montant, période, catégorie, risque, statut).
4. Sélectionne une transaction : consulte le score, les facteurs de risque et les caractéristiques atypiques.
5. Choisis une décision, ajoute un commentaire, enregistre. La décision apparaît dans l'historique, et les décisions précédentes sur la même transaction restent visibles.

### Tester avec des données en dirhams
1. `python scripts/generate_synthetic_mad.py`
2. Dans le dashboard, importe `data/synthetic/synthetic_mad.csv`.
3. **Coche « Mode non supervisé uniquement »** : le modèle XGBoost est entraîné sur IEEE-CIS et n'a pas de sens pour ces données. L'Isolation Forest est alors entraîné sur le fichier importé.

---

## 7. Méthodologie

**Séparation temporelle.** Les transactions sont triées par date puis découpées en 70 % entraînement / 15 % validation / 15 % test. Le modèle apprend sur le passé et est évalué sur le futur, ce qui simule un usage réel. Les taux de fraude sont comparables entre les trois blocs (3,5 % / 3,4 % / 3,5 %).

**Variables retenues (16).** Montant et son logarithme, catégorie (`ProductCD`), type et réseau de carte, adresses, distances, domaines email, heure (avec encodage cyclique sinus/cosinus), présence d'informations d'identité, type d'appareil. L'identifiant `TransactionID` n'est jamais donné au modèle.

**Modèles.**
- *Baseline* : régression logistique pondérée (`class_weight="balanced"`).
- *XGBoost* : `scale_pos_weight` ≈ 27,4 pour compenser le déséquilibre, métrique d'évaluation `aucpr`.
- *Isolation Forest* : 200 arbres, entraîné **sans** `isFraud`.

**Seuil de décision XGBoost.** Choisi sur la **validation** en maximisant le F1 (0,787), puis figé pour le test. Le jeu de test n'intervient jamais dans un choix de paramètre.

**Score de risque.**
- Mode hybride : `RiskScore = (0,7 × probabilité XGBoost + 0,3 × anomalie normalisée) × 100`
- Niveaux : `Low` < 30 ≤ `Medium` < 70 ≤ `High`
- Mode non supervisé : `RiskScore = anomalie normalisée × 100`. Comme les seuils 30/70 n'ont alors pas de sens, les niveaux sont définis par **rang** : top 5 % = `High`, 80ᵉ–95ᵉ percentile = `Medium`.

**Explicabilité.** Deux explications par transaction, affichées en langage métier :
1. *Facteurs de risque* : contributions SHAP du modèle XGBoost, regroupées par variable d'origine (par exemple toutes les colonnes `ProductCD_*` forment un seul facteur) ;
2. *Caractéristiques atypiques* : variables les plus éloignées de la médiane du fichier (numériques) ou les plus rares (catégorielles), calculées sur **l'ensemble** du fichier importé.

**Métriques.** Précision, rappel, F1, **PR-AUC** (métrique principale, adaptée aux classes déséquilibrées), ROC-AUC et **taux de faux positifs**. L'accuracy n'est pas utilisée : prédire « jamais de fraude » donnerait 96,5 %.

---

## 8. Résultats

### IEEE-CIS : jeu de test temporel (88 581 transactions, 3 083 fraudes)

| Modèle | Précision | Rappel | F1 | Faux positifs | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost (seuil F1 de la validation) | 0,247 | 0,263 | 0,255 | 2,9 % | 0,788 | 0,188 |
| Isolation Forest (sans étiquettes) | 0,088 | 0,089 | 0,089 | 3,3 % | 0,685 | 0,069 |
| Score combiné (0,7 / 0,3) | 0,215 | 0,331 | 0,261 | 4,4 % | 0,787 | 0,171 |

Taux de fraude de référence : 3,5 %.

**Niveaux du score combiné (seuils 30 / 70) :**

| Niveau | Transactions | Taux de fraude réel |
|---|---|---|
| High | 2 664 | 25,2 % |
| Medium | 31 724 | 5,5 % |
| Low | 54 193 | 1,3 % |

Le niveau `High` concentre donc des fraudes **7 fois plus fréquemment** que la base (précision 25,2 %, rappel 21,7 %, faux positifs 2,3 %). Le niveau `Low` en contient près de 3 fois moins.

### Données synthétiques en dirhams (mode non supervisé)
Résultats indicatifs avec `--seed 42`, 5 000 transactions et 2 % d'anomalies injectées :

| PR-AUC | ROC-AUC | Niveau High : précision / rappel / faux positifs |
|---|---|---|
| 0,42 | 0,93 | 0,26 / 0,64 / 3,8 % |

Le niveau `High` (top 5 %) contient volontairement plus d'alertes que d'anomalies réelles (2 %), ce qui plafonne la précision.

---

## 9. Limites et choix techniques

- **Performance modeste sur IEEE-CIS** (PR-AUC 0,19). Seules 16 variables sur 434 sont utilisées, pour garder des explications lisibles. Les groupes `C`, `D`, `M`, `V`, `card1–5` et `id_*` restent à exploiter.
- **L'Isolation Forest n'améliore pas XGBoost** dans le score combiné sur ces données (PR-AUC 0,171 contre 0,188). Il garde son intérêt quand aucune étiquette n'est disponible, et pour l'explication des atypies.
- **`HasIdentity`** vaut 1 à l'entraînement si la transaction figure dans `train_identity`. À l'import d'un fichier sans cette table, il est approximé par la présence de `DeviceType`.
- **Devise** : « USD » (IEEE-CIS, supposée) ou « MAD » est un **libellé** : aucune conversion n'est faite.
- **Taille de l'échantillon de l'Isolation Forest.** En mode non supervisé, chaque arbre voit jusqu'à 2 048 lignes (au lieu de 256 par défaut) : avec 31 colonnes surtout binaires après encodage, 256 lignes donnaient un PR-AUC de 0,17 contre 0,42 sur les données synthétiques.
- **Données synthétiques** : anomalies simples et volontairement détectables, utiles pour la démonstration, pas pour juger la performance en production.
- **Stack** : le sujet suggère FastAPI / React / PostgreSQL. Ce prototype utilise **Streamlit** (interface rapide à construire côté Python) et **SQLite** (historique sans serveur), suffisants pour un MVP mono-utilisateur. Le code de scoring, isolé dans `src/scoring.py`, est réutilisable derrière une API FastAPI.
- **Pas d'authentification** ni de gestion multi-utilisateurs ; le statut de vérification de la file d'attente est conservé en CSV.
- **Dérive des données** : les modèles ne sont pas réentraînés automatiquement.

---

## 10. Pistes d'amélioration

1. Enrichir les variables (`card1–5`, `C1–C14`, `D1–D15`, `M1–M9`, `id_*`) et mesurer le gain en PR-AUC.
2. Ajuster le poids du score (0,7 / 0,3) et les seuils 30 / 70 sur la validation, ou calibrer les probabilités.
3. Remplacer le CSV de la file d'attente par une table SQLite / PostgreSQL et ajouter une authentification des analystes.
4. Exposer le scoring via une API **FastAPI** et un front **React**.
5. Intégrer le retour des analystes (décisions `Legitimate` / `Fraud suspected`) pour réentraîner périodiquement le modèle supervisé.
6. Superviser la dérive (distribution des scores, taux d'alertes) dans le temps.

---

## Auteur

Projet réalisé par *(à compléter)*, dans le cadre de *(à compléter)*.