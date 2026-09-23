# Prévision de la demande retail — de la donnée au stock de sécurité

Projet de data science de bout en bout sur le jeu **Favorita** (Kaggle, *Store Sales – Time Series Forecasting*) :
prévoir les ventes journalières par magasin et famille de produits, **quantifier l'incertitude**, puis
**traduire la prévision en décision de stock** (stock de sécurité, point de commande) — et le mesurer par simulation.

> **Question business** : combien commander, et quand, pour servir la demande sans immobiliser de stock inutile ?

## Ce que fait le projet

| Étape | Contenu |
|---|---|
| 1. EDA | Tendance, saisonnalités hebdo/annuelle, promos, jours fériés, ruptures, STL, ACF (`notebooks/01_eda.ipynb`) |
| 2. Baselines | Naïf saisonnier (7 j), moyenne mobile 28 j — le niveau à battre |
| 3. Modèles statistiques | ETS (Holt-Winters), SARIMA(1,0,1)(1,1,1)₇, Prophet |
| 4. Modèle ML | LightGBM **global** (toutes les séries), stratégie **directe** sur 14 jours, décalages ≥ horizon |
| 5. Validation | **Rolling-origin** (6 origines consécutives), WAPE / MAE / RMSE / biais / MASE, par modèle, horizon et famille |
| 6. Incertitude | Régressions quantiles (5 % / 95 %) → intervalle à 90 %, couverture et pinball loss mesurées |
| 7. Décision de stock | Stock de sécurité = z × σ de l'erreur **cumulée sur le délai** ; simulation d'une politique *base-stock* |
| 8. Mise à disposition | API FastAPI (`/forecast`, `/reorder`) + app Streamlit, Docker, config Render |

## Démarrage rapide

```bash
pip install -r requirements.txt && pip install -e .

# A) Sans télécharger de données : test du pipeline sur données synthétiques (~1 min)
make demo

# B) Sur Favorita (résultats réels)
#    1. Télécharger les données sur Kaggle (Store Sales – Time Series Forecasting)
#    2. Placer les CSV dans data/raw/
#    3. python -m demandforecast.pipeline --data-dir data/raw --horizon 14 --n-folds 6
make test               # 22 tests
make app                # app Streamlit  (http://localhost:8501)
make api                # API            (http://localhost:8000/docs)
```

Le pipeline écrit dans `reports/` : `summary.md` (note de synthèse générée), `metrics_by_*.csv`, `predictions.csv`,
`inventory_simulation.csv`, `figures/*.png`, et dans `models/artifacts.joblib` le modèle final.

> Les données Favorita ne sont pas redistribuables (règles Kaggle) : `data/raw/` est ignoré par git.

## Résultats (Favorita, 36 séries, backtest rolling-origin 6 folds, horizon 14 jours)

| Modèle | WAPE | MAE | Biais | MASE |
|---|---|---|---|---|
| **LightGBM** | **0.085** | 455 | −0.014 | **0.563** |
| SARIMA | 0.121 | 645 | +0.042 | 0.778 |
| ETS | 0.121 | 648 | +0.039 | 0.782 |
| Prophet | 0.142 | 762 | +0.082 | 0.904 |
| Naïf saisonnier | 0.155 | 832 | +0.046 | 0.983 |
| Moyenne mobile | 0.222 | 1 186 | +0.023 | 1.403 |

LightGBM atteint un WAPE de **8,5 %**, soit **45 % de mieux** que le naïf saisonnier (15,5 %).

**Intervalles de prévision** (LightGBM, quantiles 5 %–95 %) : couverture empirique **87 %** (cible 90 %) —
les intervalles sont légèrement trop étroits, à corriger avant usage en production.

**Simulation de stock** (délai 3 jours, taux de service cible 95 %) :

| Modèle | Taux de service | Stock moyen (jours de demande) |
|---|---|---|
| **LightGBM** | 99,4 % | **0,36** |
| SARIMA | 99,5 % | 0,68 |
| ETS | 99,4 % | 0,69 |
| Prophet | 99,7 % | 0,81 |
| Naïf saisonnier | 99,7 % | 0,85 |
| Moyenne mobile | 99,0 % | 0,93 |

LightGBM atteint un taux de service comparable à tous les modèles avec **deux fois moins de stock** (0,36 j vs 0,68–0,93 j).

**WAPE par famille de produits** (LightGBM) :

| Famille | WAPE |
|---|---|
| PRODUCE | 0,068 |
| DAIRY | 0,077 |
| GROCERY I | 0,076 |
| BREAD/BAKERY | 0,082 |
| BEVERAGES | 0,105 |
| CLEANING | 0,109 |

## Structure

```
src/demandforecast/
  data.py        chargement Favorita, calendrier complet, générateur synthétique
  features.py    variables (décalages >= horizon, fenêtres glissantes, calendrier, fériés, promos)
  models.py      baselines, ETS, SARIMA, Prophet, LightGBM direct + quantiles
  metrics.py     WAPE, biais, MASE, pinball, couverture
  inventory.py   stock de sécurité, point de commande, simulation base-stock
  pipeline.py    backtest rolling-origin, rapport, graphiques, artefact
  serve.py       prévision et recommandation depuis l'artefact
api/main.py      FastAPI          app/streamlit_app.py   interface
tests/           22 tests         notebooks/             analyse exploratoire + modélisation pas à pas
```

## Choix méthodologiques (à savoir défendre en entretien)

- **Rolling-origin, jamais de split aléatoire** : en série temporelle, mélanger passé et futur fuit de l'information.
  Six origines consécutives donnent une estimation plus stable qu'un seul découpage.
- **WAPE plutôt que MAPE** : le MAPE explose (ou est indéfini) quand les ventes sont proches de zéro, fréquent ici.
  Le WAPE (Σ|e| / Σ ventes) se lit directement comme « % d'erreur sur les volumes ». Le **biais** est suivi séparément :
  sur-prévision = surstock, sous-prévision = rupture.
- **Stratégie directe et zéro fuite** : un seul modèle prévoit 14 jours d'un coup, donc toute variable issue des ventes passées
  est décalée d'au moins 14 jours. Test dédié (`test_no_leakage_from_future_sales`), vérifié sensible par mutation.
- **Modèle global** : un LightGBM pour toutes les séries partage l'information (effet promo, jours fériés) et
  se déploie plus simplement que des dizaines de modèles individuels.
- **Objectif Tweedie** : ventes ≥ 0, asymétriques, avec zéros.
- **σ de l'erreur cumulée** et non σ_jour × √L : les erreurs de prévision successives sont autocorrélées, la racine carrée
  sous-estimerait le risque.

## API

```bash
curl -X POST localhost:8000/forecast -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "family": "BEVERAGES", "horizon": 7}'

curl -X POST localhost:8000/reorder -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "family": "BEVERAGES", "lead_time": 3, "service_level": 0.95}'
```

## Déploiement (Render)

`render.yaml` et `Dockerfile` sont fournis. Le modèle étant ignoré par git, l'ajouter pour le déploiement
(≈ 3 Mo) : `git add -f models/artifacts.joblib`.

## Limites assumées

- Les **ventes ne sont pas la demande** : en cas de rupture, la demande réelle est censurée (le modèle apprend les zéros).
- **Simulation de stock simplifiée** : délai fixe, pas de lot minimal ni de coût de commande ; le stock de sécurité est
  calibré sur la période de test elle-même, donc les taux de service sont légèrement optimistes.
- Le **plan promo futur** est supposé connu (`onpromotion`) ; en production, il doit venir du métier.
- Sous-ensemble de magasins × familles (paramétrable) ; pas de hiérarchie ni de réconciliation.
- Couverture de l'intervalle à 90 % : **87 % empiriquement**, les intervalles sont légèrement trop étroits.

## Pistes d'amélioration

Réconciliation hiérarchique (magasin → région → total) · calibration des intervalles (conformal prediction) ·
demande censurée (modèle de rupture) · variables exogènes (prix du pétrole, transactions) ·
optimisation des hyperparamètres (Optuna) · suivi de dérive (Evidently) et ré-entraînement planifié.

alent ; API FastAPI et application Streamlit déployées.
