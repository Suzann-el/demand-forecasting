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
#    1. Accepter les règles de la compétition sur Kaggle, placer kaggle.json dans ~/.kaggle/
#    2. make data      # télécharge train.csv, holidays_events.csv… dans data/raw/
#    3. make pipeline  # backtest + rapport + modèle final
make test               # 22 tests
make app                # app Streamlit  (http://localhost:8501)
make api                # API            (http://localhost:8000/docs)
```

Le pipeline écrit dans `reports/` : `summary.md` (note de synthèse générée), `metrics_by_*.csv`, `predictions.csv`,
`inventory_simulation.csv`, `figures/*.png`, et dans `models/artifacts.joblib` le modèle final.

> Les données Favorita ne sont pas redistribuables (règles Kaggle) : `data/raw/` est ignoré par git.
> Un exemple de rapport produit sur données **synthétiques** est dans `docs/demo_synthetique/` — il ne montre que le format de sortie.

## Résultats (à compléter après `make pipeline` sur Favorita)

Copier ici le tableau de `reports/summary.md` et 2 ou 3 figures. Ne citer que des chiffres issus de **votre** exécution sur Favorita.

| Modèle | WAPE | Biais | MASE |
|---|---|---|---|
| … | … | … | … |

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
tests/           22 tests         notebooks/01_eda.ipynb  analyse exploratoire
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
- **Cohérence entraînement / service** : `test_serving_matches_batch_features` vérifie que l'API redonne exactement
  les prévisions du batch (le *train-serve skew* est une cause classique de bugs en production).

## API

```bash
curl -X POST localhost:8000/forecast -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "family": "BEVERAGES", "horizon": 7}'

curl -X POST localhost:8000/reorder -H 'Content-Type: application/json' \
  -d '{"store_nbr": 1, "family": "BEVERAGES", "lead_time": 3, "service_level": 0.95}'
```

## Déploiement (Render)

`render.yaml` et `Dockerfile` sont fournis. Le modèle étant ignoré par git, l'ajouter pour le déploiement
(≈ 3 Mo) : `git add -f models/artifacts.joblib`. Le `Dockerfile` n'a pas été construit dans l'environnement où le projet a été
généré : premier `docker build` à faire de votre côté.

## Limites assumées

- Les **ventes ne sont pas la demande** : en cas de rupture, la demande réelle est censurée (le modèle apprend les zéros).
- **Simulation de stock simplifiée** : délai fixe, pas de lot minimal ni de coût de commande ; le stock de sécurité est
  calibré sur la période de test elle-même, donc les taux de service sont légèrement optimistes.
- Le **plan promo futur** est supposé connu (`onpromotion`) ; en production, il doit venir du métier.
- Sous-ensemble de magasins × familles (paramétrable) ; pas de hiérarchie ni de réconciliation.

## Pistes d'amélioration

Réconciliation hiérarchique (magasin → région → total) · prévision probabiliste par conformal prediction ·
demande censurée (modèle de rupture) · variables exogènes (prix du pétrole, transactions, événements locaux) ·
optimisation des hyperparamètres (Optuna) avec validation temporelle · suivi de dérive (Evidently) et ré-entraînement planifié.

## Présenter ce projet (CV / entretien)

Modèle de phrase, à remplir avec **vos** chiffres :

> Prévision de la demande sur 14 jours (Favorita, N séries) : LightGBM global + intervalles quantiles, validé en rolling-origin ;
> WAPE de **X %** contre **Y %** pour le naïf saisonnier ; simulation de stock montrant **Z** points de taux de service
> à stock équivalent ; API FastAPI et application Streamlit déployées.
