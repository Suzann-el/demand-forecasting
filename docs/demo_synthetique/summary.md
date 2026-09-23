# Note de synthèse — prévision de la demande

> ⚠️ **Rapport généré sur données SYNTHÉTIQUES** (test du pipeline). Ne pas citer ces chiffres comme résultats réels : relancer sur Favorita.

**Données** : synthetic — 4 magasins x 5 familles,
2015-01-01 → 2017-08-15.
**Validation** : rolling-origin, 6 origines successives, horizon 14 jours
(test : 2017-05-24 → 2017-08-15).

## Comparaison des modèles
| model            |   WAPE |     MAE |    RMSE |   Biais |   MASE |
|:-----------------|-------:|--------:|--------:|--------:|-------:|
| lightgbm         |  0.180 | 173.292 | 262.219 |  -0.015 |  0.816 |
| prophet          |  0.182 | 174.977 | 260.424 |  -0.005 |  0.819 |
| ets              |  0.185 | 177.440 | 263.966 |  -0.021 |  0.833 |
| sarima           |  0.190 | 182.656 | 268.846 |  -0.049 |  0.858 |
| moyenne_mobile   |  0.216 | 207.192 | 300.533 |  -0.021 |  0.965 |
| naive_saisonnier |  0.246 | 236.638 | 344.114 |  -0.003 |  1.116 |

Meilleur modèle : **lightgbm** (WAPE 0.180) soit **27 %** de mieux que le naïf saisonnier.
Biais > 0 : sur-prévision (risque de surstock) ; biais < 0 : sous-prévision (risque de rupture).

## Intervalles de prévision (LightGBM, quantiles 5 % - 95 %)
|   Couverture 90 % (cible 0.90) |   Pinball q5 |   Pinball q95 |
|-------------------------------:|-------------:|--------------:|
|                          0.866 |       25.847 |        26.334 |

Une couverture inférieure à la cible nominale de 90 % signifie des intervalles trop étroits :
à prendre en compte avant de s'en servir pour dimensionner un stock.

## Impact sur le stock (simulation base-stock, délai 3 j, niveau de service 95%)
| model            |   fill_rate |   stockout_day_rate |   avg_inventory |   avg_inventory_days |
|:-----------------|------------:|--------------------:|----------------:|---------------------:|
| naive_saisonnier |       0.989 |               0.040 |       18843.793 |                0.975 |
| moyenne_mobile   |       0.990 |               0.051 |       16144.096 |                0.835 |
| ets              |       0.992 |               0.050 |       13842.674 |                0.716 |
| sarima           |       0.987 |               0.074 |       12140.128 |                0.628 |
| prophet          |       0.993 |               0.042 |       14240.902 |                0.737 |
| lightgbm         |       0.993 |               0.049 |       13508.315 |                0.699 |

`fill_rate` = part de la demande servie ; `avg_inventory_days` = stock moyen en rayon (hors commandes en transit) exprimé en jours de demande.
Le meilleur modèle est celui qui offre le meilleur taux de service **à stock moyen comparable**, ou moins de stock à taux de service comparable.

## Erreur par famille de produits
| family    |   ets |   lightgbm |   moyenne_mobile |   naive_saisonnier |   prophet |   sarima |
|:----------|------:|-----------:|-----------------:|-------------------:|----------:|---------:|
| BEVERAGES | 0.203 |      0.188 |            0.229 |              0.268 |     0.196 |    0.198 |
| CLEANING  | 0.185 |      0.175 |            0.213 |              0.260 |     0.179 |    0.183 |
| DAIRY     | 0.170 |      0.172 |            0.198 |              0.231 |     0.167 |    0.185 |
| GROCERY I | 0.185 |      0.188 |            0.223 |              0.240 |     0.190 |    0.188 |
| PRODUCE   | 0.179 |      0.175 |            0.210 |              0.239 |     0.174 |    0.196 |

## Limites
- Les ventes observées censurent la demande réelle (ruptures) : le modèle apprend des ventes, pas de la demande.
- La simulation de stock est simplifiée (pas de lot minimal, pas de coût, délai fixe) et le stock de sécurité est calibré en
  échantillon sur la période de test : les taux de service sont légèrement optimistes.
- Promotions futures supposées connues (`onpromotion`) ; en production, ce plan promo doit être fourni par le métier.
