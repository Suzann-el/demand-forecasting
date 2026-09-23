# Note de synthèse — prévision de la demande

**Données** : favorita — 6 magasins x 6 familles,
2015-01-01 → 2017-08-15.
**Validation** : rolling-origin, 6 origines successives, horizon 14 jours
(test : 2017-05-24 → 2017-08-15).

## Comparaison des modèles
| model            |   WAPE |      MAE |     RMSE |   Biais |   MASE |
|:-----------------|-------:|---------:|---------:|--------:|-------:|
| lightgbm         |  0.085 |  454.809 |  775.982 |  -0.014 |  0.563 |
| sarima           |  0.121 |  644.895 | 1057.369 |   0.042 |  0.778 |
| ets              |  0.121 |  647.979 | 1068.641 |   0.039 |  0.782 |
| prophet          |  0.142 |  761.914 | 1169.966 |   0.082 |  0.904 |
| naive_saisonnier |  0.155 |  831.582 | 1399.391 |   0.046 |  0.983 |
| moyenne_mobile   |  0.222 | 1186.212 | 1814.270 |   0.023 |  1.403 |

Meilleur modèle : **lightgbm** (WAPE 0.085) soit **45 %** de mieux que le naïf saisonnier.
Biais > 0 : sur-prévision (risque de surstock) ; biais < 0 : sous-prévision (risque de rupture).

## Intervalles de prévision (LightGBM, quantiles 5 % - 95 %)
|   Couverture 90 % (cible 0.90) |   Pinball q5 |   Pinball q95 |
|-------------------------------:|-------------:|--------------:|
|                          0.870 |       64.841 |        76.675 |

Une couverture inférieure à la cible nominale de 90 % signifie des intervalles trop étroits :
à prendre en compte avant de s'en servir pour dimensionner un stock.

## Impact sur le stock (simulation base-stock, délai 3 j, niveau de service 95%)
| model            |   fill_rate |   stockout_day_rate |   avg_inventory |   avg_inventory_days |
|:-----------------|------------:|--------------------:|----------------:|---------------------:|
| naive_saisonnier |       0.997 |               0.024 |      164276.288 |                0.846 |
| moyenne_mobile   |       0.990 |               0.040 |      180505.848 |                0.930 |
| ets              |       0.994 |               0.028 |      133830.969 |                0.689 |
| sarima           |       0.995 |               0.028 |      131315.756 |                0.677 |
| prophet          |       0.997 |               0.021 |      156201.038 |                0.805 |
| lightgbm         |       0.994 |               0.048 |       69475.005 |                0.358 |

`fill_rate` = part de la demande servie ; `avg_inventory_days` = stock moyen en rayon (hors commandes en transit) exprimé en jours de demande.
Le meilleur modèle est celui qui offre le meilleur taux de service **à stock moyen comparable**, ou moins de stock à taux de service comparable.

## Erreur par famille de produits
| family       |   ets |   lightgbm |   moyenne_mobile |   naive_saisonnier |   prophet |   sarima |
|:-------------|------:|-----------:|-----------------:|-------------------:|----------:|---------:|
| BEVERAGES    | 0.130 |      0.105 |            0.260 |              0.167 |     0.164 |    0.131 |
| BREAD/BAKERY | 0.100 |      0.082 |            0.181 |              0.123 |     0.108 |    0.105 |
| CLEANING     | 0.156 |      0.109 |            0.215 |              0.215 |     0.143 |    0.141 |
| DAIRY        | 0.122 |      0.077 |            0.214 |              0.129 |     0.114 |    0.120 |
| GROCERY I    | 0.132 |      0.076 |            0.209 |              0.172 |     0.122 |    0.132 |
| PRODUCE      | 0.086 |      0.068 |            0.204 |              0.111 |     0.161 |    0.086 |

## Limites
- Les ventes observées censurent la demande réelle (ruptures) : le modèle apprend des ventes, pas de la demande.
- La simulation de stock est simplifiée (pas de lot minimal, pas de coût, délai fixe) et le stock de sécurité est calibré en
  échantillon sur la période de test : les taux de service sont légèrement optimistes.
- Promotions futures supposées connues (`onpromotion`) ; en production, ce plan promo doit être fourni par le métier.
