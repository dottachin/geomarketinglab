# Méthodologie GeoMarketingLab

## Unité spatiale

Chaque ville pilote (Casablanca, Rabat, Marrakech) est découpée en hexagones H3 de résolution 9
(environ 0,1 km² par hexagone). Toutes les sources sont agrégées sur cette grille commune.
Les distances et surfaces sont calculées en projection métrique (UTM 29N, EPSG:32629) ; les géométries
sont stockées en WGS 84 (EPSG:4326).

## Sources de données

OpenStreetMap fournit les commerces, équipements (écoles, universités, mosquées, hôpitaux, marchés),
arrêts de tram et de bus, routes et bâtiments. L'extrait Geofabrik du Maroc est mis à jour chaque jour
par fichiers de différences. Google Earth Engine fournit la population WorldPop, la luminosité nocturne
VIIRS, la part de surface bâtie (Dynamic World) et les indices Sentinel-2 (NDVI, NDBI). Le Haut-Commissariat
au Plan (HCP) fournit les indicateurs du recensement RGPH 2024 par commune et arrondissement : population,
ménages, structure par âge, éducation, emploi, logement.

## Désagrégation des données du HCP

Les données du HCP sont publiées par commune ou arrondissement. Les effectifs (population, ménages,
établissements) sont répartis entre les hexagones proportionnellement à la population WorldPop
(interpolation dasymétrique). Les taux et pourcentages (chômage, éducation, type de logement) sont
moyennés selon la surface de recouvrement. Ces valeurs restent des estimations de contexte de quartier,
pas des mesures propres à chaque hexagone.

## Ce que le modèle prédit

Le modèle prédit la probabilité qu'un hexagone accueille déjà un commerce du type choisi, à partir de
toutes les caractéristiques sauf le nombre de commerces de ce type. Un hexagone avec une forte
adéquation mais peu de commerces existants est une opportunité. Hypothèse importante : les commerces
existants sont supposés être bien placés (biais du survivant). Le score n'est pas une prévision de
chiffre d'affaires.

## Score final et pénalité de saturation

Le score final est l'adéquation calibrée multipliée par (1 − λ × saturation). La saturation compare le
nombre de concurrents dans la zone de chalandise à la capacité estimée, égale à la population de la
zone divisée par le nombre d'habitants qu'un commerce peut servir. Une zone sans concurrent garde toute
son adéquation ; une zone déjà saturée est pénalisée.

## Évaluation

Les modèles (LightGBM, réseau de neurones avec embedding du type de commerce, réseau de graphes
GraphSAGE sur les hexagones) sont comparés par validation croisée spatiale : les hexagones sont groupés
par hexagone parent H3 de résolution 7, et les cellules voisines des blocs de test sont retirées de
l'entraînement pour éviter les fuites. Les métriques sont l'AUC-PR, la précision@k et le NDCG@k.

## Explications et niveau de confiance

Chaque recommandation est accompagnée des valeurs SHAP : les facteurs qui augmentent ou diminuent le
score. Une recommandation est signalée « faible confiance » lorsque la couverture OpenStreetMap est
faible autour de l'hexagone, lorsque les données satellitaires ou du recensement manquent, ou lorsque
le modèle a vu peu d'exemples de ce commerce dans la ville.

## Attribution et licences

Données cartographiques © contributeurs OpenStreetMap, licence ODbL. Données du recensement : HCP,
RGPH 2024. Données satellitaires : Copernicus Sentinel-2, NOAA VIIRS, WorldPop, Google Dynamic World,
via Google Earth Engine (usage académique).
