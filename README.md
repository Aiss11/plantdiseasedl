# Détection des maladies des feuilles de tomate par CNN

Projet de fin de module **Deep Learning** (AI Hub, cohorte 3).
Un CNN entraîné sur des photos de laboratoire (PlantVillage) reste-t-il fiable sur des photos prises au champ (PlantDoc) ?
Ce dépôt mesure cet écart et teste des stratégies pour le réduire.

## Contenu du dépôt

```
plant-disease-cnn/
├── README.md
├── requirements.txt
├── scripts/
│   └── download_data.py        # télécharge PlantVillage + PlantDoc (tomate uniquement)
├── notebooks/
│   └── Projet_Deep_Learning.ipynb   # notebook complet (exécuté sur Colab, GPU T4)
├── docs/
│   ├── Enonce_projet.pdf
│   ├── Fiche_impact.docx       # fiche d'impact (1 page)
│   └── Rapport_technique.docx  # rapport technique
├── data/                       # données téléchargées (ignorées par git)
├── models/                     # modèles .keras générés (ignorés par git)
└── results/
```

## Données

| Jeu de données | Contenu | Source |
|---|---|---|
| PlantVillage | Feuilles photographiées en laboratoire, fond uniforme | https://github.com/spMohanty/PlantVillage-Dataset |
| PlantDoc | Photos en conditions réelles (fonds chargés, lumière variable) | https://github.com/pratikkayal/PlantDoc-Dataset |

**Culture ciblée : la tomate.** Sept classes présentes dans les deux jeux de données ont été retenues :
`Bacterial_spot`, `Early_blight`, `Late_blight`, `Leaf_Mold`, `Septoria_leaf_spot`, `Tomato_mosaic_virus`, `Yellow_Leaf_Curl_Virus`.
`Spider_mites` est écartée (trop peu d'images dans PlantDoc).

| Ensemble | Images |
|---|---|
| PlantVillage entraînement (70 %, stratifié) | 9 442 |
| PlantVillage validation (15 %) | 2 023 |
| PlantVillage test (15 %) | 2 024 |
| PlantDoc entraînement (dont 496 train + 124 val pour le fine-tuning) | 620 |
| PlantDoc test indépendant (jamais utilisé pour l'entraînement) | 61 |

### Télécharger les données

```bash
python scripts/download_data.py                      # -> ./data
python scripts/download_data.py --dest /content/data # sur Colab
python scripts/download_data.py --verify-only        # vérifie sans télécharger
```

Le script nécessite uniquement `git` (≥ 2.25). Il utilise un clone partiel avec *sparse checkout* et ne récupère que les dossiers de tomate (environ 1 Go au lieu de ~3 Go pour les dépôts complets ; option `--full` pour tout cloner).
À la fin, il affiche le nombre d'images par classe et vérifie que les 7 classes sont présentes dans les deux jeux de données.

## Instructions d'exécution

### Option A : Google Colab (recommandé, c'est l'environnement d'origine)

1. Ouvrir `notebooks/Projet_Deep_Learning.ipynb` dans Colab (Fichier → Importer un notebook, ou depuis GitHub).
2. Activer le GPU : *Exécution → Modifier le type d'exécution → GPU (T4)*.
3. Exécuter toutes les cellules dans l'ordre. La section 2 télécharge les données dans `/content/data`.
   Variante plus légère : exécuter d'abord `!git clone <url-du-depot>` puis `!python plant-disease-cnn/scripts/download_data.py --dest /content/data` et sauter la cellule de clonage de la section 2.

### Option B : en local

```bash
git clone <url-du-depot> && cd plant-disease-cnn
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                      # remplit ./data
jupyter notebook notebooks/Projet_Deep_Learning.ipynb
```

Le notebook utilise des chemins Colab (`/content/data/...` pour les données, `/content/*.keras` pour les modèles).
En local, remplacez `/content/data` par `../data` et `/content/` par `../models/` dans les cellules concernées, et supprimez la cellule `!git clone` de la section 2 (les données sont déjà téléchargées par le script).

Les graines aléatoires sont fixées à 42 pour les découpages (`train_test_split`) et le mélange. Les résultats peuvent varier légèrement d'une exécution à l'autre à cause de l'augmentation de données et du GPU.

## Méthodologie

| Étape | Contenu |
|---|---|
| Prétraitement | Images redimensionnées en 128×128, RGB, pixels normalisés dans [0, 1], lots de 32 |
| MLP (baseline) | Flatten → Dense 128 → Dropout 0,5 → Softmax (6,3 M paramètres) |
| CNN 1 | 2 blocs Conv + MaxPooling → Dense 128 → Dropout 0,5 (7,4 M paramètres) |
| CNN 2 | Augmentation (flip, rotation, zoom, contraste) + 3 blocs Conv/BatchNorm/ReLU/MaxPooling + GlobalAveragePooling + Dense 128 + Dropout 0,4 (111 K paramètres) |
| Robustesse | Entraînement sur PlantVillage, test sur PlantDoc ; fine-tuning sur PlantDoc (BatchNorm gelées, lr = 1e-4) ; entraînement mixte PlantVillage + PlantDoc |
| Analyse | Rapport par classe (précision, rappel, F1), matrices de confusion, erreurs, Grad-CAM |

Optimiseur Adam, perte *sparse categorical crossentropy*, EarlyStopping sur `val_loss`.

## Résultats principaux

| Modèle | Accuracy PlantVillage | Accuracy PlantDoc | F1 macro PlantDoc |
|---|---|---|---|
| MLP (baseline) | 40,91 % | non évalué | non évalué |
| CNN 1 | 90,42 % | 19,67 % | 0,10 |
| CNN 2 | 89,58 % | **24,59 %** | **0,19** |
| CNN 2 fine-tuné sur PlantDoc | non évalué | 13,11 % | 0,09 |
| CNN 2 entraînement mixte | non évalué | 22,95 % | 0,16 |

À retenir :
- Le MLP est insuffisant (il prédit presque toujours `Yellow_Leaf_Curl_Virus`), ce qui justifie le CNN.
- Les deux CNN dépassent 89 % sur PlantVillage mais **s'effondrent sur PlantDoc** (≈ 20–25 %) : c'est l'écart laboratoire / terrain.
- Ni le fine-tuning ni l'entraînement mixte n'améliorent le CNN 2 initial. PlantVillage représente environ 95 % des données du mélange, et PlantDoc n'offre que quelques dizaines d'images par classe.
- Grad-CAM, sur un seul exemple, suggère que le modèle s'appuie en partie sur l'arrière-plan plutôt que sur la lésion. C'est une hypothèse à confirmer sur davantage d'images.

## Limites

- Le jeu de test PlantDoc ne compte que **61 images** : les métriques varient beaucoup avec quelques prédictions.
- Le découpage PlantVillage est fait par image et non par plante : une dépendance entre images d'un même spécimen n'a pas été vérifiée.
- Fort déséquilibre des classes (ex. `Yellow_Leaf_Curl_Virus` : 3 750 images d'entraînement contre 261 pour `Tomato_mosaic_virus`).
- Les expériences de robustesse n'ont pas fait l'objet de plusieurs répétitions : pas d'intervalle de confiance.
- L'outil est une aide à la décision, pas un diagnostic agronomique définitif. Voir `docs/Rapport_technique.docx` (section « Limites et risques »).

## Livrables

- Notebook reproductible : `notebooks/Projet_Deep_Learning.ipynb`
- Fiche d'impact : `docs/Fiche_impact.docx`
- Rapport technique : `docs/Rapport_technique.docx`
- Dépôt GitHub documenté : ce dépôt

## Licence et données

Le code et les documents de ce dépôt sont fournis dans un cadre pédagogique. Les jeux de données PlantVillage et PlantDoc appartiennent à leurs auteurs respectifs : consultez leurs dépôts pour les conditions d'utilisation.
