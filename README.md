# Deep Learning — Détection et Reconnaissance Faciale

Projet réalisé dans le cadre d'une évaluation de deep learning. L'objectif était de construire un pipeline complet de détection et reconnaissance faciale, et d'analyser les biais que le modèle peut introduire selon les conditions d'éclairage.

## Ce que fait ce projet

**Partie 1: Classification CIFAR-100**  
Comparaison d'un MLP et d'un CNN sur le dataset CIFAR-100 (100 classes, 60 000 images). Le CNN atteint ~54% de précision contre ~14% pour le MLP — ce qui montre bien l'intérêt des convolutions pour les images.

**Partie 2: Détection et biais**  
- Détection de visages sur une image réelle avec OpenCV (Haar Cascade)
- Entraînement d'un CNN de reconnaissance sur Olivetti Faces (40 personnes, 400 images)
- Analyse des biais : les images sombres ou peu contrastées sont systématiquement moins bien reconnues
- Correction avec CLAHE (égalisation adaptative du contraste) → +9 points de précision sur le test


## Résultats principaux

| Modèle | Accuracy test |
|--------|--------------|
| CNN original | 35% |
| CNN après correction CLAHE | 48.25% |

La disparité de performance entre individus (mesurée par l'écart-type) passe de σ=0.396 à σ=0.384 après correction — le modèle est légèrement plus équitable.



## Stack

Python 3.13 · TensorFlow 2.21 · Keras 3.14 · OpenCV · scikit-learn · Matplotlib



## Fichiers

- `Deep_Learning_CIFAR100.ipynb` — Partie 1
- `Partie2_Detection_Visage_Biais.ipynb` — Partie 2
- `image.jpeg` — image de test pour la détection
