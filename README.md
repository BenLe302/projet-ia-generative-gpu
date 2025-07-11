# Projet IA Générative - Génération d'Art avec GANs

## 📋 Description

Ce projet implémente un système complet de génération d'art utilisant des réseaux antagonistes génératifs (GANs). Il comprend plusieurs architectures de GANs (DCGAN, CycleGAN) pour générer des œuvres d'art dans différents styles, ainsi qu'une API REST pour servir les modèles entraînés.

## 🎯 Objectifs

- **Génération d'art** : Créer des œuvres d'art originales dans différents styles
- **Transfert de style** : Transformer des images existantes selon différents styles artistiques
- **API de production** : Servir les modèles via une API REST performante
- **Interface utilisateur** : Fournir une interface web pour interagir avec les modèles

## 🏗️ Architecture

```
Projet 8 IA Générative/
├── src/
│   ├── config.py              # Configuration du projet
│   ├── data/                  # Gestion des données
│   │   ├── __init__.py
│   │   ├── dataset.py         # Classes de dataset
│   │   ├── preprocessing.py   # Préprocessing des images
│   │   └── analysis.py        # Analyse du dataset
│   ├── models/                # Modèles GAN
│   │   ├── __init__.py
│   │   ├── base_gan.py        # Classe de base
│   │   ├── dcgan.py           # DCGAN
│   │   ├── cyclegan.py        # CycleGAN
│   │   ├── losses.py          # Fonctions de perte
│   │   └── utils.py           # Utilitaires
│   ├── training/              # Entraînement
│   │   ├── __init__.py
│   │   ├── trainer.py         # Classes d'entraînement
│   │   ├── callbacks.py       # Callbacks d'entraînement
│   │   ├── metrics.py         # Métriques d'évaluation
│   │   └── utils.py           # Utilitaires d'entraînement
│   └── api/                   # API REST
│       ├── __init__.py
│       ├── app.py             # Application FastAPI
│       ├── models.py          # Modèles Pydantic
│       └── utils.py           # Utilitaires API
├── Dataset/                   # Données d'entraînement
├── models/                    # Modèles sauvegardés
├── logs/                      # Logs et métriques
├── train.py                   # Script d'entraînement
├── run_api.py                 # Script de lancement API
├── requirements.txt           # Dépendances
├── .env.example              # Variables d'environnement
└── README.md                 # Documentation
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA 11.8+ (optionnel, pour GPU)
- 8GB+ RAM
- 4GB+ VRAM (pour entraînement GPU)

### Installation des dépendances

```bash
# Cloner le projet
git clone <repository-url>
cd "Projet 8 IA Générative"

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

```bash
# Copier le fichier de configuration
cp .env.example .env

# Éditer les variables d'environnement
# Ajuster les chemins selon votre configuration
```

## 📊 Dataset

Le projet utilise un dataset d'art situé dans `Dataset/`. La structure attendue :

```
Dataset/
├── artists.csv              # Métadonnées des artistes
└── images/
    ├── artist1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── artist2/
        ├── image1.jpg
        └── image2.jpg
```

### Analyse du dataset

```bash
# Analyser le dataset avant l'entraînement
python train.py --analyze
```

## 🎓 Entraînement

### DCGAN (Deep Convolutional GAN)

```bash
# Entraînement basique
python train.py --model dcgan --epochs 100

# Avec paramètres personnalisés
python train.py --model dcgan \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.0002 \
    --seed 42

# Avec arrêt précoce
python train.py --model dcgan \
    --epochs 500 \
    --early-stopping \
    --patience 20
```

### CycleGAN

```bash
# Entraînement CycleGAN
python train.py --model cyclegan --epochs 200
```

### Options d'entraînement

- `--model` : Type de modèle (`dcgan`, `cyclegan`)
- `--epochs` : Nombre d'époques
- `--batch-size` : Taille du batch
- `--learning-rate` : Taux d'apprentissage
- `--seed` : Graine aléatoire
- `--early-stopping` : Arrêt précoce
- `--patience` : Patience pour l'arrêt précoce
- `--log-level` : Niveau de logging
- `--analyze` : Analyser le dataset

## 🌐 API

### Lancement de l'API

```bash
# Lancement basique
python run_api.py

# Avec paramètres personnalisés
python run_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4

# Mode développement
python run_api.py --reload --log-level DEBUG
```

### Endpoints disponibles

#### Santé de l'API
```http
GET /health
```

#### Liste des modèles
```http
GET /models
```

#### Génération d'images
```http
POST /generate
Content-Type: application/json

{
    "model_name": "dcgan_final",
    "num_images": 4,
    "seed": 42,
    "temperature": 1.0,
    "format": "png"
}
```

#### Interpolation dans l'espace latent
```http
POST /interpolate
Content-Type: application/json

{
    "model_name": "dcgan_final",
    "start_seed": 42,
    "end_seed": 123,
    "steps": 10,
    "interpolation_type": "linear"
}
```

#### Transfert de style (CycleGAN)
```http
POST /style-transfer
Content-Type: application/json

{
    "model_name": "cyclegan_final",
    "image_data": "base64_encoded_image",
    "direction": "A2B"
}
```

### Documentation interactive

Une fois l'API lancée, accédez à :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

## 📈 Monitoring et Métriques

### Métriques d'évaluation

- **FID (Fréchet Inception Distance)** : Qualité des images générées
- **IS (Inception Score)** : Diversité et qualité
- **LPIPS** : Similarité perceptuelle
- **SSIM** : Similarité structurelle

### Logs et visualisations

```bash
# Les logs sont sauvegardés dans logs/
logs/
├── training_*.log           # Logs d'entraînement
├── api_*.log               # Logs de l'API
├── metrics.json            # Métriques d'entraînement
├── loss_plots.png          # Graphiques de perte
└── generated_images/       # Images générées pendant l'entraînement
```

## 🔧 Configuration avancée

### Variables d'environnement

```bash
# Chemins
PROJECT_ROOT=/path/to/project
DATASET_PATH=/path/to/dataset
MODELS_PATH=/path/to/models

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Entraînement
BATCH_SIZE=32
LEARNING_RATE=0.0002
EPOCHS=100

# GPU
CUDA_VISIBLE_DEVICES=0
```

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Tests avec couverture
pytest --cov=src

# Tests spécifiques
pytest tests/test_models.py
pytest tests/test_api.py
```

## 📝 Exemples d'utilisation

### Génération d'images via Python

```python
import torch
from src.models import DCGAN
from src.api.utils import tensor_to_pil

# Charger le modèle
model = DCGAN.load_model("models/dcgan_final.pth")
model.eval()

# Générer des images
with torch.no_grad():
    noise = torch.randn(4, 100, 1, 1)
    fake_images = model.generator(noise)
    
    # Convertir en PIL
    for i, tensor in enumerate(fake_images):
        image = tensor_to_pil(tensor)
        image.save(f"generated_{i}.png")
```

### Utilisation de l'API via curl

```bash
# Générer des images
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "dcgan_final",
       "num_images": 2,
       "seed": 42
     }'

# Interpolation
curl -X POST "http://localhost:8000/interpolate" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "dcgan_final",
       "start_seed": 42,
       "end_seed": 123,
       "steps": 5
     }'
```

## 🐛 Dépannage

### Erreurs communes

**CUDA Out of Memory**
```bash
# Réduire la taille du batch
python train.py --batch-size 16

# Ou utiliser le CPU
CUDA_VISIBLE_DEVICES="" python train.py
```

**Dataset non trouvé**
```bash
# Vérifier le chemin dans .env
echo $DATASET_PATH

# Ou spécifier explicitement
python train.py --dataset-path /path/to/dataset
```

**Modèle non chargé**
```bash
# Vérifier les modèles disponibles
ls models/

# Vérifier les logs
tail -f logs/api_*.log
```

## 📚 Ressources

### Papers de référence

- [DCGAN](https://arxiv.org/abs/1511.06434) - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- [CycleGAN](https://arxiv.org/abs/1703.10593) - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- [FID](https://arxiv.org/abs/1706.08500) - GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium

### Documentation

- [PyTorch](https://pytorch.org/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Albumentations](https://albumentations.ai/docs/)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Cyril** - *Développement initial* - [BenLe302](https://github.com/BenLe302)

## 🙏 Remerciements

- Dataset d'art utilisé
- Communauté PyTorch
- Papers de recherche en GANs
- Contributeurs open source