# Guide de Configuration GPU pour RTX 4060 Ti

Ce guide vous aidera à configurer votre NVIDIA RTX 4060 Ti pour l'entraînement de modèles GAN.

## 🚀 Installation Rapide

### Étape 1: Exécuter le script d'installation automatique

```bash
python install_pytorch_cuda.py
```

### Étape 2: Tester la configuration

```bash
python test_gpu.py
```

### Étape 3: Lancer l'entraînement

```bash
python train.py --model dcgan --epochs 10
```

## 📋 Prérequis

- ✅ NVIDIA RTX 4060 Ti installée
- ✅ Pilotes NVIDIA récents (version 525.60.11+)
- ✅ Python 3.8+ installé
- ⚠️ CUDA Toolkit 12.1 ou 11.8 (sera installé automatiquement)

## Étapes de Configuration

### 1. Vérifier la Version CUDA Compatible

Votre GPU RTX 4060 Ti supporte CUDA 11.8+ et CUDA 12.x. Je recommande CUDA 12.1 pour les meilleures performances.

### 2. Installer PyTorch avec Support CUDA

#### Option A: CUDA 12.1 (Recommandé)
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CUDA 11.8 (Alternative)
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Vérifier l'Installation

Après installation, testez avec:
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Version CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 4. Optimisations pour RTX 4060 Ti

#### Configuration Recommandée
- **Batch Size**: 16-32 (selon la taille des images)
- **Mixed Precision**: Activé (AMP)
- **Gradient Accumulation**: 2-4 steps si nécessaire
- **Memory Management**: Optimisé

#### Modifications du Code

1. **Activer Mixed Precision** dans `src/training/trainer.py`:
```python
from torch.cuda.amp import autocast, GradScaler

class GANTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()
```

2. **Optimiser la Gestion Mémoire**:
```python
# Dans train_step
with autocast(enabled=self.use_amp):
    # Forward pass
    fake_images = self.model.generate(noise)
    
if self.use_amp:
    self.scaler.scale(loss).backward()
    self.scaler.step(optimizer)
    self.scaler.update()
else:
    loss.backward()
    optimizer.step()
```

### 5. Configuration Optimale pour l'Entraînement

#### Paramètres Recommandés
```python
# Configuration pour RTX 4060 Ti
config = {
    'batch_size': 24,  # Ajustable selon la taille d'image
    'image_size': 256,  # Ou 512 si assez de VRAM
    'num_workers': 6,   # Nombre de cœurs CPU / 2
    'pin_memory': True,
    'persistent_workers': True,
    'learning_rate': 0.0002,
    'mixed_precision': True
}
```

### 6. Monitoring GPU

Pendant l'entraînement, surveillez:
- **Utilisation GPU**: `nvidia-smi -l 1`
- **Température**: < 80°C
- **Utilisation VRAM**: < 14GB (garder 2GB de marge)

### 7. Commandes d'Entraînement Optimisées

```bash
# Entraînement DCGAN avec GPU
python train.py --model dcgan --epochs 100 --batch-size 24 --use-gpu

# Avec monitoring
watch -n 1 nvidia-smi
```

## Avantages Attendus

- **Vitesse**: 10-20x plus rapide qu'avec CPU
- **Batch Size**: Plus grand (24-32 vs 8-16 sur CPU)
- **Qualité**: Entraînement plus stable avec plus d'époques
- **Temps**: 1-2h au lieu de 10-20h pour 100 époques

## Dépannage

### Erreur "CUDA out of memory"
- Réduire `batch_size` à 16 ou 12
- Réduire `image_size` à 128 ou 64
- Activer `torch.cuda.empty_cache()` entre les époques

### Performance Lente
- Vérifier que `pin_memory=True`
- Augmenter `num_workers`
- Utiliser `persistent_workers=True`

### GPU Non Détectée
- Redémarrer après installation PyTorch CUDA
- Vérifier les drivers NVIDIA
- Tester avec `torch.cuda.is_available()`

## Prochaines Étapes

1. Installer PyTorch CUDA
2. Tester la détection GPU
3. Modifier le code pour utiliser mixed precision
4. Relancer l'entraînement avec GPU
5. Monitorer les performances

Votre RTX 4060 Ti est excellente pour l'entraînement de GANs ! Avec 16GB de VRAM, vous pourrez entraîner des modèles de haute qualité.