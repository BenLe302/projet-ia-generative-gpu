#!/usr/bin/env python3
"""
Script principal d'entraînement pour les modèles GAN
Supporte DCGAN et CycleGAN avec optimisations GPU automatiques
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_gpu_info():
    """Récupère les informations GPU et ajuste automatiquement les paramètres"""
    if not torch.cuda.is_available():
        logger.warning("CUDA non disponible. Utilisation du CPU.")
        return {
            'device': 'cpu',
            'gpu_name': 'CPU',
            'vram_gb': 0,
            'batch_size': 8,
            'num_workers': 2
        }
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    
    # Ajustement automatique du batch_size selon la VRAM
    if vram_gb < 8:  # RTX 3060 Laptop (6GB)
        batch_size = 12
        num_workers = 4
    elif vram_gb < 12:  # RTX 3060 Ti, 4060 (8GB)
        batch_size = 16
        num_workers = 6
    elif vram_gb < 16:  # RTX 4060 Ti (12GB)
        batch_size = 20
        num_workers = 8
    else:  # RTX 4070+, 4080, 4090 (16GB+)
        batch_size = 24
        num_workers = 8
    
    logger.info(f"GPU détectée: {gpu_name}")
    logger.info(f"VRAM disponible: {vram_gb:.1f} GB")
    logger.info(f"Batch size ajusté: {batch_size}")
    logger.info(f"Workers: {num_workers}")
    
    return {
        'device': device,
        'gpu_name': gpu_name,
        'vram_gb': vram_gb,
        'batch_size': batch_size,
        'num_workers': num_workers
    }

def analyze_dataset(dataset_path):
    """Analyse le dataset et retourne les statistiques"""
    logger.info(f"Analyse du dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset non trouvé: {dataset_path}")
        return None
    
    # Compter les images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_count = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_count += 1
    
    logger.info(f"Nombre d'images trouvées: {image_count}")
    return {'image_count': image_count}

def create_dcgan_model(gpu_info, image_size=64, nz=100, ngf=64, ndf=64):
    """Crée un modèle DCGAN optimisé pour le GPU détecté"""
    from src.models.dcgan import Generator, Discriminator
    
    device = gpu_info['device']
    
    # Ajuster la taille selon la VRAM
    if gpu_info['vram_gb'] < 8:
        image_size = 64
        ngf = ndf = 64
    elif gpu_info['vram_gb'] < 12:
        image_size = 128
        ngf = ndf = 64
    else:
        image_size = 256
        ngf = ndf = 64
    
    logger.info(f"Création DCGAN - Taille image: {image_size}x{image_size}")
    
    generator = Generator(nz=nz, ngf=ngf, nc=3, image_size=image_size).to(device)
    discriminator = Discriminator(ndf=ndf, nc=3, image_size=image_size).to(device)
    
    # Compter les paramètres
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    logger.info(f"Paramètres Generator: {gen_params:,}")
    logger.info(f"Paramètres Discriminator: {disc_params:,}")
    
    return {
        'generator': generator,
        'discriminator': discriminator,
        'image_size': image_size,
        'nz': nz
    }

def create_cyclegan_model(gpu_info):
    """Crée un modèle CycleGAN optimisé pour le GPU détecté"""
    from src.models.cyclegan import Generator, Discriminator
    
    device = gpu_info['device']
    
    # Ajuster selon la VRAM
    if gpu_info['vram_gb'] < 8:
        ngf = ndf = 32
    else:
        ngf = ndf = 64
    
    logger.info(f"Création CycleGAN - ngf/ndf: {ngf}")
    
    gen_AB = Generator(ngf=ngf).to(device)
    gen_BA = Generator(ngf=ngf).to(device)
    disc_A = Discriminator(ndf=ndf).to(device)
    disc_B = Discriminator(ndf=ndf).to(device)
    
    return {
        'gen_AB': gen_AB,
        'gen_BA': gen_BA,
        'disc_A': disc_A,
        'disc_B': disc_B
    }

def setup_training(model_type, gpu_info, epochs, dataset_path):
    """Configure l'entraînement selon le modèle et le GPU"""
    from src.training.trainer import DCGANTrainer, CycleGANTrainer
    from src.training.callbacks import ModelCheckpoint, ImageSaver
    from src.data.dataset import get_dataloader
    
    # Configuration des callbacks
    callbacks = [
        ModelCheckpoint(
            save_dir='checkpoints',
            save_every=10,
            save_best=True
        ),
        ImageSaver(
            save_dir='generated_images',
            save_every=5,
            num_samples=16
        )
    ]
    
    # Créer le dataloader
    dataloader = get_dataloader(
        dataset_path=dataset_path,
        batch_size=gpu_info['batch_size'],
        image_size=128 if gpu_info['vram_gb'] >= 8 else 64,
        num_workers=gpu_info['num_workers']
    )
    
    if model_type == 'dcgan':
        model_dict = create_dcgan_model(gpu_info)
        trainer = DCGANTrainer(
            generator=model_dict['generator'],
            discriminator=model_dict['discriminator'],
            device=gpu_info['device'],
            callbacks=callbacks
        )
    elif model_type == 'cyclegan':
        model_dict = create_cyclegan_model(gpu_info)
        trainer = CycleGANTrainer(
            gen_AB=model_dict['gen_AB'],
            gen_BA=model_dict['gen_BA'],
            disc_A=model_dict['disc_A'],
            disc_B=model_dict['disc_B'],
            device=gpu_info['device'],
            callbacks=callbacks
        )
    else:
        raise ValueError(f"Modèle non supporté: {model_type}")
    
    return trainer, dataloader

def train_dcgan(epochs, dataset_path, gpu_info):
    """Entraîne un modèle DCGAN"""
    logger.info(f"Début entraînement DCGAN - {epochs} époques")
    
    trainer, dataloader = setup_training('dcgan', gpu_info, epochs, dataset_path)
    
    # Entraînement
    start_time = time.time()
    trainer.train(dataloader, epochs)
    end_time = time.time()
    
    logger.info(f"Entraînement terminé en {end_time - start_time:.2f} secondes")
    logger.info(f"Modèles sauvegardés dans: checkpoints/")
    logger.info(f"Images générées dans: generated_images/")

def train_cyclegan(epochs, dataset_path, gpu_info):
    """Entraîne un modèle CycleGAN"""
    logger.info(f"Début entraînement CycleGAN - {epochs} époques")
    
    trainer, dataloader = setup_training('cyclegan', gpu_info, epochs, dataset_path)
    
    # Entraînement
    start_time = time.time()
    trainer.train(dataloader, epochs)
    end_time = time.time()
    
    logger.info(f"Entraînement terminé en {end_time - start_time:.2f} secondes")
    logger.info(f"Modèles sauvegardés dans: checkpoints/")
    logger.info(f"Images générées dans: generated_images/")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Entraînement de modèles GAN')
    parser.add_argument('--model', type=str, choices=['dcgan', 'cyclegan'], 
                       required=True, help='Type de modèle à entraîner')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Nombre d\'époques (défaut: 100)')
    parser.add_argument('--dataset', type=str, default='data/dataset',
                       help='Chemin vers le dataset (défaut: data/dataset)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Taille du batch (auto si non spécifié)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("DÉMARRAGE ENTRAÎNEMENT GAN")
    logger.info("=" * 60)
    
    # Obtenir les informations GPU
    gpu_info = get_gpu_info()
    
    # Override batch_size si spécifié
    if args.batch_size:
        gpu_info['batch_size'] = args.batch_size
        logger.info(f"Batch size manuel: {args.batch_size}")
    
    # Analyser le dataset
    dataset_stats = analyze_dataset(args.dataset)
    if not dataset_stats:
        logger.error("Impossible d'analyser le dataset. Arrêt.")
        return
    
    # Créer les dossiers nécessaires
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Lancer l'entraînement
    try:
        if args.model == 'dcgan':
            train_dcgan(args.epochs, args.dataset, gpu_info)
        elif args.model == 'cyclegan':
            train_cyclegan(args.epochs, args.dataset, gpu_info)
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()