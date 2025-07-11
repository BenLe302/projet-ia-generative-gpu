#!/usr/bin/env python3
"""
Script d'installation automatique de PyTorch avec support CUDA
Optimisé pour RTX 3060, 4060 Ti et autres GPU NVIDIA
"""

import subprocess
import sys
import platform
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Exécute une commande et gère les erreurs"""
    logger.info(f"Exécution: {description}")
    logger.info(f"Commande: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Succès: {description}")
        if result.stdout:
            logger.info(f"Sortie: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de: {description}")
        logger.error(f"Code d'erreur: {e.returncode}")
        if e.stdout:
            logger.error(f"Sortie: {e.stdout.strip()}")
        if e.stderr:
            logger.error(f"Erreur: {e.stderr.strip()}")
        return False

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    logger.info(f"Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ requis")
        return False
    
    logger.info("✅ Version Python compatible")
    return True

def check_system():
    """Vérifie le système d'exploitation"""
    system = platform.system()
    logger.info(f"Système: {system}")
    
    if system not in ['Windows', 'Linux', 'Darwin']:
        logger.warning(f"Système non testé: {system}")
    
    return True

def uninstall_existing_pytorch():
    """Désinstalle les versions existantes de PyTorch"""
    logger.info("Désinstallation des versions existantes de PyTorch...")
    
    packages_to_remove = [
        'torch',
        'torchvision', 
        'torchaudio',
        'torchtext',
        'torchdata'
    ]
    
    for package in packages_to_remove:
        command = f"{sys.executable} -m pip uninstall {package} -y"
        run_command(command, f"Désinstallation de {package}")
    
    logger.info("✅ Nettoyage terminé")

def install_pytorch_cuda121():
    """Installe PyTorch avec CUDA 12.1 (recommandé)"""
    logger.info("Installation de PyTorch avec CUDA 12.1...")
    
    command = (
        f"{sys.executable} -m pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    
    success = run_command(command, "Installation PyTorch CUDA 12.1")
    
    if success:
        logger.info("✅ PyTorch CUDA 12.1 installé avec succès")
    else:
        logger.error("❌ Échec installation PyTorch CUDA 12.1")
        logger.info("Tentative avec CUDA 11.8...")
        return install_pytorch_cuda118()
    
    return success

def install_pytorch_cuda118():
    """Installe PyTorch avec CUDA 11.8 (alternative)"""
    logger.info("Installation de PyTorch avec CUDA 11.8...")
    
    command = (
        f"{sys.executable} -m pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    success = run_command(command, "Installation PyTorch CUDA 11.8")
    
    if success:
        logger.info("✅ PyTorch CUDA 11.8 installé avec succès")
    else:
        logger.error("❌ Échec installation PyTorch CUDA 11.8")
        logger.info("Tentative avec version CPU...")
        return install_pytorch_cpu()
    
    return success

def install_pytorch_cpu():
    """Installe PyTorch version CPU (fallback)"""
    logger.info("Installation de PyTorch version CPU...")
    
    command = f"{sys.executable} -m pip install torch torchvision torchaudio"
    
    success = run_command(command, "Installation PyTorch CPU")
    
    if success:
        logger.info("✅ PyTorch CPU installé avec succès")
        logger.warning("⚠️ Version CPU installée - performances limitées")
    else:
        logger.error("❌ Échec installation PyTorch CPU")
    
    return success

def install_additional_packages():
    """Installe les packages additionnels requis"""
    logger.info("Installation des packages additionnels...")
    
    packages = [
        'numpy',
        'pillow',
        'matplotlib',
        'tqdm',
        'tensorboard',
        'opencv-python',
        'scikit-learn',
        'pandas'
    ]
    
    for package in packages:
        command = f"{sys.executable} -m pip install {package}"
        run_command(command, f"Installation de {package}")
    
    logger.info("✅ Packages additionnels installés")

def verify_installation():
    """Vérifie l'installation de PyTorch"""
    logger.info("Vérification de l'installation...")
    
    try:
        import torch
        import torchvision
        import torchaudio
        
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Torchvision version: {torchvision.__version__}")
        logger.info(f"✅ Torchaudio version: {torchaudio.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA disponible: {torch.version.cuda}")
            logger.info(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ VRAM: {vram:.1f} GB")
        else:
            logger.warning("⚠️ CUDA non disponible - utilisation CPU")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erreur d'importation: {e}")
        return False

def main():
    """Fonction principale d'installation"""
    logger.info("=" * 60)
    logger.info("INSTALLATION AUTOMATIQUE PYTORCH + CUDA")
    logger.info("=" * 60)
    
    # Vérifications préliminaires
    if not check_python_version():
        sys.exit(1)
    
    if not check_system():
        sys.exit(1)
    
    # Mise à jour pip
    logger.info("Mise à jour de pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Mise à jour pip")
    
    # Désinstallation des versions existantes
    uninstall_existing_pytorch()
    
    # Installation PyTorch
    success = install_pytorch_cuda121()
    
    if not success:
        logger.error("❌ Échec de l'installation PyTorch")
        sys.exit(1)
    
    # Installation packages additionnels
    install_additional_packages()
    
    # Vérification finale
    if verify_installation():
        logger.info("=" * 60)
        logger.info("✅ INSTALLATION TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info("Vous pouvez maintenant exécuter:")
        logger.info("  python test_gpu.py")
        logger.info("  python train.py --model dcgan --epochs 10")
    else:
        logger.error("❌ Problème lors de la vérification")
        sys.exit(1)

if __name__ == '__main__':
    main()