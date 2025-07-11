#!/usr/bin/env python3
"""
Script de test et diagnostic GPU pour l'entraînement de modèles GAN
Vérifie la configuration PyTorch + CUDA et les performances
"""

import sys
import time
import logging
import platform
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_python_environment():
    """Teste l'environnement Python"""
    logger.info("=== TEST ENVIRONNEMENT PYTHON ===")
    
    # Version Python
    version = sys.version_info
    logger.info(f"Version Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ requis")
        return False
    else:
        logger.info("✅ Version Python compatible")
    
    # Système d'exploitation
    logger.info(f"Système: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    
    return True

def test_pytorch_installation():
    """Teste l'installation de PyTorch"""
    logger.info("=== TEST INSTALLATION PYTORCH ===")
    
    try:
        import torch
        import torchvision
        import torchaudio
        
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ Torchvision version: {torchvision.__version__}")
        logger.info(f"✅ Torchaudio version: {torchaudio.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Erreur d'importation PyTorch: {e}")
        logger.error("Exécutez: python install_pytorch_cuda.py")
        return False

def test_cuda_availability():
    """Teste la disponibilité CUDA"""
    logger.info("=== TEST DISPONIBILITÉ CUDA ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info("✅ CUDA disponible")
            logger.info(f"✅ Version CUDA: {torch.version.cuda}")
            logger.info(f"✅ Version cuDNN: {torch.backends.cudnn.version()}")
            return True
        else:
            logger.warning("⚠️ CUDA non disponible")
            logger.warning("Vérifiez l'installation des drivers NVIDIA")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur test CUDA: {e}")
        return False

def test_gpu_detection():
    """Teste la détection du GPU"""
    logger.info("=== TEST DÉTECTION GPU ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ Aucun GPU CUDA détecté")
            return False
        
        # Informations GPU
        gpu_count = torch.cuda.device_count()
        logger.info(f"✅ Nombre de GPU: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  Multiprocessors: {props.multi_processor_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur détection GPU: {e}")
        return False

def test_gpu_memory():
    """Teste la mémoire GPU"""
    logger.info("=== TEST MÉMOIRE GPU ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ GPU non disponible pour test mémoire")
            return False
        
        device = torch.device('cuda:0')
        
        # Mémoire avant allocation
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(device)
        memory_total = torch.cuda.get_device_properties(device).total_memory
        
        logger.info(f"Mémoire totale: {memory_total / 1e9:.1f} GB")
        logger.info(f"Mémoire utilisée avant: {memory_before / 1e6:.1f} MB")
        
        # Test allocation mémoire
        test_tensor = torch.randn(1000, 1000, device=device)
        memory_after = torch.cuda.memory_allocated(device)
        
        logger.info(f"Mémoire utilisée après: {memory_after / 1e6:.1f} MB")
        logger.info(f"✅ Test allocation mémoire réussi")
        
        # Nettoyage
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test mémoire: {e}")
        return False

def test_gpu_performance():
    """Teste les performances GPU"""
    logger.info("=== TEST PERFORMANCES GPU ===")
    
    try:
        import torch
        import torch.nn as nn
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ GPU non disponible pour test performance")
            return False
        
        device = torch.device('cuda:0')
        
        # Test calcul matriciel
        logger.info("Test calcul matriciel...")
        
        # CPU
        start_time = time.time()
        a_cpu = torch.randn(1000, 1000)
        b_cpu = torch.randn(1000, 1000)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # GPU
        start_time = time.time()
        a_gpu = torch.randn(1000, 1000, device=device)
        b_gpu = torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()  # Attendre la fin des opérations
        
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logger.info(f"Temps CPU: {cpu_time:.4f}s")
        logger.info(f"Temps GPU: {gpu_time:.4f}s")
        logger.info(f"Accélération: {speedup:.1f}x")
        
        if speedup > 5:
            logger.info("✅ Excellentes performances GPU")
        elif speedup > 2:
            logger.info("✅ Bonnes performances GPU")
        else:
            logger.warning("⚠️ Performances GPU limitées")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test performance: {e}")
        return False

def test_mixed_precision():
    """Teste le support Mixed Precision"""
    logger.info("=== TEST MIXED PRECISION ===")
    
    try:
        import torch
        from torch.cuda.amp import autocast, GradScaler
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ GPU non disponible pour test Mixed Precision")
            return False
        
        device = torch.device('cuda:0')
        
        # Test autocast
        with autocast():
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
        
        # Test GradScaler
        scaler = GradScaler()
        
        logger.info("✅ Mixed Precision supporté")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test Mixed Precision: {e}")
        return False

def test_gan_compatibility():
    """Teste la compatibilité pour l'entraînement GAN"""
    logger.info("=== TEST COMPATIBILITÉ GAN ===")
    
    try:
        import torch
        import torch.nn as nn
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ GPU non disponible pour test GAN")
            return False
        
        device = torch.device('cuda:0')
        
        # Test création modèle simple
        class SimpleGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(100, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 3*64*64),
                    nn.Tanh()
                )
            
            def forward(self, x):
                return self.net(x).view(-1, 3, 64, 64)
        
        # Test sur GPU
        model = SimpleGenerator().to(device)
        noise = torch.randn(16, 100, device=device)
        
        with torch.no_grad():
            fake_images = model(noise)
        
        logger.info(f"✅ Test modèle GAN réussi")
        logger.info(f"Forme sortie: {fake_images.shape}")
        
        # Recommandations batch size
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        if vram_gb < 8:
            recommended_batch = 12
        elif vram_gb < 12:
            recommended_batch = 16
        elif vram_gb < 16:
            recommended_batch = 20
        else:
            recommended_batch = 24
        
        logger.info(f"Batch size recommandé: {recommended_batch}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test GAN: {e}")
        return False

def generate_report(results):
    """Génère un rapport de diagnostic"""
    logger.info("=== RAPPORT DE DIAGNOSTIC ===")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    logger.info(f"Tests réussis: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("🎉 TOUS LES TESTS RÉUSSIS")
        logger.info("Votre configuration est prête pour l'entraînement GAN !")
        logger.info("")
        logger.info("Commandes recommandées:")
        logger.info("  python train.py --model dcgan --epochs 10")
        logger.info("  python train.py --model cyclegan --epochs 50")
    elif passed_tests >= total_tests * 0.8:
        logger.info("✅ Configuration majoritairement fonctionnelle")
        logger.info("Quelques optimisations possibles")
    else:
        logger.warning("⚠️ Configuration nécessite des corrections")
        logger.info("Consultez le GPU_SETUP_GUIDE.md")

def main():
    """Fonction principale de test"""
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC GPU POUR ENTRAÎNEMENT GAN")
    logger.info("=" * 60)
    
    # Exécution des tests
    results = {
        'python_env': test_python_environment(),
        'pytorch_install': test_pytorch_installation(),
        'cuda_available': test_cuda_availability(),
        'gpu_detection': test_gpu_detection(),
        'gpu_memory': test_gpu_memory(),
        'gpu_performance': test_gpu_performance(),
        'mixed_precision': test_mixed_precision(),
        'gan_compatibility': test_gan_compatibility()
    }
    
    # Génération du rapport
    generate_report(results)
    
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC TERMINÉ")
    logger.info("=" * 60)
    
    # Code de sortie
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()