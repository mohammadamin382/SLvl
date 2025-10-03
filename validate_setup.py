#!/usr/bin/env python3
"""
Setup Validation Script
Validates that the environment is correctly configured
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    required = {
        'torch': 'PyTorch',
        'chess': 'python-chess',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("All dependencies installed!\n")
    return True


def check_torch_cuda():
    """Check PyTorch CUDA availability"""
    print("Checking PyTorch CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print(f"  ⚠ CUDA not available - will use CPU")

    print(f"  ✓ PyTorch version: {torch.__version__}\n")
    return True


def check_gpu_modules():
    """Check GPU optimization modules"""
    print("Checking GPU modules...")

    try:
        from gpus import get_gpu_optimizer, detect_gpu

        gpu_type = detect_gpu()
        print(f"  ✓ GPU detection: {gpu_type or 'CPU/Generic'}")

        optimizer = get_gpu_optimizer()
        print(f"  ✓ GPU optimizer: {optimizer.name}")

        if hasattr(optimizer, 'memory_gb'):
            print(f"  ✓ Detected memory: {optimizer.memory_gb:.2f} GB")

    except Exception as e:
        print(f"  ✗ Error loading GPU modules: {e}")
        return False

    print()
    return True


def check_model():
    """Check model can be instantiated"""
    print("Checking model...")

    try:
        from src.models import ChessTransformer

        model = ChessTransformer(dim=256, depth=4, num_heads=4)
        params = model.count_parameters()
        print(f"  ✓ Model instantiated")
        print(f"  ✓ Test model parameters: {params:,}")

    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        return False

    print()
    return True


def check_data_loader():
    """Check data loader modules"""
    print("Checking data loader...")

    try:
        from src.data import PGNDataset, GameFilter, MoveConverter

        move_converter = MoveConverter()
        print(f"  ✓ Move converter: {len(move_converter.move_to_idx)} moves")

        filter = GameFilter(min_elo=2000, min_moves=10)
        print(f"  ✓ Game filter created")

    except Exception as e:
        print(f"  ✗ Error loading data modules: {e}")
        return False

    print()
    return True


def check_config_files():
    """Check configuration files exist"""
    print("Checking configuration files...")

    configs = [
        'settings.yaml',
        'data_settings_format.yaml'
    ]

    all_exist = True
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} - MISSING")
            all_exist = False

    print()
    return all_exist


def check_directories():
    """Check required directories"""
    print("Checking directories...")

    dirs = {
        'data/pgn': 'PGN data directory',
        'checkpoints': 'Checkpoints (will be created)',
        'logs': 'Logs (will be created)',
    }

    for dir_path, desc in dirs.items():
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path} - {desc}")
        else:
            print(f"  ⚠ {dir_path} - {desc} (will be created)")

    print()
    return True


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Chess Supervised Learning - Setup Validation")
    print("=" * 60)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("PyTorch CUDA", check_torch_cuda),
        ("GPU Modules", check_gpu_modules),
        ("Model Architecture", check_model),
        ("Data Loader", check_data_loader),
        ("Configuration Files", check_config_files),
        ("Directories", check_directories),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error in {name}: {e}\n")
            results.append((name, False))

    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")

    all_passed = all(result for _, result in results)

    print()
    if all_passed:
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("1. Place PGN files in data/pgn/")
        print("2. Run dry-run: python train.py --dry-run")
        print("3. Start training: python train.py")
    else:
        print("✗ Some checks failed. Please resolve issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
