#!/usr/bin/env python3
"""
DreamJourney Environment Test Script
验证DreamJourney项目所需环境是否安装成功
"""

import sys

def test_torch():
    """测试PyTorch安装"""
    print("1. Testing PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA version: {torch.version.cuda}")
            print(f"   ✓ GPU count: {torch.cuda.device_count()}")
        
        # 简单功能测试
        x = torch.randn(2, 3)
        y = torch.mm(x, x.t())
        print("   ✓ Basic tensor operations work")
        return True
        
    except ImportError as e:
        print(f"   ✗ PyTorch import failed: {e}")
        print("   → Please install PyTorch: pip install torch")
        return False
    except Exception as e:
        print(f"   ✗ PyTorch functionality test failed: {e}")
        return False

def test_transformers():
    """测试transformers库"""
    print("\n2. Testing transformers...")
    try:
        import transformers
        print(f"   ✓ Transformers version: {transformers.__version__}")
        
        
        return True
        
    except ImportError as e:
        print(f"   ✗ Transformers import failed: {e}")
        print("   → Please install transformers: pip install transformers")
        return False
    except Exception as e:
        print(f"   ✗ Transformers test failed: {e}")
        return False

def test_pytorch3d():
    """测试PyTorch3D安装"""
    print("\n3. Testing PyTorch3D...")
    try:
        import pytorch3d
        print(f"   ✓ PyTorch3D version: {pytorch3d.__version__}")
        
        # 测试核心组件
        from pytorch3d.structures import Meshes
        from pytorch3d.transforms import Rotate
        
        import torch
        verts = torch.randn(1, 3, 3)
        faces = torch.tensor([[[0, 1, 2]]])
        mesh = Meshes(verts=verts, faces=faces)
        print(f"   ✓ Created test mesh with {mesh.num_verts_per_mesh().item()} vertices")
        
        return True
        
    except ImportError as e:
        print(f"   ✗ PyTorch3D import failed: {e}")
        print("   → Please install PyTorch3D following the setup instructions")
        return False
    except Exception as e:
        print(f"   ✗ PyTorch3D test failed: {e}")
        return False

def test_other_dependencies():
    """测试其他重要依赖"""
    print("\n4. Testing other dependencies...")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('yaml', 'PyYAML'),
        ('spacy', 'spacy'),
    ]
    
    all_success = True
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"   ✓ {module_name} (from {package_name})")
        except ImportError:
            print(f"   ✗ {module_name} (from {package_name}) not found")
            print(f"   → Please install: pip install {package_name}")
            all_success = False
    
    # 特殊检查spacy模型
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("   ✓ spacy English model (en_core_web_sm)")
    except OSError:
        print("   ✗ spacy English model not found")
        print("   → Please download: python -m spacy download en_core_web_sm")
        all_success = False
    except ImportError:
        pass  # spacy already reported as missing above
    
    return all_success

def main():
    """主函数"""
    print("=" * 60)
    print("DreamJourney Environment Test")
    print("=" * 60)
    
    results = []
    results.append(test_torch())
    results.append(test_transformers())
    results.append(test_pytorch3d())
    results.append(test_other_dependencies())
    
    print("\n" + "=" * 60)
    
    if all(results):
        print("✓ ALL TESTS PASSED! Environment is ready for DreamJourney.")
        print("You can now run the project successfully!")
    else:
        print("✗ SOME TESTS FAILED! Please fix the issues above.")
        print("Make sure you're in the correct conda environment:")
        print("  conda activate dreamJourney")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main() 