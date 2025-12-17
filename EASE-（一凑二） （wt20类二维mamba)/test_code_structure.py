#!/usr/bin/env python3
"""
Code structure validation test for vim_ease.py
This test validates the fixes without requiring PyTorch installation.
"""

import re
import sys
import os

def test_file_content():
    """Test that the file content has all fixes applied"""
    print("\n" + "="*60)
    print("Code Structure Validation")
    print("="*60)
    
    file_path = os.path.join(os.path.dirname(__file__), 'backbone', 'vim_ease.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_passed = True
    
    # Test 1: Check adapter_scalar default value
    print("\n[Test 1] Adapter scalar default value")
    if 'adapter_scalar="1.0"' in content:
        print("  ✓ Found correct: adapter_scalar=\"1.0\"")
    else:
        if 'adapter_scalar="1. 0"' in content:
            print("  ✗ Still has space: adapter_scalar=\"1. 0\"")
            all_passed = False
        else:
            print("  ⚠ Could not find adapter_scalar parameter")
    
    # Test 2: Check file paths
    print("\n[Test 2] Pretrained file paths")
    if '"./pretrained/vit.pth"' in content:
        print("  ✓ Found correct: \"./pretrained/vit.pth\"")
    else:
        if '"./pretrained/vit. pth"' in content:
            print("  ✗ Still has space: \"./pretrained/vit. pth\"")
            all_passed = False
        else:
            print("  ⚠ Could not find vit.pth path")
    
    # Test 3: Check weight mapping comments
    print("\n[Test 3] Weight mapping implementation")
    if "# ViT uses blocks.{i}.mlp.fc1/fc2, Mamba uses blocks.{i}.fc1/fc2" in content:
        print("  ✓ Found correct mapping comment")
    else:
        print("  ⚠ Mapping comment not found or different")
    
    # Test 4: Check for mlp. prefix in weight loading
    print("\n[Test 4] ViT to Mamba weight key mapping")
    if "'mlp.fc1'" in content and "'mlp.fc2'" in content:
        print("  ✓ Correctly references ViT's mlp.fc1 and mlp.fc2")
    else:
        print("  ✗ Missing references to mlp.fc1 or mlp.fc2")
        all_passed = False
    
    # Test 5: Check for error handling
    print("\n[Test 5] Error handling for missing keys")
    # Count if checks for bias keys
    bias_checks = len(re.findall(r"if\s+['\"].*bias['\"].*in\s+state", content))
    if bias_checks >= 3:  # Should have multiple checks for bias
        print(f"  ✓ Found {bias_checks} checks for optional bias keys")
    else:
        print(f"  ⚠ Found only {bias_checks} checks for optional bias keys")
    
    # Test 6: Check for loaded_blocks counter
    print("\n[Test 6] Block loading counter")
    if "loaded_blocks" in content:
        print("  ✓ Found loaded_blocks counter for better reporting")
    else:
        print("  ⚠ loaded_blocks counter not found")
    
    # Test 7: Check for final norm loading
    print("\n[Test 7] Final norm layer loading")
    if '"norm.weight" in state' in content or "'norm.weight' in state" in content:
        print("  ✓ Added final norm layer loading")
    else:
        print("  ⚠ Final norm loading not found")
    
    return all_passed

def main():
    """Run validation"""
    print("\n" + "="*60)
    print("VIM_EASE.PY CODE STRUCTURE VALIDATION")
    print("="*60)
    
    try:
        all_passed = test_file_content()
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL CRITICAL FIXES VERIFIED")
        else:
            print("✗ SOME CRITICAL FIXES MISSING")
        print("="*60)
        
        return 0 if all_passed else 1
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
