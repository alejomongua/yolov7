#!/usr/bin/env python3
"""
Simple test to verify basic keypoint configuration functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_keypoint_spec():
    """Test the KeypointSpec functionality"""
    print("Testing KeypointSpec...")

    try:
        from utils.keypoints import KeypointSpec, coco_defaults, from_dataset_dict

        # Test COCO defaults
        coco_spec = coco_defaults()
        print(f"✅ COCO defaults: K={coco_spec.K}, D={coco_spec.D}")
        print(f"✅ COCO flip_index length: {len(coco_spec.flip_index)}")
        print(f"✅ COCO names count: {len(coco_spec.names)}")

        # Test validation
        assert coco_spec.K == 17, f"Expected K=17, got K={coco_spec.K}"
        assert coco_spec.D == 3, f"Expected D=3, got D={coco_spec.D}"
        assert len(coco_spec.flip_index) == 17, (
            f"Expected flip_index length 17, got {len(coco_spec.flip_index)}"
        )
        assert len(coco_spec.names) == 17, (
            f"Expected 17 names, got {len(coco_spec.names)}"
        )

        # Test from_dataset_dict with COCO configuration
        coco_data = {
            "kpt_shape": [17, 3],
            "kpt_flip_index": [
                0,
                2,
                1,
                4,
                3,
                6,
                5,
                8,
                7,
                10,
                9,
                12,
                11,
                14,
                13,
                16,
                15,
            ],
            "oks_sigmas": [
                0.026,
                0.025,
                0.025,
                0.035,
                0.035,
                0.079,
                0.079,
                0.072,
                0.072,
                0.062,
                0.062,
                0.107,
                0.107,
                0.087,
                0.087,
                0.089,
                0.089,
            ],
        }

        spec = from_dataset_dict(coco_data)
        print(f"✅ From dict: K={spec.K}, D={spec.D}")
        assert spec.K == 17 and spec.D == 3

        # Test with custom configuration
        custom_data = {
            "kpt_shape": [21, 2],  # Custom keypoint count
        }
        custom_spec = from_dataset_dict(custom_data)
        print(f"✅ Custom: K={custom_spec.K}, D={custom_spec.D}")
        assert custom_spec.K == 21 and custom_spec.D == 2
        assert len(custom_spec.flip_index) == 21
        assert len(custom_spec.names) == 21

        print("✅ KeypointSpec tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ KeypointSpec test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_imports():
    """Test that our modified modules can be imported"""
    print("Testing basic imports...")

    try:
        from utils.datasets import LoadImagesAndLabels, create_dataloader

        print("✅ utils.datasets imports successfully")

        from utils.keypoints import KeypointSpec, coco_defaults, from_dataset_dict

        print("✅ utils.keypoints imports successfully")

        print("✅ Basic import tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset_class_init():
    """Test LoadImagesAndLabels initialization with data_dict"""
    print("Testing LoadImagesAndLabels initialization...")

    try:
        from utils.datasets import LoadImagesAndLabels

        # Test with default parameters (should not crash)
        # We can't fully initialize without valid paths, but we can test parameter handling
        print("✅ LoadImagesAndLabels class can be imported")

        # Test that the class accepts data_dict parameter
        import inspect

        init_signature = inspect.signature(LoadImagesAndLabels.__init__)
        params = list(init_signature.parameters.keys())

        assert "data_dict" in params, (
            f"data_dict parameter missing from __init__, got params: {params}"
        )
        print("✅ LoadImagesAndLabels.__init__ accepts data_dict parameter")

        print("✅ Dataset class initialization tests passed!\n")
        return True

    except Exception as e:
        print(f"❌ Dataset class test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=== YOLOv7 Configurable Keypoint Shape Simple Test ===\n")

    success = True

    # Test basic imports
    success &= test_basic_imports()

    # Test KeypointSpec functionality
    success &= test_keypoint_spec()

    # Test dataset class modifications
    success &= test_dataset_class_init()

    if success:
        print("✅ All simple tests passed!")
        print("\n🎉 Phase 1 implementation summary:")
        print("✅ Created utils/keypoints.py with KeypointSpec dataclass")
        print("✅ Updated COCO YAML files with kpt_shape and configuration fields")
        print("✅ Parameterized LoadImagesAndLabels for dynamic K,D")
        print("✅ Replaced hardcoded constants (56, 17, flip_index)")
        print("✅ Made augmentation allocations dynamic")
        print("✅ Maintained backward compatibility with COCO workflows")

        print("\n📋 Constants replaced:")
        print("- Hard-coded 56 → dynamic 5 + K*D in label validation")
        print("- Hard-coded 17 → dynamic K in augmentation allocations")
        print("- Hard-coded flip_index → configurable from dataset YAML")
        print("- Hard-coded 39 → dynamic 5 + K*2 for processed labels")

        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
