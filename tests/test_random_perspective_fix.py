#!/usr/bin/env python3
"""
Test the fixed random_perspective function with dynamic K,D keypoints
specifically for the 23x3 keypoint case to verify the reshape fix.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils.datasets import random_perspective


def test_random_perspective_23x3():
    """Test random_perspective function with 23x3 keypoints"""

    print("Testing random_perspective function with 23x3 keypoints...")

    # Create dummy image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Create targets with 23x3 keypoints
    # Format: [class, x, y, w, h, kpt1_x, kpt1_y, kpt1_v, ..., kpt23_x, kpt23_y, kpt23_v]
    K = 23  # Number of keypoints
    D = 3  # Dimensions per keypoint (x, y, visibility)
    expected_width = 5 + K * D  # 5 + 23*3 = 74

    print(f"Testing with K={K}, D={D}, expected_width={expected_width}")

    # Create 2 target objects for testing
    N = 2
    targets = np.random.rand(N, expected_width).astype(np.float32)

    # Set reasonable bounding box values (columns 1-4: x, y, w, h)
    targets[:, 1:5] = np.array(
        [[100, 100, 200, 200], [300, 300, 150, 150]]
    )  # pixel coordinates

    # Set keypoint coordinates in pixel space (columns 5 onwards)
    # For each keypoint: x, y, visibility
    for i in range(K):
        kpt_idx = 5 + i * D
        targets[:, kpt_idx] = np.random.uniform(50, 550, N)  # x coordinates
        targets[:, kpt_idx + 1] = np.random.uniform(50, 550, N)  # y coordinates
        targets[:, kpt_idx + 2] = np.random.choice(
            [0, 1, 2], N
        )  # visibility (0=not visible, 1=visible, 2=occluded)

    print(f"Original targets shape: {targets.shape}")

    # Test the random_perspective function
    try:
        transformed_img, transformed_targets = random_perspective(
            img,
            targets,
            segments=(),
            degrees=10,
            translate=0.1,
            scale=0.1,
            shear=10,
            perspective=0.0,
            border=(0, 0),
            kpt_label=True,
        )

        print(f"✓ random_perspective succeeded!")
        print(f"Transformed targets shape: {transformed_targets.shape}")

        # Verify the shape is preserved
        assert transformed_targets.shape == targets.shape, (
            f"Shape mismatch: {transformed_targets.shape} != {targets.shape}"
        )

        print(f"✓ Shape preservation test passed")

        # Test with different keypoint configurations
        test_configs = [
            (17, 3),  # COCO
            (17, 2),  # COCO without visibility
            (13, 3),  # Custom with visibility
            (10, 2),  # Custom without visibility
        ]

        for test_K, test_D in test_configs:
            test_width = 5 + test_K * test_D
            test_targets = np.random.rand(1, test_width).astype(np.float32)
            test_targets[:, 1:5] = np.array([[150, 150, 100, 100]])  # bbox

            # Set keypoint coordinates
            for i in range(test_K):
                kpt_idx = 5 + i * test_D
                test_targets[:, kpt_idx] = np.random.uniform(100, 250, 1)  # x
                test_targets[:, kpt_idx + 1] = np.random.uniform(100, 250, 1)  # y
                if test_D >= 3:
                    test_targets[:, kpt_idx + 2] = np.random.choice(
                        [0, 1, 2], 1
                    )  # visibility

            try:
                _, test_transformed = random_perspective(
                    img,
                    test_targets,
                    kpt_label=True,
                )
                print(f"✓ K={test_K}, D={test_D} configuration passed")
            except Exception as e:
                print(f"✗ K={test_K}, D={test_D} configuration failed: {e}")
                raise

        print("✓ All keypoint configurations test passed")

    except Exception as e:
        print(f"✗ random_perspective failed: {e}")
        raise

    print("✓ All random_perspective tests passed!")


def test_dynamic_kd_detection():
    """Test the dynamic K,D detection logic"""

    print("Testing dynamic K,D detection logic...")

    # Test cases: (label_width, expected_D, expected_K)
    test_cases = [
        (74, 3, 23),  # 23x3 keypoints: 5 + 23*3 = 74
        (56, 3, 17),  # COCO 17x3: 5 + 17*3 = 56
        (39, 2, 17),  # COCO 17x2: 5 + 17*2 = 39
        (25, 2, 10),  # 10x2: 5 + 10*2 = 25
        (35, 3, 10),  # 10x3: 5 + 10*3 = 35
    ]

    img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)

    for label_w, expected_D, expected_K in test_cases:
        targets = np.random.rand(1, label_w).astype(np.float32)
        targets[:, 1:5] = np.array([[50, 50, 100, 100]])  # bbox

        # Set keypoint data
        for i in range(expected_K):
            kpt_idx = 5 + i * expected_D
            targets[:, kpt_idx] = np.random.uniform(25, 175, 1)  # x
            targets[:, kpt_idx + 1] = np.random.uniform(25, 175, 1)  # y
            if expected_D >= 3:
                targets[:, kpt_idx + 2] = np.random.choice([0, 1, 2], 1)  # visibility

        try:
            _, result_targets = random_perspective(
                img,
                targets,
                kpt_label=True,
            )

            # Verify shape is preserved
            assert result_targets.shape[1] == label_w, (
                f"Width mismatch for case (w={label_w}, D={expected_D}, K={expected_K}): "
                f"got {result_targets.shape[1]}, expected {label_w}"
            )

            print(f"✓ Case w={label_w}, D={expected_D}, K={expected_K} passed")

        except Exception as e:
            print(f"✗ Case w={label_w}, D={expected_D}, K={expected_K} failed: {e}")
            raise

    print("✓ Dynamic K,D detection tests passed!")


if __name__ == "__main__":
    test_random_perspective_23x3()
    test_dynamic_kd_detection()
    print("\n🎉 All tests passed! The random_perspective fix is working correctly.")
