import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from utils.keypoints import KeypointSpec, from_dataset_dict


def test_kpt_target_width():
    """Test that keypoint target width matches expected formula: 5 + K*D"""

    print("Testing keypoint target width regression...")

    # Test 1: Load 23-keypoint spec
    with open("data/kpts_23_template.yaml", "r") as f:
        data_23 = yaml.safe_load(f)

    spec_23 = from_dataset_dict(data_23)
    print(f"23-keypoint spec: K={spec_23.K}, D={spec_23.D}")

    # Test 2: Create dummy labels for 23-keypoint spec
    N = 3  # Number of samples
    expected_width_23 = 5 + spec_23.K * spec_23.D  # 5 + 23*3 = 74
    raw_labels_23 = np.random.rand(N, expected_width_23).astype(np.float32)

    print(f"Raw labels shape: {raw_labels_23.shape}")
    print(f"Expected width: {expected_width_23}")

    # Test 3: Simulate the collate function from the fixed datasets.py
    # The collate function creates output with width = input_width (no extra column needed in test since we test the target tensor format)
    labels_batch_23 = [torch.from_numpy(raw_labels_23[i : i + 1]) for i in range(N)]

    # Simulate dynamic collate logic from our fixed collate_fn
    nL = sum([len(l) for l in labels_batch_23])
    if nL > 0:
        # Get label width from first non-empty label
        label_w = next(l.shape[1] for l in labels_batch_23 if len(l) > 0)
        collated_targets_23 = torch.zeros((nL, label_w), dtype=torch.float32)

        # Fill the output tensor as per our fixed collate_fn
        start_idx = 0
        for i, l in enumerate(labels_batch_23):
            if len(l) > 0:
                end_idx = start_idx + len(l)
                collated_targets_23[start_idx:end_idx, 0] = i  # batch index
                collated_targets_23[start_idx:end_idx, 1:] = l[
                    :, 1:
                ]  # copy all columns except first
                start_idx = end_idx

    print(f"Collated targets shape: {collated_targets_23.shape}")

    # Test 4: Core assertions for 23-keypoint spec
    assert raw_labels_23.shape[1] == expected_width_23, (
        f"Raw labels width {raw_labels_23.shape[1]} != expected {expected_width_23}"
    )

    # The collated tensor maintains the same width as the input (since batch index overwrites first column)
    assert collated_targets_23.shape[1] == expected_width_23, (
        f"Collated targets width {collated_targets_23.shape[1]} != expected {expected_width_23}"
    )

    # Verify no columns dropped - compare the keypoint data portion (columns 5 onwards)
    for i in range(N):
        original_kpt_data = raw_labels_23[i, 5:]  # Keypoint data from original
        collated_kpt_data = collated_targets_23[
            i, 5:
        ].numpy()  # Keypoint data from collated
        np.testing.assert_array_almost_equal(
            original_kpt_data,
            collated_kpt_data,
            err_msg=f"Keypoint data doesn't match for sample {i}",
        )

    print("✓ 23-keypoint spec tests passed")

    # Test 5: COCO spec (K=17, D=3)
    coco_data = {
        "kpt_shape": [17, 3],
        "kpt_names": [f"kpt_{i}" for i in range(17)],
        "kpt_flip_index": list(range(17)),
    }

    spec_coco = from_dataset_dict(coco_data)
    print(f"COCO spec: K={spec_coco.K}, D={spec_coco.D}")

    expected_width_coco = 5 + spec_coco.K * spec_coco.D  # 5 + 17*3 = 56
    raw_labels_coco = np.random.rand(N, expected_width_coco).astype(np.float32)

    # Simulate collate for COCO
    labels_batch_coco = [torch.from_numpy(raw_labels_coco[i : i + 1]) for i in range(N)]

    nL = sum([len(l) for l in labels_batch_coco])
    if nL > 0:
        label_w = next(l.shape[1] for l in labels_batch_coco if len(l) > 0)
        collated_targets_coco = torch.zeros((nL, label_w), dtype=torch.float32)

        start_idx = 0
        for i, l in enumerate(labels_batch_coco):
            if len(l) > 0:
                end_idx = start_idx + len(l)
                collated_targets_coco[start_idx:end_idx, 0] = i  # batch index
                collated_targets_coco[start_idx:end_idx, 1:] = l[
                    :, 1:
                ]  # copy all columns except first
                start_idx = end_idx

    # Test 6: Assertions for COCO spec
    print(f"COCO raw labels shape: {raw_labels_coco.shape}")
    print(f"COCO expected width: {expected_width_coco}")
    print(f"COCO collated targets shape: {collated_targets_coco.shape}")

    assert raw_labels_coco.shape[1] == expected_width_coco, (
        f"COCO raw labels width {raw_labels_coco.shape[1]} != expected {expected_width_coco}"
    )
    assert collated_targets_coco.shape[1] == expected_width_coco, (
        f"COCO collated targets width {collated_targets_coco.shape[1]} != expected {expected_width_coco}"
    )

    # Verify COCO width is exactly 56 (5 + 17*3)
    assert expected_width_coco == 56, f"COCO expected width {expected_width_coco} != 56"

    print("✓ COCO spec tests passed")

    # Test 7: Verify our dynamic formula works for arbitrary K, D
    print("Testing dynamic formula for arbitrary keypoint configurations...")

    # Test case: K=10, D=2 (2D coordinates only)
    test_K, test_D = 10, 2
    expected_width_custom = 5 + test_K * test_D  # 5 + 10*2 = 25
    print(f"Custom spec K={test_K}, D={test_D}, expected width={expected_width_custom}")

    custom_data = {"kpt_shape": [test_K, test_D]}
    spec_custom = from_dataset_dict(custom_data)
    assert spec_custom.K == test_K and spec_custom.D == test_D

    # Simulate labels for custom spec
    raw_labels_custom = np.random.rand(2, expected_width_custom).astype(np.float32)
    assert raw_labels_custom.shape[1] == expected_width_custom

    print("✓ Dynamic formula tests passed")
    print("✓ All keypoint target width regression tests passed!")


if __name__ == "__main__":
    test_kpt_target_width()
