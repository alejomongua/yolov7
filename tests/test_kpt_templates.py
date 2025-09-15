"""
Test suite for keypoint dataset templates and KeypointSpec parsing.

Validates that dataset YAML templates can be properly parsed and that
KeypointSpec objects are created with correct properties and invariants.
"""

import yaml
import pytest
from utils.keypoints import from_dataset_dict


def test_23_keypoint_template_parsing():
    """Test that the 23-keypoint template can be parsed correctly."""
    # Load the 23-keypoint template YAML
    with open("data/kpts_23_template.yaml", "r") as f:
        dataset_dict = yaml.safe_load(f)

    # Parse into KeypointSpec
    spec = from_dataset_dict(dataset_dict)

    # Validate basic properties
    assert spec.K == 23, f"Expected K=23, got K={spec.K}"
    assert spec.D == 3, f"Expected D=3, got D={spec.D}"

    # Validate names
    assert len(spec.names) == 23, f"Expected 23 names, got {len(spec.names)}"
    expected_names = [f"k{i}" for i in range(23)]
    assert spec.names == expected_names, (
        f"Names mismatch: expected {expected_names}, got {spec.names}"
    )

    # Validate flip_index is identity permutation
    assert len(spec.flip_index) == 23, (
        f"Expected flip_index length 23, got {len(spec.flip_index)}"
    )
    expected_flip = list(range(23))
    assert spec.flip_index == expected_flip, (
        f"Expected identity flip_index {expected_flip}, got {spec.flip_index}"
    )

    # Validate flip_index is a permutation of range(K)
    flip_set = set(spec.flip_index)
    expected_set = set(range(23))
    assert flip_set == expected_set, (
        f"flip_index is not a permutation of range(23): {flip_set} != {expected_set}"
    )

    # Validate oks_sigmas
    assert spec.oks_sigmas is not None, "oks_sigmas should not be None"
    assert len(spec.oks_sigmas) == 23, (
        f"Expected 23 oks_sigmas, got {len(spec.oks_sigmas)}"
    )
    for i, sigma in enumerate(spec.oks_sigmas):
        assert sigma > 0, f"oks_sigmas[{i}] = {sigma} should be positive"
        assert sigma == 0.05, (
            f"Expected all sigmas to be 0.05, got {sigma} at index {i}"
        )

    # Validate skeleton (should be empty)
    assert spec.skeleton == [], f"Expected empty skeleton, got {spec.skeleton}"

    # Validate all skeleton indices are in bounds (trivially true for empty skeleton)
    for connection in spec.skeleton:
        for kpt_idx in connection:
            assert 0 <= kpt_idx < spec.K, (
                f"Skeleton index {kpt_idx} out of bounds [0, {spec.K})"
            )


def test_coco_template_backward_compatibility():
    """Test that existing COCO template still works correctly."""
    # Load the COCO template YAML
    with open("data/coco_kpts.yaml", "r") as f:
        dataset_dict = yaml.safe_load(f)

    # Parse into KeypointSpec
    spec = from_dataset_dict(dataset_dict)

    # Validate COCO properties
    assert spec.K == 17, f"Expected COCO K=17, got K={spec.K}"
    assert spec.D == 3, f"Expected COCO D=3, got D={spec.D}"

    # Validate flip_index bounds and length
    assert len(spec.flip_index) == 17, (
        f"Expected COCO flip_index length 17, got {len(spec.flip_index)}"
    )
    for i, flip_idx in enumerate(spec.flip_index):
        assert 0 <= flip_idx < 17, (
            f"COCO flip_index[{i}] = {flip_idx} out of bounds [0, 17)"
        )

    # Validate flip_index is a permutation
    flip_set = set(spec.flip_index)
    expected_set = set(range(17))
    assert flip_set == expected_set, (
        f"COCO flip_index is not a permutation: {flip_set} != {expected_set}"
    )

    # Validate oks_sigmas for COCO
    assert spec.oks_sigmas is not None, "COCO oks_sigmas should not be None"
    assert len(spec.oks_sigmas) == 17, (
        f"Expected 17 COCO oks_sigmas, got {len(spec.oks_sigmas)}"
    )
    for i, sigma in enumerate(spec.oks_sigmas):
        assert sigma > 0, f"COCO oks_sigmas[{i}] = {sigma} should be positive"

    # Validate names
    assert len(spec.names) == 17, f"Expected 17 COCO names, got {len(spec.names)}"

    # Validate skeleton connections are in bounds
    for connection in spec.skeleton:
        assert len(connection) == 2, (
            f"Skeleton connection {connection} should have 2 elements"
        )
        for kpt_idx in connection:
            assert 0 <= kpt_idx < 17, (
                f"COCO skeleton index {kpt_idx} out of bounds [0, 17)"
            )


def test_flip_index_permutation_property():
    """Test that flip_index satisfies permutation properties for both templates."""
    templates = [("data/kpts_23_template.yaml", 23), ("data/coco_kpts.yaml", 17)]

    for template_path, expected_k in templates:
        with open(template_path, "r") as f:
            dataset_dict = yaml.safe_load(f)

        spec = from_dataset_dict(dataset_dict)

        # Test that flip_index is a valid permutation
        assert len(spec.flip_index) == expected_k
        assert set(spec.flip_index) == set(range(expected_k))

        # Test that applying flip twice returns to original (involution property)
        double_flipped = [
            spec.flip_index[spec.flip_index[i]] for i in range(expected_k)
        ]
        assert double_flipped == list(range(expected_k)), (
            f"Double flip should be identity for {template_path}"
        )


def test_keypoint_spec_validation():
    """Test that KeypointSpec validation catches invalid configurations."""
    # Test with the 23-keypoint template to ensure validation passes
    with open("data/kpts_23_template.yaml", "r") as f:
        dataset_dict = yaml.safe_load(f)

    # This should not raise any exceptions
    spec = from_dataset_dict(dataset_dict)

    # Verify all the expected properties are correctly set
    assert spec.K == 23
    assert spec.D == 3
    assert len(spec.names) == 23
    assert len(spec.flip_index) == 23
    assert len(spec.oks_sigmas) == 23
    assert spec.skeleton == []


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_23_keypoint_template_parsing()
    test_coco_template_backward_compatibility()
    test_flip_index_permutation_property()
    test_keypoint_spec_validation()
    print("All tests passed!")
