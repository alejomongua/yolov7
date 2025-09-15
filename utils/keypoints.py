"""
Keypoint specification and configuration utilities for YOLOv7 pose estimation.

This module provides configurable keypoint shapes (K, D) and removes hardcoded
constants while maintaining backward compatibility with COCO datasets.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class KeypointSpec:
    """
    Specification for keypoint configuration.

    Args:
        K: Number of keypoints per object
        D: Dimensions per keypoint (typically 2 for x,y or 3 for x,y,visibility)
        names: List of keypoint names (length K)
        skeleton: List of skeleton connections as [kpt1_idx, kpt2_idx] pairs
        flip_index: List mapping keypoint indices for horizontal flip (length K)
        oks_sigmas: Optional OKS (Object Keypoint Similarity) sigmas for evaluation (length K)
    """

    K: int
    D: int
    names: List[str]
    skeleton: List[List[int]]
    flip_index: List[int]
    oks_sigmas: Optional[List[float]] = None

    def __post_init__(self):
        """Validate the keypoint specification after initialization."""
        self._validate()

    def _validate(self):
        """Validate keypoint specification fields."""
        if self.K <= 0:
            raise ValueError(f"K (number of keypoints) must be positive, got {self.K}")
        if self.D <= 0:
            raise ValueError(
                f"D (dimensions per keypoint) must be positive, got {self.D}"
            )

        if len(self.names) != self.K:
            raise ValueError(
                f"Length of names ({len(self.names)}) must equal K ({self.K})"
            )

        if len(self.flip_index) != self.K:
            raise ValueError(
                f"Length of flip_index ({len(self.flip_index)}) must equal K ({self.K})"
            )

        # Validate flip_index bounds
        for i, flip_idx in enumerate(self.flip_index):
            if not (0 <= flip_idx < self.K):
                raise ValueError(
                    f"flip_index[{i}] = {flip_idx} is out of bounds [0, {self.K})"
                )

        # Validate skeleton connections
        for connection in self.skeleton:
            if len(connection) != 2:
                raise ValueError(
                    f"Skeleton connection {connection} must have exactly 2 elements"
                )
            for kpt_idx in connection:
                if not (0 <= kpt_idx < self.K):
                    raise ValueError(
                        f"Skeleton keypoint index {kpt_idx} is out of bounds [0, {self.K})"
                    )

        # Validate oks_sigmas if provided
        if self.oks_sigmas is not None:
            if len(self.oks_sigmas) != self.K:
                raise ValueError(
                    f"Length of oks_sigmas ({len(self.oks_sigmas)}) must equal K ({self.K})"
                )
            for i, sigma in enumerate(self.oks_sigmas):
                if sigma <= 0:
                    raise ValueError(f"oks_sigmas[{i}] = {sigma} must be positive")


def coco_defaults() -> KeypointSpec:
    """
    Return the default COCO keypoint specification.

    Returns:
        KeypointSpec: COCO specification with K=17, D=3
    """
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]
    # Convert to 0-based indexing
    skeleton = [[a - 1, b - 1] for a, b in skeleton]

    flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    oks_sigmas = [
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
    ]

    return KeypointSpec(
        K=17,
        D=3,
        names=names,
        skeleton=skeleton,
        flip_index=flip_index,
        oks_sigmas=oks_sigmas,
    )


def from_dataset_dict(dataset_dict: Dict[str, Any]) -> KeypointSpec:
    """
    Create a KeypointSpec from a dataset dictionary.

    Args:
        dataset_dict: Dictionary containing dataset configuration

    Returns:
        KeypointSpec: Keypoint specification based on dataset configuration

    The function reads:
    - kpt_shape: [K, D] from dataset dict if present; else uses [17, 3]
    - Optional fields if present: kpt_names, kpt_skeleton, kpt_flip_index, oks_sigmas
    - For K==17, uses COCO defaults when optional fields are missing
    - For other K, creates safe fallbacks
    """
    # Read kpt_shape or default to COCO
    kpt_shape = dataset_dict.get("kpt_shape", [17, 3])
    if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) != 2:
        raise ValueError(f"kpt_shape must be a list/tuple of length 2, got {kpt_shape}")

    K, D = int(kpt_shape[0]), int(kpt_shape[1])

    # If K==17, we can use COCO defaults as fallbacks
    if K == 17:
        coco_spec = coco_defaults()
        default_names = coco_spec.names
        default_skeleton = coco_spec.skeleton
        default_flip_index = coco_spec.flip_index
        default_oks_sigmas = coco_spec.oks_sigmas
    else:
        # Create safe fallbacks for non-COCO keypoint counts
        default_names = [f"k{i}" for i in range(K)]
        default_skeleton = []  # No skeleton connections for unknown keypoint layouts
        default_flip_index = list(range(K))  # Identity mapping (no flip)
        default_oks_sigmas = None

    # Read optional fields with fallbacks
    names = dataset_dict.get("kpt_names", default_names)
    skeleton = dataset_dict.get("kpt_skeleton", default_skeleton)
    flip_index = dataset_dict.get("kpt_flip_index", default_flip_index)
    oks_sigmas = dataset_dict.get("oks_sigmas", default_oks_sigmas)

    # Validate types and convert if needed
    if not isinstance(names, list):
        raise ValueError(f"kpt_names must be a list, got {type(names)}")

    if not isinstance(skeleton, list):
        raise ValueError(f"kpt_skeleton must be a list, got {type(skeleton)}")

    if not isinstance(flip_index, list):
        raise ValueError(f"kpt_flip_index must be a list, got {type(flip_index)}")

    if oks_sigmas is not None and not isinstance(oks_sigmas, list):
        raise ValueError(f"oks_sigmas must be a list or None, got {type(oks_sigmas)}")

    # Create and return the specification
    try:
        return KeypointSpec(
            K=K,
            D=D,
            names=names,
            skeleton=skeleton,
            flip_index=flip_index,
            oks_sigmas=oks_sigmas,
        )
    except Exception as e:
        raise ValueError(f"Invalid keypoint specification in dataset: {e}")
