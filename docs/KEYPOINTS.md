# Configurable Keypoints Guide

This document provides comprehensive guidance on YOLOv7-pose's configurable keypoint system, which supports arbitrary numbers of keypoints (K) and dimensions per keypoint (D).

## KeypointSpec Fields

The keypoint configuration is defined in dataset YAML files through the following fields:

### Core Configuration

- **`kpt_shape: [K, D]`** - Required. Defines K keypoints with D dimensions each
- **`names`** - Optional. List of keypoint names (length K)
- **`skeleton`** - Optional. List of [point1, point2] pairs for drawing connections (0-based indices in range [0,K))
- **`flip_index`** - Optional. Permutation array for horizontal flipping (length K, values in [0,K))
- **`oks_sigmas`** - Optional. Object Keypoint Similarity sigmas for evaluation (length K)

### Defaults and Backward Compatibility

- When `kpt_shape` is missing: defaults to COCO format `[17, 3]`
- When `oks_sigmas` is missing: defaults to COCO sigmas (length 17) or uniform values for other K
- When `flip_index` is missing: identity mapping (no flipping)
- When `skeleton` is missing: no connections drawn
- When `names` is missing: generic names like "kpt_0", "kpt_1", etc.

## Data Flow and Usage

### Data Loading

- **File**: [`utils/datasets.py`](../utils/datasets.py)
- **Behavior**: Parses label files expecting `5 + K*D` columns per detection
  - Columns 0-4: `[class, x_center, y_center, width, height]`
  - Columns 5+: `K*D` keypoint coordinates/visibility values

### Model Configuration

- **File**: [`models/yolo.py`](../models/yolo.py)
- **Fields**:
  - `nkpt`: Number of keypoints (K). Must match dataset
  - `kpt_dim`: Dimensions per keypoint (D). Defaults to dataset D or 3 if omitted
- **Head**: Model outputs `nc + 5 + K*D` values per detection

### Postprocessing

- **File**: [`utils/general.py`](../utils/general.py)
- **Behavior**: NMS and postprocessing dynamically slice keypoint channels based on K and D
- **Output**: Detections with shape `[..., 6 + K*D]` (adds confidence score)

### Loss Computation

- **File**: [`utils/loss.py`](../utils/loss.py)
- **Behavior**: Loads `oks_sigmas` from dataset and computes keypoint loss respecting K and D
- **OKS**: Uses Object Keypoint Similarity for evaluation when sigmas are provided

### Metrics and Evaluation

- **File**: [`utils/metrics.py`](../utils/metrics.py)
- **Behavior**: Evaluation metrics adapt to K keypoints using dataset-specific `oks_sigmas`

### Visualization

- **File**: [`utils/plots.py`](../utils/plots.py)
- **Behavior**:
  - Renders keypoints based on `skeleton` connections
  - Handles visibility conditionally based on D dimensions
  - Uses `names` for labeling when available

### Export and Inference

- **Export**: [`models/export.py`](../models/export.py) embeds keypoint metadata in exported models
- **ONNX Inference**: [`onnx_inference/yolo_pose_onnx_inference.py`](../onnx_inference/yolo_pose_onnx_inference.py) reads metadata from model or dataset YAML

## Authoring a New Dataset YAML

### Starting Template

Begin with [`data/kpts_23_template.yaml`](../data/kpts_23_template.yaml) as a reference:

```yaml
# Required
kpt_shape: [23, 3]  # K=23 keypoints, D=3 dimensions (x,y,visibility)

# Recommended
flip_index: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21]
skeleton: [[0,1], [1,2], [2,3], ...]  # Connection pairs (0-based)
oks_sigmas: [0.026, 0.025, 0.025, ...]  # Length K evaluation sigmas
names: ['nose', 'left_eye', 'right_eye', ...]  # Length K keypoint names

# Standard dataset fields
path: ./datasets/custom_keypoints
train: images/train
val: images/val
nc: 1
names: ['person']
```

### Requirements

- **`kpt_shape: [K, D]`** is mandatory
- **Skeleton indices** must be 0-based and in range [0, K)
- **`flip_index`** must be a permutation of [0, 1, ..., K-1]
- **`oks_sigmas`** length must equal K if provided

## Dimension (D) Guidance

### Minimum Requirements

- **D ≥ 2**: Required for x,y coordinates
- **D ≥ 3**: Third dimension treated as visibility (0=not visible, 1=occluded, 2=visible)
- **D > 3**: Extra dimensions passed through model, available for downstream processing

### Visibility Handling

- When **D = 2**: All keypoints assumed visible during training and inference
- When **D ≥ 3**: Visibility values used for loss computation and visualization
- **Visibility encoding**: 0=not annotated, 1=occluded, 2=visible (COCO standard)

## Migration from Fixed 17-Keypoint Format

### Key Changes Made

- **Hard-coded values replaced**:
  - `56` → `5 + K*D` (label columns)
  - `57` → `1 + 5 + K*D` or `nc + 5 + K*D` (output channels)
- **Dynamic sizing**: All components now use expressions based on K and D
- **Metadata flow**: Keypoint configuration propagates from dataset through training to inference

### Migration Checklist

1. **Update dataset YAML**: Add `kpt_shape`, verify other keypoint fields
2. **Update model config**: Ensure `nkpt` matches your K value
3. **Verify labels**: Check that your label files have `5 + K*D` columns
4. **Update flip_index**: Ensure it's a valid permutation for your K keypoints
5. **Update skeleton**: Use 0-based indices in range [0, K)
6. **Update oks_sigmas**: Provide K values for proper evaluation

## Troubleshooting

### Common Issues

**Label Size Mismatch**

```
Expected 5 + K*D columns, got X
```

- **Solution**: Verify your label files have exactly `5 + K*D` columns
- **Check**: K and D values in your dataset YAML `kpt_shape`

**OKS Evaluation Errors**

```
oks_sigmas length mismatch: expected K, got X
```

- **Solution**: Ensure `len(oks_sigmas) == K` in your dataset YAML
- **Alternative**: Remove `oks_sigmas` to use default uniform values

**Incorrect Flipping**

```
Keypoints appear in wrong positions after augmentation
```

- **Solution**: Verify `flip_index` is a valid permutation of [0, 1, ..., K-1]
- **Check**: Each value in [0, K) appears exactly once

**Model Architecture Mismatch**

```
Model expects nkpt=X but dataset has K keypoints
```

- **Solution**: Update model config `nkpt` to match dataset `kpt_shape[0]`
- **Alternatively**: Update dataset to match pre-trained model expectations

### Validation Commands

**Quick smoke test**:

```bash
python -m pytest tests/test_kpt_templates.py -q
```

**Validate dataset loading**:

```bash
python -c "from utils.datasets import LoadImagesAndLabels;
           dataset = LoadImagesAndLabels('path/to/labels', data_dict={'kpt_shape': [K, D]})"
```

**Check model compatibility**:

```bash
python -c "from models.yolo import Model;
           Model('cfg/yolov7-w6-pose.yaml', ch=3, nc=1, anchors=None).eval()"
```

## Examples

### 17 Keypoints (COCO)

- **Template**: [`data/coco_kpts.yaml`](../data/coco_kpts.yaml)
- **Config**: `kpt_shape: [17, 3]`
- **Labels**: 56 columns (5 + 17\*3)

### 23 Keypoints (Extended)

- **Template**: [`data/kpts_23_template.yaml`](../data/kpts_23_template.yaml)
- **Config**: `kpt_shape: [23, 3]`
- **Labels**: 74 columns (5 + 23\*3)

### Custom Format

```yaml
kpt_shape: [21, 2] # 21 keypoints, x,y only (no visibility)
# Results in 47 columns (5 + 21*2)
```
