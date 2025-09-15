# yolov7-pose

Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806).

## Configurable Keypoints (K,D)

This project supports arbitrary K keypoints and D dimensions per keypoint, enabling flexible pose estimation beyond the standard 17-keypoint COCO format.

### Configuration

Configure keypoints via dataset YAML files:

- **K keypoints, D dimensions**: Set via `kpt_shape: [K, D]`
- **Templates**: [`data/coco_kpts.yaml`](data/coco_kpts.yaml), [`data/coco_kpts_128.yaml`](data/coco_kpts_128.yaml), [`data/kpts_23_template.yaml`](data/kpts_23_template.yaml)
- **Additional fields**: `flip_index`, `skeleton`, `oks_sigmas`, `names`

### Quick Start: Switch to 23 Keypoints

1. **Prepare dataset YAML**: Copy/adapt [`data/kpts_23_template.yaml`](data/kpts_23_template.yaml)
2. **Configure model**: Ensure [`cfg/yolov7-w6-pose.yaml`](cfg/yolov7-w6-pose.yaml) has `nkpt: 23` (kpt_dim defaults to dataset D or 3)

3. **Train**:

```bash
python train.py --data data/kpts_23_template.yaml --cfg cfg/yolov7-w6-pose.yaml --weights '' --epochs 1 --device 0
```

4. **Detect** (auto-reads K,D from model):

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source data/images
```

5. **Export ONNX** (embeds keypoint metadata):

```bash
python models/export.py --weights runs/train/exp/weights/best.pt --include onnx
```

6. **ONNX inference** (reads metadata or dataset):

```bash
python onnx_inference/yolo_pose_onnx_inference.py --onnx-path model.onnx --dataset data/kpts_23_template.yaml
```

### Backward Compatibility

- Missing `kpt_shape` defaults to COCO: K=17, D=3
- Hard-coded values (17/56/57) replaced by expressions using K and D

### Validation

```bash
python -m pytest tests/test_kpt_templates.py -q
```

**📖 Detailed Guide**: [`docs/KEYPOINTS.md`](docs/KEYPOINTS.md)

## Dataset preparison

[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

```shell
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy

TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

```shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

- [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
- [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
- [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
- [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
- [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
- [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
