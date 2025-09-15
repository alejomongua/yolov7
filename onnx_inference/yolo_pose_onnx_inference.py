import os
import sys
import json
import yaml
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.keypoints import KeypointSpec, coco_defaults, from_dataset_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path", type=str, default="./yolov5s6_pose_640_ti_lite_54p9_82p2.onnx"
)
parser.add_argument("--img-path", type=str, default="./sample_ips.txt")
parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
parser.add_argument(
    "--kpt-shape", type=str, default=None, help="Keypoint shape as 'K,D' (e.g., '17,3')"
)
parser.add_argument(
    "--dataset", type=str, default=None, help="Path to dataset YAML file"
)
args = parser.parse_args()


_CLASS_COLOR_MAP = [
    (0, 0, 255),  # Person (blue).
    (255, 0, 0),  # Bear (red).
    (0, 255, 0),  # Tree (lime).
    (255, 0, 255),  # Bird (fuchsia).
    (0, 255, 255),  # Sky (aqua).
    (255, 255, 0),  # Cat (yellow).
]


def get_keypoint_spec():
    """Get keypoint specification from CLI args, dataset, or defaults."""
    spec = None

    # Priority 1: Dataset YAML if provided
    if args.dataset and os.path.exists(args.dataset):
        try:
            with open(args.dataset, "r") as f:
                dataset_dict = yaml.safe_load(f)
            spec = from_dataset_dict(dataset_dict)
            print(f"Using keypoint spec from dataset: K={spec.K}, D={spec.D}")
        except Exception as e:
            print(f"Warning: Failed to load dataset {args.dataset}: {e}")

    # Priority 2: CLI kpt-shape if provided
    if spec is None and args.kpt_shape:
        try:
            K, D = map(int, args.kpt_shape.split(","))
            if K == 17:
                # Use COCO defaults for K=17
                spec = coco_defaults()
                spec.D = D  # Update D if different
            else:
                # Create basic spec for non-COCO keypoint counts
                spec = KeypointSpec(
                    K=K,
                    D=D,
                    names=[f"k{i}" for i in range(K)],
                    skeleton=[],  # No skeleton for unknown layouts
                    flip_index=list(range(K)),  # Identity mapping
                )
            print(f"Using keypoint spec from CLI: K={K}, D={D}")
        except Exception as e:
            print(f"Warning: Failed to parse kpt-shape '{args.kpt_shape}': {e}")

    # Priority 3: Read from ONNX metadata if available
    if spec is None:
        try:
            session = onnxruntime.InferenceSession(args.model_path, None)
            metadata = session.get_modelmeta().custom_metadata_map

            if "kpt_shape" in metadata:
                kpt_shape_str = metadata["kpt_shape"]
                K, D = json.loads(kpt_shape_str)

                # Try to read other metadata
                skeleton = json.loads(metadata.get("kpt_skeleton", "[]"))
                names = json.loads(
                    metadata.get(
                        "kpt_names", f"[{','.join([f'"k{i}"' for i in range(K)])}]"
                    )
                )
                flip_index = json.loads(
                    metadata.get("kpt_flip_index", f"[{','.join(map(str, range(K)))}]")
                )
                oks_sigmas = json.loads(metadata.get("oks_sigmas", "null"))

                spec = KeypointSpec(
                    K=K,
                    D=D,
                    names=names,
                    skeleton=skeleton,
                    flip_index=flip_index,
                    oks_sigmas=oks_sigmas,
                )
                print(f"Using keypoint spec from ONNX metadata: K={K}, D={D}")
        except Exception as e:
            print(f"Warning: Failed to read ONNX metadata: {e}")

    # Priority 4: Default to COCO
    if spec is None:
        spec = coco_defaults()
        print(f"Using default COCO keypoint spec: K={spec.K}, D={spec.D}")

    return spec


def generate_colors(K):
    """Generate a color palette for K keypoints."""
    base_palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    # Create palette sized to K by cycling through base palette
    return np.array([base_palette[i % len(base_palette)] for i in range(K)])


def read_img(img_file, img_mean=127.5, img_scale=1 / 127.5):
    img = cv2.imread(img_file)[:, :, ::-1]
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def model_inference(model_path=None, input=None):
    # onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    return output


def model_inference_image_list(
    model_path, img_path=None, mean=None, scale=None, dst_path=None
):
    os.makedirs(args.dst_path, exist_ok=True)
    img_file_list = list(open(img_path))
    pbar = enumerate(img_file_list)
    max_index = 20
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))
    for img_index, img_file in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img_file = img_file.rstrip()
        input = read_img(img_file, mean, scale)
        output = model_inference(model_path, input)
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        post_process(img_file, dst_file, output[0], score_threshold=0.3)


def post_process(img_file, dst_file, output, score_threshold=0.3, spec=None):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    # Get keypoint spec
    if spec is None:
        spec = get_keypoint_spec()

    K, D = spec.K, spec.D
    no_kpt = K * D

    # Parse output: [x1, y1, x2, y2, conf, class, kpt1_x, kpt1_y, kpt1_v, ...]
    det_bboxes = output[:, 0:4]
    det_scores = output[:, 4]
    det_labels = output[:, 5]
    kpts = output[:, 6 : 6 + no_kpt].reshape(-1, K, D)  # Reshape to (N, K, D)

    img = cv2.imread(img_file)
    dst_txt_file = dst_file.replace("png", "txt")
    f = open(dst_txt_file, "wt")

    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx].flatten()  # Flatten back to 1D for compatibility

        if det_scores[idx] > 0:
            f.write(
                "{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(
                    det_labels[idx],
                    det_scores[idx],
                    det_bbox[1],
                    det_bbox[0],
                    det_bbox[3],
                    det_bbox[2],
                )
            )

        if det_scores[idx] > score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx]) % len(_CLASS_COLOR_MAP)]
            img = cv2.rectangle(
                img,
                (int(det_bbox[0]), int(det_bbox[1])),
                (int(det_bbox[2]), int(det_bbox[3])),
                color_map[::-1],
                2,
            )
            cv2.putText(
                img,
                "id:{}".format(int(det_labels[idx])),
                (int(det_bbox[0] + 5), int(det_bbox[1]) + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_map[::-1],
                2,
            )
            cv2.putText(
                img,
                "score:{:2.1f}".format(det_scores[idx]),
                (int(det_bbox[0] + 5), int(det_bbox[1]) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_map[::-1],
                2,
            )
            plot_skeleton_kpts(img, kpt, spec=spec)

    cv2.imwrite(dst_file, img)
    f.close()


def plot_skeleton_kpts(im, kpts, spec=None):
    """Plot keypoints and skeleton with dynamic specification."""
    if spec is None:
        spec = get_keypoint_spec()

    K, D = spec.K, spec.D
    steps = D  # Use D as steps
    skeleton_connections = spec.skeleton

    # Generate colors for this keypoint configuration
    palette = generate_colors(K)
    pose_kpt_color = palette
    pose_limb_color = (
        palette[: len(skeleton_connections)] if skeleton_connections else palette[:0]
    )

    radius = 5
    num_kpts = len(kpts) // steps

    # Plot keypoints
    for kid in range(min(num_kpts, K)):
        if kid < len(pose_kpt_color):
            r, g, b = pose_kpt_color[kid]
        else:
            r, g, b = 255, 255, 255  # White fallback

        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]

        # Check confidence if available (D >= 3)
        if D >= 3 and steps >= 3:
            conf = kpts[steps * kid + 2]
            if conf <= 0.5:
                continue

        cv2.circle(
            im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
        )

    # Plot skeleton connections
    for sk_id, sk in enumerate(skeleton_connections):
        if sk_id < len(pose_limb_color):
            r, g, b = pose_limb_color[sk_id]
        else:
            r, g, b = 255, 255, 255  # White fallback

        # Skeleton connections are 0-based in our KeypointSpec
        kpt1_idx = sk[0]
        kpt2_idx = sk[1]

        # Ensure indices are valid
        if not (0 <= kpt1_idx < num_kpts and 0 <= kpt2_idx < num_kpts):
            continue

        pos1 = (int(kpts[kpt1_idx * steps]), int(kpts[kpt1_idx * steps + 1]))
        pos2 = (int(kpts[kpt2_idx * steps]), int(kpts[kpt2_idx * steps + 1]))

        # Check confidence if available
        if D >= 3 and steps >= 3:
            conf1 = kpts[kpt1_idx * steps + 2]
            conf2 = kpts[kpt2_idx * steps + 2]
            if conf1 <= 0.5 or conf2 <= 0.5:
                continue

        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def main():
    # Get keypoint specification first
    spec = get_keypoint_spec()
    print(
        f"Keypoint configuration: K={spec.K}, D={spec.D}, skeleton_connections={len(spec.skeleton)}"
    )

    model_inference_image_list(
        model_path=args.model_path,
        img_path=args.img_path,
        mean=0.0,
        scale=0.00392156862745098,
        dst_path=args.dst_path,
    )


if __name__ == "__main__":
    main()
