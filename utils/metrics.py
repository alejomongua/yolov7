# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general
from .keypoints import coco_defaults


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=()):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / "PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / "F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / "P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / "R_curve.png", names, ylabel="Recall")

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir="", names=()):
        try:
            import seaborn as sn

            array = self.matrix / (
                self.matrix.sum(0).reshape(1, self.nc + 1) + 1e-6
            )  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(
                names
            ) == self.nc  # apply names to ticklabels
            sn.heatmap(
                array,
                annot=self.nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                xticklabels=names + ["background FP"] if labels else "auto",
                yticklabels=names + ["background FN"] if labels else "auto",
            ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel("True")
            fig.axes[0].set_ylabel("Predicted")
            fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(
    px, py, save_dir="mc_curve.png", names=(), xlabel="Confidence", ylabel="Metric"
):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def compute_oks(pred_kpts, gt_kpts, sigmas=None, area=None):
    """
    Compute Object Keypoint Similarity (OKS) between predicted and ground truth keypoints.

    Args:
        pred_kpts: Predicted keypoints tensor of shape (N, K, D) where N=num_predictions, K=num_keypoints, D=dimensions
        gt_kpts: Ground truth keypoints tensor of shape (M, K, D) where M=num_gt, K=num_keypoints, D=dimensions
        sigmas: OKS sigmas tensor of shape (K,). If None, defaults to COCO for K=17 else uniform 0.05
        area: Area tensor of shape (M,) for normalizing distance. If None, uses unit area (no normalization)

    Returns:
        oks_matrix: OKS similarity matrix of shape (N, M)
    """
    if pred_kpts.numel() == 0 or gt_kpts.numel() == 0:
        return torch.zeros(
            pred_kpts.shape[0], gt_kpts.shape[0], device=pred_kpts.device
        )

    # Ensure consistent shapes
    if len(pred_kpts.shape) == 2:
        # Reshape from (N, K*D) to (N, K, D)
        K = (
            pred_kpts.shape[1] // gt_kpts.shape[-1]
            if len(gt_kpts.shape) == 3
            else len(sigmas)
            if sigmas is not None
            else 17
        )
        D = pred_kpts.shape[1] // K
        pred_kpts = pred_kpts.view(-1, K, D)

    if len(gt_kpts.shape) == 2:
        # Reshape from (M, K*D) to (M, K, D)
        K = (
            gt_kpts.shape[1] // pred_kpts.shape[-1]
            if len(pred_kpts.shape) == 3
            else len(sigmas)
            if sigmas is not None
            else 17
        )
        D = gt_kpts.shape[1] // K
        gt_kpts = gt_kpts.view(-1, K, D)

    N, K, D = pred_kpts.shape
    M = gt_kpts.shape[0]

    # Set default sigmas if not provided
    if sigmas is None:
        if K == 17:
            # Use COCO defaults for K=17
            coco_spec = coco_defaults()
            sigmas = torch.tensor(
                coco_spec.oks_sigmas, device=pred_kpts.device, dtype=torch.float32
            )
        else:
            # Use uniform sigmas for non-COCO keypoint counts
            sigmas = torch.full(
                (K,), 0.05, device=pred_kpts.device, dtype=torch.float32
            )

    # Validate sigmas length
    if len(sigmas) != K:
        raise ValueError(
            f"sigmas length ({len(sigmas)}) must equal number of keypoints ({K})"
        )

    # Use only x,y coordinates for distance computation
    pred_xy = pred_kpts[:, :, :2]  # (N, K, 2)
    gt_xy = gt_kpts[:, :, :2]  # (M, K, 2)

    # Visibility mask: if D >= 3, use visibility; else assume all visible if coordinates != 0
    if D >= 3:
        pred_vis = pred_kpts[:, :, 2] > 0  # (N, K)
        gt_vis = gt_kpts[:, :, 2] > 0  # (M, K)
    else:
        # For D < 3, consider keypoint visible if both x,y are non-zero
        pred_vis = (pred_xy[:, :, 0] != 0) & (pred_xy[:, :, 1] != 0)  # (N, K)
        gt_vis = (gt_xy[:, :, 0] != 0) & (gt_xy[:, :, 1] != 0)  # (M, K)

    # Set default area if not provided
    if area is None:
        area = torch.ones(M, device=pred_kpts.device)

    # Compute squared distances between all pred-gt pairs
    # pred_xy: (N, K, 2) -> (N, 1, K, 2)
    # gt_xy: (M, K, 2) -> (1, M, K, 2)
    pred_xy_exp = pred_xy.unsqueeze(1)  # (N, 1, K, 2)
    gt_xy_exp = gt_xy.unsqueeze(0)  # (1, M, K, 2)

    # Squared distances: (N, M, K)
    dist_sq = ((pred_xy_exp - gt_xy_exp) ** 2).sum(dim=-1)

    # Visibility mask for valid comparisons: (N, M, K)
    pred_vis_exp = pred_vis.unsqueeze(1)  # (N, 1, K)
    gt_vis_exp = gt_vis.unsqueeze(0)  # (1, M, K)
    vis_mask = pred_vis_exp & gt_vis_exp  # (N, M, K)

    # Area normalization: area -> (1, M, 1) for broadcasting
    area_exp = area.unsqueeze(0).unsqueeze(-1)  # (1, M, 1)

    # Sigmas normalization: sigmas -> (1, 1, K) for broadcasting
    sigmas_exp = sigmas.unsqueeze(0).unsqueeze(0)  # (1, 1, K)

    # OKS computation: exp(-d^2 / (2 * s^2 * k^2))
    # where d is distance, s is area, k is sigma
    oks_per_kpt = torch.exp(-dist_sq / (2 * area_exp * sigmas_exp**2))

    # Apply visibility mask
    oks_per_kpt = oks_per_kpt * vis_mask.float()

    # Sum over keypoints and normalize by number of visible keypoints
    oks_sum = oks_per_kpt.sum(dim=-1)  # (N, M)
    vis_count = vis_mask.sum(dim=-1).float()  # (N, M)

    # Avoid division by zero
    oks_matrix = torch.where(
        vis_count > 0, oks_sum / vis_count, torch.zeros_like(oks_sum)
    )

    return oks_matrix


def compute_ap_oks(
    pred_kpts, pred_scores, gt_kpts, sigmas=None, area=None, iou_thresholds=None
):
    """
    Compute Average Precision using OKS similarity for keypoint detection.

    Args:
        pred_kpts: Predicted keypoints tensor of shape (N, K, D)
        pred_scores: Prediction confidence scores of shape (N,)
        gt_kpts: Ground truth keypoints tensor of shape (M, K, D)
        sigmas: OKS sigmas tensor of shape (K,). If None, uses defaults
        area: Area tensor of shape (M,) for OKS computation. If None, uses unit area
        iou_thresholds: OKS thresholds for AP computation. If None, uses [0.5, 0.55, ..., 0.95]

    Returns:
        ap_scores: AP scores for each threshold
        precisions: Precision curves
        recalls: Recall curves
    """
    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.5, 1.0, 0.05, device=pred_kpts.device)

    if pred_kpts.numel() == 0:
        return (
            torch.zeros(len(iou_thresholds)),
            torch.zeros((len(iou_thresholds), 0)),
            torch.zeros((len(iou_thresholds), 0)),
        )

    # Sort predictions by confidence score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_kpts_sorted = pred_kpts[sorted_indices]
    pred_scores_sorted = pred_scores[sorted_indices]

    # Compute OKS matrix
    oks_matrix = compute_oks(pred_kpts_sorted, gt_kpts, sigmas, area)  # (N, M)

    num_gt = gt_kpts.shape[0]
    num_pred = pred_kpts_sorted.shape[0]

    ap_scores = []
    all_precisions = []
    all_recalls = []

    for threshold in iou_thresholds:
        # Find matches above threshold
        matches = oks_matrix > threshold  # (N, M)

        # Track which GT have been matched
        gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=pred_kpts.device)
        pred_matched = torch.zeros(num_pred, dtype=torch.bool, device=pred_kpts.device)

        tp = torch.zeros(num_pred, device=pred_kpts.device)
        fp = torch.zeros(num_pred, device=pred_kpts.device)

        # Process predictions in order of confidence
        for pred_idx in range(num_pred):
            # Find best matching GT for this prediction
            pred_matches = matches[pred_idx]  # (M,)
            if pred_matches.any():
                # Find unmatched GT with highest OKS
                unmatched_gt = pred_matches & ~gt_matched
                if unmatched_gt.any():
                    # Find best unmatched GT
                    best_gt_idx = torch.argmax(
                        oks_matrix[pred_idx] * unmatched_gt.float()
                    )
                    gt_matched[best_gt_idx] = True
                    pred_matched[pred_idx] = True
                    tp[pred_idx] = 1
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1

        # Compute precision and recall curves
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        recalls = tp_cumsum / (num_gt + 1e-16)

        # Compute AP using 101-point interpolation
        recall_thresholds = torch.linspace(0, 1, 101, device=pred_kpts.device)
        interpolated_precisions = torch.zeros_like(recall_thresholds)

        for i, r_thresh in enumerate(recall_thresholds):
            # Find precisions for recalls >= r_thresh
            valid_recalls = recalls >= r_thresh
            if valid_recalls.any():
                interpolated_precisions[i] = precisions[valid_recalls].max()

        ap = interpolated_precisions.mean()
        ap_scores.append(ap)
        all_precisions.append(precisions)
        all_recalls.append(recalls)

    return torch.stack(ap_scores), all_precisions, all_recalls
