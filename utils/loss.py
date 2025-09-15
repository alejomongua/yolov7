# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel
from utils.keypoints import from_dataset_dict, coco_defaults


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )
        BCE_kptv = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(
            eps=h.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = (
            model.module.model[-1] if is_parallel(model) else model.model[-1]
        )  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            model.gr,
            h,
            autobalance,
        )
        for k in "na", "nc", "nl", "anchors", "nkpt":
            setattr(self, k, getattr(det, k))

        # Get KeypointSpec for dynamic K,D,sigmas
        spec = getattr(model, "keypoint_spec", None)
        if spec is None:
            # Fallback: try to construct from model attributes if they exist
            if hasattr(model, "nkpt") and hasattr(model, "kpt_dim"):
                # Create a minimal spec using model attributes
                K, D = model.nkpt, model.kpt_dim
                if K == 17:
                    spec = coco_defaults()
                else:
                    # Create minimal spec without full metadata
                    from utils.keypoints import KeypointSpec

                    spec = KeypointSpec(
                        K=K,
                        D=D,
                        names=[f"k{i}" for i in range(K)],
                        skeleton=[],
                        flip_index=list(range(K)),
                        oks_sigmas=None,
                    )
            else:
                # Ultimate fallback to COCO defaults
                spec = coco_defaults()

        # Set keypoint dimensions
        self.nkpt = spec.K
        self.kpt_dim = spec.D
        self.no_kpt = spec.K * spec.D

        # Set OKS sigmas
        if spec.oks_sigmas is not None:
            self.oks_sigmas = torch.tensor(
                spec.oks_sigmas, device=device, dtype=torch.float32
            )
        else:
            if spec.K == 17:
                # Use COCO defaults for K=17
                coco_spec = coco_defaults()
                self.oks_sigmas = torch.tensor(
                    coco_spec.oks_sigmas, device=device, dtype=torch.float32
                )
            else:
                # Create uniform sigmas for non-COCO keypoint counts
                self.oks_sigmas = torch.full(
                    (spec.K,), 0.05, device=device, dtype=torch.float32
                )

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj, lkpt, lkptv = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        # Use dynamic sigmas from keypoint spec
        sigmas = self.oks_sigmas
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                if self.kpt_label:
                    # Dynamic keypoint prediction processing
                    if self.kpt_dim >= 3:
                        # Extract x,y coordinates and visibility scores dynamically
                        pkpt_x = ps[:, 6 :: self.kpt_dim] * 2.0 - 0.5  # x coordinates
                        pkpt_y = ps[:, 7 :: self.kpt_dim] * 2.0 - 0.5  # y coordinates
                        pkpt_score = ps[:, 8 :: self.kpt_dim]  # visibility scores

                        # Target keypoints reshaped to (N, K, D) for easier indexing
                        tkpt_reshaped = tkpt[i].view(-1, self.nkpt, self.kpt_dim)
                        tkpt_x = tkpt_reshaped[:, :, 0]  # x coordinates
                        tkpt_y = tkpt_reshaped[:, :, 1]  # y coordinates

                        # Visibility mask based on x coordinates (if x!=0, keypoint is visible)
                        kpt_mask = tkpt_x != 0
                        lkptv += self.BCEcls(pkpt_score, kpt_mask.float())

                        # OKS-based loss for position
                        d = (pkpt_x - tkpt_x) ** 2 + (pkpt_y - tkpt_y) ** 2
                        s = torch.prod(tbox[i][:, -2:], dim=1, keepdim=True)
                        kpt_loss_factor = (
                            torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)
                        ) / torch.sum(kpt_mask != 0)
                        lkpt += (
                            kpt_loss_factor
                            * (
                                (1 - torch.exp(-d / (s * (4 * sigmas**2) + 1e-9)))
                                * kpt_mask
                            ).mean()
                        )
                    else:
                        # For D < 3, assume all keypoints are visible, no visibility loss
                        pkpt_x = ps[:, 6 :: self.kpt_dim] * 2.0 - 0.5  # x coordinates
                        pkpt_y = ps[:, 7 :: self.kpt_dim] * 2.0 - 0.5  # y coordinates

                        # Target keypoints reshaped to (N, K, D) for easier indexing
                        tkpt_reshaped = tkpt[i].view(-1, self.nkpt, self.kpt_dim)
                        tkpt_x = tkpt_reshaped[:, :, 0]  # x coordinates
                        tkpt_y = tkpt_reshaped[:, :, 1]  # y coordinates

                        # Mask based on non-zero coordinates
                        kpt_mask = tkpt_x != 0

                        # OKS-based loss for position only
                        d = (pkpt_x - tkpt_x) ** 2 + (pkpt_y - tkpt_y) ** 2
                        s = torch.prod(tbox[i][:, -2:], dim=1, keepdim=True)
                        kpt_loss_factor = (
                            torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)
                        ) / torch.sum(kpt_mask != 0)
                        lkpt += (
                            kpt_loss_factor
                            * (
                                (1 - torch.exp(-d / (s * (4 * sigmas**2) + 1e-9)))
                                * kpt_mask
                            ).mean()
                        )
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(
                    0
                ).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lkptv *= self.hyp["cls"]
        lkpt *= self.hyp["kpt"]
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkpt + lkptv
        return loss * bs, torch.cat((lbox, lobj, lcls, lkpt, lkptv, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        if self.kpt_label:
            # Dynamic gain computation: 5 (base) + K*D (keypoints) + 1 (anchor index)
            total_label_cols = 5 + self.no_kpt + 1
            gain = torch.ones(
                total_label_cols, device=targets.device
            )  # normalized to gridspace gain
        else:
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            if self.kpt_label:
                # Dynamic gain setup for keypoints: fill kpt positions with alternating [grid_h, grid_w] pattern
                kpt_start = 6  # keypoints start after [img, cls, x, y, w, h]
                kpt_end = kpt_start + self.no_kpt
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # bbox gain
                # For keypoints, alternate grid dimensions based on x,y pattern
                for k in range(self.nkpt):
                    for d in range(self.kpt_dim):
                        idx = kpt_start + k * self.kpt_dim + d
                        if idx < kpt_end:
                            if d == 0:  # x coordinate
                                gain[idx] = torch.tensor(p[i].shape)[3]  # grid_w
                            elif d == 1:  # y coordinate
                                gain[idx] = torch.tensor(p[i].shape)[2]  # grid_h
                            # For d >= 2 (visibility), keep gain as 1 (no scaling needed)
            else:
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            if self.kpt_label:
                for kpt in range(self.nkpt):
                    # Process keypoint coordinates (x,y) for grid adjustment
                    kpt_start_idx = 6 + kpt * self.kpt_dim
                    kpt_xy_end_idx = kpt_start_idx + min(
                        2, self.kpt_dim
                    )  # Only process x,y coordinates
                    if kpt_xy_end_idx <= 6 + self.no_kpt:
                        # Subtract grid offset for x,y coordinates only
                        xy_slice = t[:, kpt_start_idx:kpt_xy_end_idx]
                        mask = xy_slice != 0
                        xy_slice[mask] -= gij[mask]
                tkpt.append(t[:, 6 : 6 + self.no_kpt])  # Extract all keypoint data
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, tkpt, indices, anch
