from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms
import torchvision

from .utils import obb_postprocess as np_obb_postprocess


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros(x.shape).to(x.device)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    return y

def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0].view(-1, h*w)

    cls_num = outputs.shape[0] - 36
    bboxes, labels, maskconf = outputs.T.split([4, cls_num, 32], 1)
    bboxes = xywh2xyxy(bboxes)

    scores, labels = labels.max(1, keepdim=False)

    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, )), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    
    idx = torchvision.ops.nms(bboxes, scores, iou_threshold=iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def pose_postprocess(
        data: Union[Tuple, Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = torch.transpose(data[0], 0, 1).contiguous()
    bboxes, scores, kpts = outputs.split([4, 1, 51], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), bboxes.new_zeros((0, 0, 0))
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = bboxes.chunk(2, -1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], -1)
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    return bboxes, scores, kpts.reshape(idx.shape[0], -1, 3)


def decode(output, anchors, grid_size):
    # 获取预测的偏移量和尺寸调整量
    delta_x = output[..., 0]
    delta_y = output[..., 1]

    delta_w = output[..., 2]
    delta_h = output[..., 3]

    box_confidence = output[...,4].sigmoid()
    class_probs = output[...,5:].sigmoid()

    x_anchor, y_anchor = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
    x_anchor = x_anchor.unsqueeze(0).unsqueeze(0).float()
    y_anchor = y_anchor.unsqueeze(0).unsqueeze(0).float()

    print(x_anchor.shape, delta_x.shape)
    # 计算实际的边界框坐标和尺寸
    x = x_anchor + delta_x
    y = y_anchor + delta_y

    w = anchors[:, 0].unsqueeze(-1).unsqueeze(-1) * torch.exp(delta_w)
    h = anchors[:, 1].unsqueeze(-1).unsqueeze(-1) * torch.exp(delta_h)

    # print(anchors.shape)
    # print(w.shape)
    # print(h.shape)
    # 计算边界框的左上角和右下角坐标
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    # print(pred.shape)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return bboxes, box_confidence, class_probs

def yolov5_poseprocess(preds, anchors, grid_sizes, score_thr=0.69):
    bbox = []
    score = []
    label = []
    for pred, anchor, grid_size  in zip(preds, anchors, grid_sizes):
        pred = pred.cpu().detach()
        anchor = torch.Tensor(anchor)
        bboxes, bbox_confidences, class_probs = decode(pred, anchor, grid_size)
        class_scores, labels =  class_probs.max(dim=-1)
        scores =  class_scores * bbox_confidences

        score_id = scores > score_thr
        bbox.append(bboxes[score_id].view(-1,4))
        label.append(labels[score_id].view(-1))
        score.append(scores[score_id].view(-1))

    # pred
    bbox = torch.cat(bbox)
    label = torch.cat(label)
    score = torch.cat(score)
    nms_indices = torchvision.ops.nms(bbox, score, 0.8)

    return bbox[nms_indices], score[nms_indices], label[nms_indices]

def det_postprocess(preds, score_thr=0.69):
    preds = preds.squeeze().permute(1,0)
    decoded_bboxes = preds[:, :4]
    decoded_bboxes = xywh2xyxy(decoded_bboxes)
    cls_scores = preds[:,4:]
    cls_score_id = cls_scores.amax(1) > score_thr

    ths_scores = cls_scores[cls_score_id,:] # torch.szie[num,80]
    thr_bboxes = decoded_bboxes[cls_score_id,:] # torch.szie[num,4]

    ths_conf, thr_cls = ths_scores.max(1, keepdim=False) #torch.szie[num],torch.szie[num]
    keep_idxs = torchvision.ops.nms(thr_bboxes, ths_conf, iou_threshold=0.6)
    scores = ths_conf[keep_idxs].cpu().detach()
    labels = thr_cls[keep_idxs].cpu().detach()
    bboxes = thr_bboxes[keep_idxs].cpu().detach()
    bboxes[:, 0::2] = torch.clip(bboxes[:, 0::2], 0, 640).cpu().detach()
    bboxes[:, 1::2] = torch.clip(bboxes[:, 1::2], 0, 640).cpu().detach()
    return bboxes, scores, labels


def obb_postprocess(
        data: Union[Tuple, Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    device = data.device
    points, scores, labels = np_obb_postprocess(data.cpu().numpy(), conf_thres,
                                                iou_thres)
    return torch.from_numpy(points).to(device), torch.from_numpy(scores).to(
        device), torch.from_numpy(labels).to(device)


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))