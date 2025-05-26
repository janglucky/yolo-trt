from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import ALPHA, CLASSES_SEG, COLORS, MASK_COLORS
from models.torch_utils import seg_postprocess
from models.utils import blob, letterbox, path_to_list

def decode_bboxes(pred_boxes, anchors):
    """
    解码边界框回归参数到实际的边界框坐标
    pred_boxes: 模型预测的边界框回归参数
    anchors: 先验框
    返回解码后的边界框坐标（x1, y1, x2, y2）
    """
    # 这里是一个简化的解码示例，实际解码方式取决于模型的编码方式
    # 假设锚框的编码方式是基于中心点坐标、宽度和高度的对数空间编码
    # 对预测的边界框参数进行解码，得到实际的边界框在图像中的坐标范围
    # pred_boxes shape: (num_anchors, 4)
    # anchors shape: (num_anchors, 4)

    # 计算边界框的中心坐标、宽度和高度
    pred_x = anchors[:, 0] + pred_boxes[:, 0]
    pred_y = anchors[:, 1] + pred_boxes[:, 1]
    pred_w = np.exp(pred_boxes[:, 2]) * anchors[:, 2]
    pred_h = np.exp(pred_boxes[:, 3]) * anchors[:, 3]

    # 转换为边界框的左上角和右下角坐标
    x1 = pred_x - pred_w / 2
    y1 = pred_y - pred_h / 2
    x2 = pred_x + pred_w / 2
    y2 = pred_y + pred_h / 2

    return np.stack([x1, y1, x2, y2], axis=-1)

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['output0', 'output1'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))

        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, seg_img = blob(rgb, return_seg=True)

        
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)

        # inference
        data = Engine(tensor)

        seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]],
                                device=device)
        bboxes, scores, labels, masks = seg_postprocess(
            data, bgr.shape[:2], args.conf_thres, args.iou_thres)
        if bboxes.numel() == 0:
            # if no bounding box
            print(f'{image}: no object!')
            continue
        masks = masks[:, dh:H - dh, dw:W - dw, :]
        indices = (labels % len(MASK_COLORS)).long()
        mask_colors = torch.asarray(MASK_COLORS, device=device)[indices]
        mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                          draw.shape[:2][::-1])

        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES_SEG[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.3,
                        help='Confidence threshold')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)