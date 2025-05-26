from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import time
import cv2
import torch
import torchvision
import numpy as np
from config import CLASSES_DET, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros(x.shape).to(x.device)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def postprocess(preds, score_thr=0.69):
    preds = preds.squeeze().permute(1,0)
    decoded_bboxes = preds[:, :4]
    print(decoded_bboxes)
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
    return bboxes, labels, scores


class_name = ["person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"]


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    for image in images:
        save_image = save_path / image.name
        bgr = cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))

        bgr = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)

        tensor = torch.asarray(tensor, device=device)
        # inference
        start = time.time()
        data = Engine(tensor)
        end = time.time()
        print(f"cost {(end - start)*1000}ms")
        bboxes, ids, scores = postprocess(data)

        dwdh = dwdh.cpu().int().tolist()

        for bbox,id,score in zip(bboxes,ids, scores):
            cls = class_name[int(id.item())]
            bbox = bbox.round().int().tolist()
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (x1 - dwdh[0]) / ratio, (y1 - dwdh[1]) / ratio, (x2 - dwdh[0])/ ratio, (y2- dwdh[1]) / ratio
            color = COLORS[cls]

            cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


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
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)