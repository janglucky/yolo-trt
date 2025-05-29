from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import time
import cv2
import torch
import torchvision
import numpy as np
from config import CLASSES_DET, COLORS
from models.torch_utils import yolov5_poseprocess
from models.utils import blob, letterbox, path_to_list 


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

anchors = [
  [[10,13], [16,30], [33,23]],  # P3/8
  [[30,61], [62,45], [59,119]],  # P4/16
  [[116,90], [156,198], [373,326]]  # P5/32
]

grid_sizes = [80 ,40, 20]

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
        bboxes, scores, ids = yolov5_poseprocess(data, anchors, grid_sizes)

        dwdh = dwdh.cpu().int().tolist()

        for bbox,id,score in zip(bboxes,ids, scores):
            cls = CLASSES_DET[int(id.item())]
            bbox = bbox.round().int().tolist()

            print(bbox)
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (x1 - dwdh[0]) / ratio, (y1 - dwdh[1]) / ratio, (x2 - dwdh[0])/ ratio, (y2- dwdh[1]) / ratio
            color = COLORS[cls]

            text = f'{cls}:{score:.3f}'
            (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8, 1)
            _y1 = min(y1 + 1, draw.shape[0])

            cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.rectangle(draw, (int(x1), int(_y1)), (int(x1 + _w), int(_y1 + _h + _bl)),
                          (0, 0, 255), -1)
            cv2.putText(draw, text, (int(x1), int(_y1 + _h)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)


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