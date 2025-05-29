import onnxruntime
import onnx
import cv2
import numpy as np
import torch
 
 
"# 注意v5-7 的训练使用letterbox预处理缩放图片的时候是没用自动化pad的， 即函数内的auto=False，但是在推理的时候却用了自动化pad, 既这时候函数内的auto=True, 因此onnx 推理也是用的auto=True"
 
 
 
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']  # coco80类别
 
class YOLOV5():
    def __init__(self, onnxpath):
        # ==============  指定先用gpu， 没有gpu 则使用cpu ====================
        # 创建 session_options
        self.session_options = onnxruntime.SessionOptions()
        self.onnx_session = onnxruntime.InferenceSession(onnxpath, self.session_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
 
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
    # 获取模型输入的名称，并创建字典。 在ONNX Runtime 中进行模型推理时，不能直接将原始图片数据作为输入传递给模型。因为模型期望的输入数据通常具有特定的格式、维度和数据类型，这些要求是由模型的训练和转换过程决定的。get_input_feed方法的作用正是为了准备符合模型要求的输入数据。
    def get_input_name(self):
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    # 得到onnx 模型输出节点
    def get_output_name(self):
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
    #-------------------------------------------------------
	#   输入图像
	#-------------------------------------------------------
    def get_input_feed(self,img_tensor):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = img_tensor
 
        return input_feed
    #-------------------------------------------------------
	#   1.cv2读取图像并resize
	#	2.图像转BGR2RGB和HWC2CHW
	#	3.图像归一化
	#	4.图像增加维度
	#	5.onnx_session 推理
	#-------------------------------------------------------
    def inference(self, img_path):
        org_img = cv2.imread(img_path)   # hwc
        # 图片等比缩放
        pad_img, r, (dw, dh) = letterbox(org_img, (640, 640))
        img = pad_img[:, :, ::-1].transpose(2, 0, 1)  # BGR 2 RGB 和 HWC 2 CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
 
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, org_img, pad_img
 
 
### 将中心点坐标转换为左上角右下角坐标
def xywh2xyxy(x):
    # [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
 
 
#dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
#thresh: 阈值
def nms(dets, thresh):                           # 非极大值抑制
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    #-------------------------------------------------------
	#   计算框的面积
    #	置信度从大到小排序
	#-------------------------------------------------------
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
 
    while index.size > 0:
        i = index[0]
        keep.append(i)
		#-------------------------------------------------------
        #   计算相交面积
        #	1.相交
        #	2.不相交
        #-------------------------------------------------------
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
 
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
 
        overlaps = w * h
        #-------------------------------------------------------
        #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
        #	IOU小于thresh的框保留下来
        #-------------------------------------------------------
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep
 
 
 
### 根据置信度过滤无用框
def filter_box(org_box,conf_thres,iou_thres): #过滤掉无用的框
    #-------------------------------------------------------
	#   删除为1的维度
    #	删除置信度小于conf_thres的BOX
	#-------------------------------------------------------
    org_box=np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    box = org_box[conf == True]
    #-------------------------------------------------------
    #	通过argmax获取置信度最大的类别
	#-------------------------------------------------------
    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))
 
    # -------------------------------------------------------
    #   分别对每个类别进行过滤
    #	1.将第6列元素替换为类别下标
    #	2.xywh2xyxy 坐标转换
    #	3.经过非极大抑制后输出的BOX下标
    #	4.利用下标取出非极大抑制后的BOX
    # ------------------------------------------------------
    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []
        curr_out_box = []
        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])
        curr_cls_box = np.array(curr_cls_box)
        # curr_cls_box_old = np.copy(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = nms(curr_cls_box, iou_thres)
        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    return output
 
## 画图
def draw(image: object, box_data: object) -> object:
    #-------------------------------------------------------
    #	取整，方便画框
	#-------------------------------------------------------
    boxes=box_data[...,:4].astype(np.int32)
    scores=box_data[...,4]
    classes=box_data[...,5].astype(np.int32)
 
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        print()
 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
 
 
""""#图片缩放成640 x 640  ======> img：原始图片。   # new_shape：目标尺寸，默认为 640x640。      # color：边框填充的颜色，默认为灰色 (114, 114, 114)。
# auto：是否自动调整，使边缘填充的宽高是 32 的倍数,(这里选择False ， 这样处理完的图片就是640 x 640, 否则最长边是640 但是最短边会自动缩放成32的倍数，这样最短边就不会填充成640 了)。
 # scaleFill：如果为 True，图像将直接拉伸到目标大小，而不是保持长宽比。       # scaleup：如果为 False，则只缩小图片，不放大图片，以避免损失质量。"""
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle
    shape = img.shape[:2]  #[h, w]
 
    # 确保 new_shape 变量始终是一个包含两个元素的元组 (width, height)，即图片的目标尺寸
    # 如果 new_shape 被传入为 640（一个整数），这行代码会将 new_shape 变为 (640, 640)，表示目标尺寸是 640x640 的正方形图片。如果 new_shape 原本就是一个元组，如 (640, 480)，则不会执行这行代码，因为它已经是一个元组。
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old) # == 这两行代码的目的是计算图片的缩放比例 r，并确保图片只会缩小而不会放大
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # if not scaleup: （如果scaleu不是True 即 如果scaleu是False），也就是在 scaleup 为 False 的情况下执行代码。
        r = min(r, 1.0)
 
    # 等比缩放图像 。 round() 将计算出的浮点数尺寸四舍五入为最接近的整数，确保尺寸是整数像素值。
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # 计算定义的模型输入尺寸与图片等比例缩放后的图片尺寸之间的差值，—（即计算图像在宽度和高度方向上需要的填充量（即多余的部分），dw 表示在宽度上的填充，dh 表示在高度上的填充）目的是为了后面把不是正方形的等比例缩放图片变成正方形图片
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # pad to the nearest 32-pixel multiples
    elif scaleFill:  # stretch
        dw, dh = 0, 0
        new_unpad = new_shape
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    # 判断原始图片尺寸是否等于等比例缩放尺寸
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)
 
# 映射信息到原图
def scale_boxes(pad_img_shape, boxes, ori_img_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(pad_img_shape[0] / ori_img_shape[0], pad_img_shape[1] / ori_img_shape[1])  # gain  = old / new
        pad = (pad_img_shape[1] - ori_img_shape[1] * gain) / 2, (pad_img_shape[0] - ori_img_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
 
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, ori_img_shape)
    return boxes
 
def clip_boxes(boxes, shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
 
 
if __name__ == '__main__':
    onnx_path = 'yolov5s.onnx'
    model = YOLOV5(onnx_path)
    # 模型输出， 缩放后的原始图片
    output, or_img, pad_img = model.inference('1.jpeg')
 
    # 根据置信度过滤无用框   ， 这里的outbox是预处理后的图片上的坐标 ====> outbox [[609.1487     195.49065    699.1415     365.24677      0.9417041     0. ]
    outbox = filter_box(output, conf_thres=0.5, iou_thres=0.5)
 
    ###########  ====== =========在预处理图上画框并显示 ====================
    draw(pad_img, outbox)
    # cv2.imshow('88', pad_img)
    # cv2.waitKey(0)

    cv2.imwrite('res.png', pad_img)
 
 
    #########  ================ 把预处理图片上检测的框映射到原图上 并画框显示。执行这段代码后的outbox的坐标已经返回到原图上了 =====================
    # outbox[:, :4] = scale_boxes(pad_img.shape, outbox[:, :4], or_img.shape).round()     # pad_img.shape == (508, 892, 3),        or_img.shape == (1440, 2560, 3)
    # print("outbox", outbox)
    # draw(or_img, outbox)
    # cv2.imshow('88', or_img)
    # cv2.waitKey(0)
 
 
 