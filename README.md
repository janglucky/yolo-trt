# yolo-trt
YOLO inference demo with TensorRT

# Get Started
## convert onnx to TensrRT format
```shell
trtexec \
--onnx=yolov8.onnx \
--saveEngine=yolov8.engine \
--fp16
```
## Python DEMO
### Detection
Run in commond line:
```shell
python infer_yolov8-det.py \
--engine yolov8.onnx \
--imgs 1.jpg \
--out-dir ./output
```
### Segmentation
Run in commond line:
```shell
python infer_yolov8-seg.py \
--engine yolov8.onnx \
--imgs 1.jpg \
--out-dir ./output
```
### Pose
Run in commond line:
```shell
python infer_yolov8-pose.py \
--engine yolov8.onnx \
--imgs 1.jpg \
--out-dir ./output
```
## C++ DEMO