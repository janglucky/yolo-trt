# yolo-trt
YOLO inference demo with TensorRT

## convert onnx to TensrRT format
```shell
trtexec \
--onnx=yolov8.onnx \
--saveEngine=yolov8.engine \
--fp16
```