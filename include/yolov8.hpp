//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
using namespace det;
// using namespace cv;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs,float score_thres, float iou_thres, int topk);
    void                 postprocess_pose(std::vector<Object>& objs,float score_thres, float iou_thres, int topk);
    void                 postprocess_seg(std::vector<Object>& objs,float score_thres, float iou_thres, int topk);
    void                 postprocess_obb(std::vector<Object>& objs,float score_thres, float iou_thres, int topk);
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    static void          draw_masks(const cv::Mat&                                image,
                                    cv::Mat&                                      res,
                                    const std::vector<Object>&                    objs,
                                    const std::vector<std::string>&               CLASS_NAMES,
                                    const std::vector<std::vector<unsigned int>>& COLORS,
                                    const std::vector<std::vector<unsigned int>>& MASK_COLORS);
    static void          draw_poses(const cv::Mat&                                 image,
                                    cv::Mat&                                      res,
                                    const std::vector<Object>&                    objs,
                                    const std::vector<std::vector<unsigned int>>& SKELETON,
                                    const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                                    const std::vector<std::vector<unsigned int>>& LIMB_COLORS);
    
    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);

#ifdef TRT_10
    this->num_bindings = this->engine->getNbIOTensors();
#else
    this->num_bindings = this->num_bindings = this->engine->getNbBindings();
#endif

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;
#ifdef TRT_10
        std::string        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name  = this->engine->getBindingName(i);
#endif
        binding.name  = name;
        binding.dsize = type_to_size(dtype);
#ifdef TRT_10
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->num_inputs += 1;
#ifdef TRT_10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        }
        else {
#ifdef TRT_10
            dims = this->context->getTensorShape(name.c_str());
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
#ifdef TRT_10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto& bindings : this->output_bindings) {
        void *d_ptr, *h_ptr;

        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    int      width      = in_binding.dims.d[3];
    int      height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));

#ifdef TRT_10
    auto name = this->input_bindings[0].name.c_str();
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->context->setTensorAddress(name, this->device_ptrs[0]);
#else
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
#endif
}

void YOLOv8::infer()
{
#ifdef TRT_10
    this->context->enqueueV3(this->stream);
#else
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#endif
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8::postprocess_obb(std::vector<Object>& objs, float score_thres = 0.69,float iou_thres=0.85,int topk=100)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];
    auto num_labels   = num_channels - 5;
    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::RotatedRect>    bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;



    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        auto angle_ptr  = row_ptr + 4 + num_labels;

        float score = *max_s_ptr;
        if (score > score_thres) {
            float x = (*bboxes_ptr++ - dw) * ratio;
            float y = (*bboxes_ptr++ - dh) * ratio;
            float w = (*bboxes_ptr++) * ratio;
            float h = (*bboxes_ptr) * ratio;

            if (w < 1.f || h < 1.f) {
                continue;
            }

            x = clamp(x, 0.f, width);
            y = clamp(y, 0.f, height);
            w = clamp(w, 0.f, width);
            h = clamp(h, 0.f, height);

            float angle = *angle_ptr / CV_PI * 180.f;

            cv::RotatedRect bbox;
            bbox.center.x    = x;
            bbox.center.y    = y;
            bbox.size.width  = w;
            bbox.size.height = h;
            bbox.angle       = angle;

            bboxes.push_back(bbox);
            labels.push_back(std::distance(scores_ptr, max_s_ptr));
            scores.push_back(score);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::postprocess_pose(std::vector<Object>& objs, float score_thres = 0.69,float iou_thres=0.85,int topk=100)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::RotatedRect>    bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 5;

        float score = *scores_ptr;
        if (score > score_thres) {
            float x = (*bboxes_ptr++ - dw) * ratio;
            float y = (*bboxes_ptr++ - dh) * ratio;
            float w = (*bboxes_ptr++) * ratio;
            float h = (*bboxes_ptr) * ratio;

            if (w < 1.f || h < 1.f) {
                continue;
            }

            x = clamp(x, 0.f, width);
            y = clamp(y, 0.f, height);
            w = clamp(w, 0.f, width);
            h = clamp(h, 0.f, height);

            cv::RotatedRect bbox;
            bbox.center.x           = x;
            bbox.center.y           = y;
            bbox.size.width         = w;
            bbox.size.height        = h;
            std::vector<float> kps;
            for (int k = 0; k < 17; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(0);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::postprocess_seg(std::vector<Object>& objs, float score_thres = 0.69,float iou_thres=0.85,int topk=100)
{
    objs.clear();
    auto input_h      = this->input_bindings[0].dims.d[2];
    auto input_w      = this->input_bindings[0].dims.d[3];
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];
    auto num_labels   = num_channels - 36;

    std::cout << num_labels << std::endl;   

    auto seg_channels = this->output_bindings[1].dims.d[1];
    auto seg_h = this->output_bindings[1].dims.d[2];
    auto seg_w = this->output_bindings[1].dims.d[3];

    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float>           scores;
    std::vector<int>             labels;
    std::vector<int>             indices;
    std::vector<cv::Mat>         mask_confs;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(this->host_ptrs[1]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        auto maskconf_ptr  = row_ptr + 4 + num_labels;

        float score = *max_s_ptr;
        if (score > score_thres) {
            float x = (*bboxes_ptr++ - dw) * ratio;
            float y = (*bboxes_ptr++ - dh) * ratio;
            float w = (*bboxes_ptr++) * ratio;
            float h = (*bboxes_ptr) * ratio;

            if (w < 1.f || h < 1.f) {
                continue;
            }

            x = clamp(x, 0.f, width);
            y = clamp(y, 0.f, height);
            w = clamp(w, 0.f, width);
            h = clamp(h, 0.f, height);

            cv::RotatedRect bbox;
            bbox.center.x    = x;
            bbox.center.y    = y;
            bbox.size.width  = w;
            bbox.size.height = h;
            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, maskconf_ptr);
            
            mask_confs.push_back(mask_conf);
            bboxes.push_back(bbox);
            labels.push_back(std::distance(scores_ptr, max_s_ptr));
            scores.push_back(score);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    cv::Mat masks;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    if(!masks.empty()) {

        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {seg_h, seg_w});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            
            int center_x = (int)objs[i].rect.center.x;
            int center_y = (int)objs[i].rect.center.y;
            int w        = (int)objs[i].rect.size.width;
            int h        = (int)objs[i].rect.size.height;
            cv::Rect rec(center_x - w/2, center_y - h/2, w, h);
            objs[i].mask = mask(rec) > 0.5f;
        }

    }
}


void YOLOv8::postprocess(std::vector<Object>& objs, float score_thres = 0.69,float iou_thres=0.85,int topk=100)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];
    auto num_labels   = num_channels - 4;

    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float>           scores;
    std::vector<int>             labels;
    std::vector<int>             indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);

        float score = *max_s_ptr;
        if (score > score_thres) {
            float x = (*bboxes_ptr++ - dw) * ratio;
            float y = (*bboxes_ptr++ - dh) * ratio;
            float w = (*bboxes_ptr++) * ratio;
            float h = (*bboxes_ptr) * ratio;


            if (w < 1.f || h < 1.f) {
                continue;
            }

            x = clamp(x, 0.f, width);
            y = clamp(y, 0.f, height);
            w = clamp(w, 0.f, width);
            h = clamp(h, 0.f, height);

            cv::RotatedRect bbox;
            bbox.center.x    = x;
            bbox.center.y    = y;
            bbox.size.width  = w;
            bbox.size.height = h;

            bboxes.push_back(bbox);
            labels.push_back(std::distance(scores_ptr, max_s_ptr));
            scores.push_back(score);
        }
    }

    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::draw_poses(const cv::Mat&                                image,
                       cv::Mat&                                      res,
                       const std::vector<Object>&                    objs,
                       const std::vector<std::vector<unsigned int>>& SKELETON,
                       const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                       const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    res = image.clone();
    const int num_point = 17;
    for (auto& obj : objs) {
        // cv::rectangle(res, obj.rect, {0, 0, 255}, 2);
        cv::Mat points;
        cv::boxPoints(obj.rect, points);
        points.convertTo(points, CV_32S);
        cv::polylines(res, points, true, {0, 0, 255}, 2);

        char text[256];
        sprintf(text, "person %.1f%%", obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.center.x;
        int y = (int)obj.rect.center.y + 1;
        int w = (int)obj.rect.size.width;
        int h = (int)obj.rect.size.height;
        x = x - w / 2;
        y = y - h / 2;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);

        auto& kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++) {
            if (k < num_point) {
                int   kps_x = std::round(kps[k * 3]);
                int   kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto& ske    = SKELETON[k];
            int   pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int   pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
    }
}

void YOLOv8::draw_masks(const cv::Mat&                               image,
                       cv::Mat&                                      res,
                       const std::vector<Object>&                    objs,
                       const std::vector<std::string>&               CLASS_NAMES,
                       const std::vector<std::vector<unsigned int>>& COLORS,
                       const std::vector<std::vector<unsigned int>>& MASK_COLORS)
{
    res = image.clone();
    cv::Mat mask = image.clone();
    for (auto& obj : objs) {
        int idx = obj.label;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);

        cv::Mat points;
        cv::boxPoints(obj.rect, points);
        points.convertTo(points, CV_32S);
        cv::polylines(res, points, true, color, 2);


        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int center_x = (int)obj.rect.center.x;
        int center_y = (int)obj.rect.center.y;
        int w        = (int)obj.rect.size.width;
        int h        = (int)obj.rect.size.height;

        cv::Rect rec(center_x - w/2, center_y - h/2, w, h);
        mask(rec).setTo(mask_color, obj.mask);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = center_x - w / 2;
        int y = center_y + 1 - h / 2;


        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}

void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        // cv::rectangle(res, obj.rect, color, 2);
        // std::cout << "(("<<obj.rect.center.x<<","<< obj.rect.center.y << "),(" << obj.rect.size.width <<","<< obj.rect.size.height <<"),"<<obj.rect.angle <<")"<<std::endl;
        cv::Mat points;
        cv::boxPoints(obj.rect, points);
        points.convertTo(points, CV_32S);
        cv::polylines(res, points, true, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.center.x;
        int y = (int)obj.rect.center.y + 1;
        int w = (int)obj.rect.size.width;
        int h = (int)obj.rect.size.height;

        x = x - w / 2;
        y = y - h / 2;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
#endif  // DETECT_END2END_YOLOV8_HPP