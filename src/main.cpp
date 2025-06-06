//
// Created by ubuntu on 1/20/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
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
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::string> OBB_CLASS_NAMES = {"plane",
                                            "ship",
                                            "storage tank",
                                            "baseball diamond",
                                            "tennis court",
                                            "basketball court",
                                            "ground track field",
                                            "harbor",
                                            "bridge",
                                            "large vehicle",
                                            "small vehicle",
                                            "helicopter",
                                            "roundabout",
                                            "soccer ball field",
                                            "swimming pool"};

const std::vector<std::vector<unsigned int>> OBB_COLORS = {{0, 114, 189},
                                                       {217, 83, 25},
                                                       {237, 177, 32},
                                                       {126, 47, 142},
                                                       {119, 172, 48},
                                                       {77, 190, 238},
                                                       {162, 20, 47},
                                                       {76, 76, 76},
                                                       {153, 153, 153},
                                                       {255, 0, 0},
                                                       {255, 128, 0},
                                                       {191, 191, 0},
                                                       {0, 255, 0},
                                                       {0, 0, 255},
                                                       {170, 0, 255}};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {255, 128, 0},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6, 12},
                                                         {7, 13},
                                                         {6, 7},
                                                         {6, 8},
                                                         {7, 9},
                                                         {8, 10},
                                                         {9, 11},
                                                         {2, 3},
                                                         {1, 2},
                                                         {1, 3},
                                                         {2, 4},
                                                         {3, 5},
                                                         {4, 6},
                                                         {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 51, 255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0},
                                                            {0, 255, 0}};

enum TASK{
    DETE,
    SEGM,
    POSE,
    OBB,
    UNKOWN
};

int get_task(std::string t) {
    if(t == "det") {
        return DETE;
    } else if(t == "seg") {
        return SEGM;
    } else if(t == "pose") {
        return POSE;
    } else if(t == "obb") {
        return OBB;
    } else {
        return UNKOWN;
    }
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        fprintf(stderr, "Usage: %s [det/seg/pose] [engine_path] [image_path/image_dir/video_path] [device]\n", argv[0]);
        return -1;
    }

    const int device = atoi(argv[4]);
    // cuda:0
    cudaSetDevice(device);

    const std::string task(argv[1]);
    const std::string engine_file_path{argv[2]};
    const fs::path    path{argv[3]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat             res, image;
    std::vector<Object> objs;

    // cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8->copy_from_Mat(image);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            switch (get_task(task))
            {
            case DETE:
                yolov8->postprocess(objs);
                yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
                break;
            case SEGM:
                yolov8->postprocess_seg(objs);
                yolov8->draw_masks(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
                break;
            case POSE:
                yolov8->postprocess_pose(objs);
                yolov8->draw_poses(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            case OBB:
                yolov8->postprocess_obb(objs);
                yolov8->draw_objects(image, res, objs, OBB_CLASS_NAMES, OBB_COLORS);
                break;
            default:
                break;
            }
           
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8->copy_from_Mat(image);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            switch (get_task(task))
            {
            case DETE:
                yolov8->postprocess(objs);
                yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
                break;
            case SEGM:
                yolov8->postprocess_seg(objs);
                yolov8->draw_masks(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);
                break;
            case POSE:
                yolov8->postprocess_pose(objs);
                yolov8->draw_poses(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
                break;
            case OBB:
                yolov8->postprocess_obb(objs, 0.65, 0.25);
                yolov8->draw_objects(image, res, objs, OBB_CLASS_NAMES, OBB_COLORS);
                break;
            default:
                break;
            }
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            const auto outputName = p.substr(0, p.find_last_of('.')) + "_annotated.jpg";
            cv::imwrite(outputName, res);

        }
    }
    // cv::destroyAllWindows();
    delete yolov8;
    return 0;
}