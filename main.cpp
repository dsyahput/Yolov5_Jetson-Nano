#include "YOLOv5Detector.h"
#include <fstream>

int main(int argc, char** argv) {

    YOLOv5Detector detector;

    const std::string MODEL_PATH = "../models/yolov5n.engine";
    const std::string CLASS_NAMES_FILE = "../models/coco.names";

    std::vector<std::string> class_names = detector.readClassNames(CLASS_NAMES_FILE);
    
    if (class_names.empty()) {
        std::cerr << "Failed to load class names from: " << CLASS_NAMES_FILE << std::endl;
        return -1;
    }

    if (!detector.initialize(MODEL_PATH)) {
        std::cerr << "Failed to initialize detector with model: " << MODEL_PATH << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully from: " << MODEL_PATH << std::endl;

    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video!" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (cap.read(frame)) {

        std::vector<std::vector<Detection>> detections = detector.detect(frame, class_names);

        cv::imshow("Detection", frame);

        // Exit on ESC key
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}