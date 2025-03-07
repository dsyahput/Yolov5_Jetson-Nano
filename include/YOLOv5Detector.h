#ifndef YOLOV5_DETECTOR_H
#define YOLOV5_DETECTOR_H

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

using namespace nvinfer1;

class YOLOv5Detector {
private:
    // TensorRT components
    IRuntime* m_runtime;
    ICudaEngine* m_engine;
    IExecutionContext* m_context;
    
    // CUDA components
    cudaStream_t m_cudaStream;
    float* m_gpuInputBuffer;
    float* m_gpuOutputBuffer;
    float* m_cpuOutputBuffer;
    
    // Constants
    static const int m_outputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    static Logger m_logger;
    
    // Methods
    void prepareBuffers();
    void deserializeEngine(const std::string& enginePath);

public:
    YOLOv5Detector();
    ~YOLOv5Detector();
    
    // Initialization
    bool initialize(const std::string& enginePath);
    
    // Core functionalities
    void infer(void** gpuBuffers, float* output, int batchsize);
    std::vector<std::string> readClassNames(const std::string& filename);
    std::vector<std::vector<Detection>> detect(cv::Mat& frame, const std::vector<std::string>& class_names);
    
    // Getters for direct buffer access if needed
    float* getGpuInputBuffer() const { return m_gpuInputBuffer; }
    float* getGpuOutputBuffer() const { return m_gpuOutputBuffer; }
    float* getCpuOutputBuffer() const { return m_cpuOutputBuffer; }
    cudaStream_t getCudaStream() const { return m_cudaStream; }
};

#endif // YOLOV5_DETECTOR_H