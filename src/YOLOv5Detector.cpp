#include "YOLOv5Detector.h"

// Initialize static members
Logger YOLOv5Detector::m_logger;

YOLOv5Detector::YOLOv5Detector() 
    : m_runtime(nullptr), m_engine(nullptr), m_context(nullptr),
      m_gpuInputBuffer(nullptr), m_gpuOutputBuffer(nullptr), m_cpuOutputBuffer(nullptr) {
    // Initialize CUDA device
    cudaSetDevice(kGpuId);
    CUDA_CHECK(cudaStreamCreate(&m_cudaStream));
}

YOLOv5Detector::~YOLOv5Detector() {
    // Free CUDA resources
    if (m_cudaStream) cudaStreamDestroy(m_cudaStream);
    if (m_gpuInputBuffer) CUDA_CHECK(cudaFree(m_gpuInputBuffer));
    if (m_gpuOutputBuffer) CUDA_CHECK(cudaFree(m_gpuOutputBuffer));
    if (m_cpuOutputBuffer) delete[] m_cpuOutputBuffer;
    
    // Free TensorRT resources
    if (m_context) m_context->destroy();
    if (m_engine) m_engine->destroy();
    if (m_runtime) m_runtime->destroy();
    
    // Clean up preprocessor
    cuda_preprocess_destroy();
}

void YOLOv5Detector::prepareBuffers() {
    assert(m_engine->getNbBindings() == 2);
    const int inputIndex = m_engine->getBindingIndex(kInputTensorName);
    const int outputIndex = m_engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    
    // Allocate GPU buffers
    CUDA_CHECK(cudaMalloc((void**)&m_gpuInputBuffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&m_gpuOutputBuffer, kBatchSize * m_outputSize * sizeof(float)));
    
    // Allocate CPU output buffer
    m_cpuOutputBuffer = new float[kBatchSize * m_outputSize];
}

void YOLOv5Detector::deserializeEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error reading engine file: " << enginePath << std::endl;
        assert(false);
    }
    
    // Read engine file
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    file.read(serializedEngine, size);
    file.close();

    // Create runtime, engine and context
    m_runtime = createInferRuntime(m_logger);
    m_engine = m_runtime->deserializeCudaEngine(serializedEngine, size);
    m_context = m_engine->createExecutionContext();
    
    delete[] serializedEngine;
}

bool YOLOv5Detector::initialize(const std::string& enginePath) {
    // Check if engine file path is valid
    if (enginePath.empty()) {
        std::cerr << "Error: Engine file path cannot be empty" << std::endl;
        return false;
    }
    
    // Initialize CUDA preprocessor
    cuda_preprocess_init(kMaxInputImageSize);
    
    // Load TensorRT engine
    deserializeEngine(enginePath);
    
    // Prepare GPU and CPU buffers
    prepareBuffers();
    
    return true;
}

std::vector<std::string> YOLOv5Detector::readClassNames(const std::string& filename) {
    std::vector<std::string> class_names;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open class names file: " << filename << std::endl;
        return class_names; // Return empty vector
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_names.push_back(line);
        }
    }

    file.close();
    return class_names;
}

void YOLOv5Detector::infer(void** gpuBuffers, float* output, int batchsize) {
    m_context->enqueue(batchsize, gpuBuffers, m_cudaStream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpuBuffers[1], batchsize * m_outputSize * sizeof(float), 
                              cudaMemcpyDeviceToHost, m_cudaStream));
    cudaStreamSynchronize(m_cudaStream);
}

std::vector<std::vector<Detection>> YOLOv5Detector::detect(cv::Mat& frame, const std::vector<std::string>& class_names) {
    // Resize frame
    cv::resize(frame, frame, cv::Size(640, 480));
    
    // Preprocess
    std::vector<cv::Mat> cvImage = {frame};
    cuda_batch_preprocess(cvImage, m_gpuInputBuffer, kInputW, kInputH, m_cudaStream);
    
    // Run inference
    void* gpuBuffers[2] = {m_gpuInputBuffer, m_gpuOutputBuffer};
    infer(gpuBuffers, m_cpuOutputBuffer, 1);
    
    // Post-process
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, m_cpuOutputBuffer, 1, m_outputSize, kConfThresh, kNmsThresh);
    
    // Draw bounding boxes
    draw_bbox(cvImage, res_batch, class_names);
    frame = cvImage[0];
    
    return res_batch;
}