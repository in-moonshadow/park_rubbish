#include <ros/ros.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <geometry_msgs/Pose2D.h>

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <unordered_set>  // 用于记录已保存的类别ID
using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const std::string ENGINE_PATH = "/home/wheeltec/park/src/yolov5/park.engine"; // 引擎文件路径


void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}

void deserialize_engine(const std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

// 新增一个辅助函数，放到 main 函数之前或合适的位置
static cv::Rect get_rect(const cv::Mat& img, const float bbox[4], float scale, float pad_w, float pad_h) {
    // bbox中心点+宽高格式，坐标在模型输入尺度(kInputW x kInputH)
    float left = bbox[0] - bbox[2] / 2.0f;
    float top = bbox[1] - bbox[3] / 2.0f;
    float right = bbox[0] + bbox[2] / 2.0f;
    float bottom = bbox[1] + bbox[3] / 2.0f;

    // 去除padding，除以缩放，映射回原图坐标
    left = (left - pad_w) / scale;
    top = (top - pad_h) / scale;
    right = (right - pad_w) / scale;
    bottom = (bottom - pad_h) / scale;

    // 限制到图像边界
    left = std::max(0.0f, std::min(left, (float)(img.cols - 1)));
    top = std::max(0.0f, std::min(top, (float)(img.rows - 1)));
    right = std::max(0.0f, std::min(right, (float)(img.cols - 1)));
    bottom = std::max(0.0f, std::min(bottom, (float)(img.rows - 1)));

    return cv::Rect(cv::Point(static_cast<int>(left + 0.5f), static_cast<int>(top + 0.5f)),
                    cv::Point(static_cast<int>(right + 0.5f), static_cast<int>(bottom + 0.5f)));
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "yolov5_detector");
    ros::NodeHandle nh;

    ros::Publisher detection_pub = nh.advertise<vision_msgs::Detection2DArray>("yolov5/detections", 10);

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    cudaStream_t stream;
    float* gpu_buffers[2] = {nullptr, nullptr};
    float* cpu_output_buffer = nullptr;

    cudaSetDevice(kGpuId);
    deserialize_engine(ENGINE_PATH, &runtime, &engine, &context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        ROS_ERROR("Failed to open camera device /dev/video0");
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    ROS_INFO("Start capturing and detecting...");
    // 用于记录已保存过图像的类别ID
    std::unordered_set<int> saved_classes;

    // 检测结果保存路径
    const std::string output_image_dir = "/home/wheeltec/park/src/yolov5/images/";

    ros::Rate loop_rate(15);

    while (ros::ok()) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            ROS_ERROR("Failed to read frame from camera");
            break;
        }
        if (frame.empty() || frame.channels() != 3) {
            ROS_WARN("Invalid frame captured.");
            continue;
        }

        std::vector<cv::Mat> img_batch = {frame};
        cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

        auto start = std::chrono::steady_clock::now();
        infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
        auto end = std::chrono::steady_clock::now();

        int infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kMaxNumOutputBbox, kConfThresh, kNmsThresh);

        vision_msgs::Detection2DArray detection_array_msg;
        detection_array_msg.header.stamp = ros::Time::now();
        detection_array_msg.header.frame_id = "camera_frame";

        // 先查询当前是否有订阅者
        bool has_subscribers = (detection_pub.getNumSubscribers() > 0);

        if (!res_batch.empty()) {
	    cv::Mat frame_with_boxes = frame.clone();

	    // 计算缩放比例和padding（根据预处理letterbox规则）
	    int orig_w = frame.cols;
	    int orig_h = frame.rows;
	    float r_w = kInputW / (float)orig_w;
	    float r_h = kInputH / (float)orig_h;
	    float scale = std::min(r_w, r_h);
	    float pad_w = (kInputW - orig_w * scale) / 2.0f;
	    float pad_h = (kInputH - orig_h * scale) / 2.0f;

	    for (const auto& det : res_batch[0]) {
		int classId = static_cast<int>(det.class_id);

		cv::Rect box = get_rect(frame, det.bbox, scale, pad_w, pad_h);

		float conf = det.conf;

		if (has_subscribers && saved_classes.find(classId) == saved_classes.end()) {
		    char text[64];
		    snprintf(text, sizeof(text), "%d:%.2f", classId, conf);

		    cv::rectangle(frame_with_boxes, box, cv::Scalar(255, 0, 0), 2);
		    cv::putText(frame_with_boxes, text, cv::Point(box.x, box.y + 15),
		                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

		    std::string output_image_path = output_image_dir + std::to_string(classId) + ".jpg";

		    if (cv::imwrite(output_image_path, frame_with_boxes)) {
		        ROS_INFO("Saved first detection image for class %d to %s", classId, output_image_path.c_str());
		        saved_classes.insert(classId);
		    } else {
		        ROS_ERROR("Failed to save detection image for class %d to %s", classId, output_image_path.c_str());
		    }
		}

		// 填充Detection2D消息，坐标仍用模型输入坐标系的bbox，保持一致
		vision_msgs::Detection2D detection_msg;
		detection_msg.bbox.center.x = det.bbox[0];
		detection_msg.bbox.center.y = det.bbox[1];
		detection_msg.bbox.size_x = det.bbox[2];
		detection_msg.bbox.size_y = det.bbox[3];

		vision_msgs::ObjectHypothesisWithPose hypothesis;
		hypothesis.id = classId;
		hypothesis.score = conf;

		detection_msg.results.push_back(hypothesis);
		detection_array_msg.detections.push_back(detection_msg);
	    }
	}

        detection_pub.publish(detection_array_msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    // 释放资源
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();

    context->destroy();
    engine->destroy();
    runtime->destroy();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


