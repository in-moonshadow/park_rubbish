#include "ros/ros.h"
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "CNN.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unordered_set>  // 用于记录已保存的类别ID

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "yolov11_camera_node");
    ros::NodeHandle nh;

    // 创建检测结果发布者
    ros::Publisher det_pub = nh.advertise<vision_msgs::Detection2DArray>("/yolo_detections", 10);

    // 加载模型
    std::string OnnxFile = "/home/wheeltec/park/src/park/models/park.onnx";
    std::string SaveTrtFilePath = "/home/wheeltec/park/src/park/models/5_6s.trt";
    CNN YOLO(OnnxFile, SaveTrtFilePath, 1, 3, 640, 640);

    // 打开摄像头
    cv::VideoCapture cap(0, cv::CAP_V4L2); // 使用 V4L2 后端
    if (!cap.isOpened())
    {
        ROS_ERROR("Cannot open camera");
        return -1;
    }

    // 设置摄像头分辨率为 640x480
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 检查当前分辨率
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (width != 640 || height != 480)
    {
        ROS_WARN("Failed to set camera resolution to 640x480. Current resolution is %dx%d", width, height);
    }
    else
    {
        ROS_INFO("Camera resolution set to: %dx%d", width, height);
    }

    // 用于记录已保存过图像的类别ID
    std::unordered_set<int> saved_classes;

    // 检测结果保存路径
    const std::string output_image_dir = "/home/wheeltec/park/src/park/images/";

    ros::Rate loop_rate(15);

    while (ros::ok())
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            ROS_WARN("Empty frame grabbed");
            ros::spinOnce();
            loop_rate.sleep();
            continue;
        }

        int img_width = frame.cols;
        int img_height = frame.rows;

        // 推理
        YOLO.Inference(frame);

        // 构造检测结果消息
        vision_msgs::Detection2DArray det_array_msg;
        det_array_msg.header.stamp = ros::Time::now();
        det_array_msg.header.frame_id = "camera";

        // 先查询当前是否有订阅者
        bool has_subscribers = (det_pub.getNumSubscribers() > 0);

        for (size_t i = 0; i < YOLO.DetectiontRects_.size(); i += 6)
        {
            int classId = static_cast<int>(YOLO.DetectiontRects_[i + 0]);
            float conf = YOLO.DetectiontRects_[i + 1];
            int xmin = static_cast<int>(YOLO.DetectiontRects_[i + 2] * img_width + 0.5);
            int ymin = static_cast<int>(YOLO.DetectiontRects_[i + 3] * img_height + 0.5);
            int xmax = static_cast<int>(YOLO.DetectiontRects_[i + 4] * img_width + 0.5);
            int ymax = static_cast<int>(YOLO.DetectiontRects_[i + 5] * img_height + 0.5);

            // 只有当检测结果发布者有订阅者，并且该类别尚未保存过图像时，才保存
            if (has_subscribers && saved_classes.find(classId) == saved_classes.end())
            {
                cv::Mat frame_with_boxes = frame.clone();
                char text[64];
                snprintf(text, sizeof(text), "%d:%.2f", classId, conf);
                cv::rectangle(frame_with_boxes, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
                cv::putText(frame_with_boxes, text, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

                std::string output_image_path = output_image_dir + std::to_string(classId) + ".jpg";

                if (cv::imwrite(output_image_path, frame_with_boxes))
                {
                    ROS_INFO("Saved first detection image for class %d to %s", classId, output_image_path.c_str());
                    saved_classes.insert(classId);
                }
                else
                {
                    ROS_ERROR("Failed to save detection image for class %d to %s", classId, output_image_path.c_str());
                }
            }

            vision_msgs::Detection2D det_msg;
            geometry_msgs::Pose2D bbox_pose;
            bbox_pose.x = (xmin + xmax) / 2.0;
            bbox_pose.y = (ymin + ymax) / 2.0;
            bbox_pose.theta = 0.0;

            det_msg.bbox.center = bbox_pose;
            det_msg.bbox.size_x = xmax - xmin;
            det_msg.bbox.size_y = ymax - ymin;

            vision_msgs::ObjectHypothesisWithPose hypo;
            hypo.id = classId;
            hypo.score = conf;
            det_msg.results.push_back(hypo);

            det_array_msg.detections.push_back(det_msg);
        }

        // 发布检测结果
        det_pub.publish(det_array_msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    // 释放摄像头资源
    cap.release();
    return 0;
}
