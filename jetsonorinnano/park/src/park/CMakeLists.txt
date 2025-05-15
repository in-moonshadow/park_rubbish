cmake_minimum_required(VERSION 3.5)
project(park LANGUAGES CXX CUDA)

# 设置CUDA工具包路径（Jetson默认路径）
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# 标准配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# CUDA架构设置（Orin: sm_87）
set(CMAKE_CUDA_ARCHITECTURES 87)

# 编译优化参数
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -Xcompiler=-fPIC \
    --default-stream per-thread \
    -use_fast_math")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -O3 \
    -march=native \
    -flto")

# ROS1 catkin
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  vision_msgs  # 添加ROS1的vision_msgs，假设存在
  message_generation  # 如果需要生成消息/服务
)

# 查找CUDA组件
find_package(CUDA REQUIRED COMPONENTS nppial nppig nppidei cublas cudart curand)

# TensorRT配置
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    /usr/include/aarch64-linux-gnu
    REQUIRED
)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/lib/aarch64-linux-gnu
    REQUIRED
)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/lib/aarch64-linux-gnu
    REQUIRED
)

# OpenCV配置
find_package(OpenCV 4.10 REQUIRED)
set(OpenCV_LIBS ${OpenCV_LIBRARIES} CACHE STRING "OpenCV libraries")

# 包含目录
set(COMMON_INCLUDE /home/wheeltec/park/src/park/src/common)
set(KERNELS_INCLUDE /home/wheeltec/park/src/park/src/kernels)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${COMMON_INCLUDE}
  ${KERNELS_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 启用CUDA语言
enable_language(CUDA)

# 定义catkin包
catkin_package(
  INCLUDE_DIRS include ${COMMON_INCLUDE} ${KERNELS_INCLUDE}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs vision_msgs 
  DEPENDS CUDA OpenCV
)

# 可执行文件：yolo_trt
add_executable(yolo_trt
  src/main.cpp
  src/CNN.cpp
  src/postprocess_cuda.cpp
  src/kernels/get_nms_before_boxes.cu
)

# 链接库
target_link_libraries(yolo_trt
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_nppial_LIBRARY}
  ${CUDA_nppig_LIBRARY}
  ${CUDA_nppidei_LIBRARY}
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cudart_LIBRARY}
  ${CUDA_curand_LIBRARY}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  ${catkin_LIBRARIES}  # 确保包含image_transport和cv_bridge
)


# 目标属性
set_target_properties(yolo_trt PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  LINK_FLAGS "-flto"
)

# 安装配置
install(TARGETS yolo_trt 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件
install(DIRECTORY include/ ${COMMON_INCLUDE}/ ${KERNELS_INCLUDE}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

# 安装启动文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
