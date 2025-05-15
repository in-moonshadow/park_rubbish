cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# CUDA include and link dirs
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Tensorrt
include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找ROS包，注意替换为你使用的ROS版本，比如noetic、melodic
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  vision_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(yolov5 yolov5_det.cpp ${SRCS} ${PLUGIN_SRCS})

target_link_libraries(yolov5
  nvinfer
  cudart
  myplugins
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)
