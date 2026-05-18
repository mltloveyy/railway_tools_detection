cmake_minimum_required(VERSION 3.10)
project(MNN_YOLO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} src/main.cc src/inference.cc src/result.cc)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib/x86)

target_link_libraries(${PROJECT_NAME} MNN MNN_Express MNNOpenCV jsoncpp yaml-cpp)