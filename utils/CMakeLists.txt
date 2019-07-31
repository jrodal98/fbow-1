# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src /home/rodalj/Develop/fbow/utils /home/mjmarin/libs/ORB_SLAM2/include)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
LINK_LIBRARIES(${PROJECT_NAME} ${PROTOBUF_LIBRARY})


ADD_EXECUTABLE(fbow_create_voc_step0 fbow_create_voc_step0.cpp dirreader.h)
ADD_EXECUTABLE(fbow_create_voc_step0_list fbow_create_voc_step0_list.cpp)
ADD_EXECUTABLE(fbow_create_voc_step1 fbow_create_voc_step1.cpp)
#ADD_EXECUTABLE(fbow_create_voc_step1_opencv fbow_create_voc_step1_opencv.cpp)
ADD_EXECUTABLE(fbow_transform fbow_transform.cpp)
ADD_EXECUTABLE(image_matching image_matching.cpp)
ADD_EXECUTABLE(create_fbow create_feature_file.cpp agent_brain.pb.h agent_brain.pb.cc )
INSTALL(TARGETS  fbow_create_voc_step0    fbow_create_voc_step1 fbow_transform RUNTIME DESTINATION bin)
