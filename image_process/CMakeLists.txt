cmake_minimum_required(VERSION 3.0.2)
project(image_process)

find_package(catkin REQUIRED COMPONENTS
  camera
  cv_bridge
  message_generation
  rospy
  sensor_msgs
)

catkin_python_setup()

# 固定盘面 V1 的 Beam 与实体回溯热循环，保持独立 C ABI 供 ctypes 调用。
add_library(task_sequence_optimizer_native SHARED
  native/task_sequence_optimizer_native.cpp
)
target_compile_features(task_sequence_optimizer_native PRIVATE cxx_std_14)
target_compile_options(task_sequence_optimizer_native PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)
set_target_properties(task_sequence_optimizer_native PROPERTIES
  OUTPUT_NAME task_sequence_optimizer_native
)

add_service_files(FILES
  PrepareTask.srv
  GetTaskTarget.srv
  DetectBlockOffset.srv
  DetectBoardOffset.srv
  GetOperatorPrompt.srv
  RespondOperatorPrompt.srv
  YoloPreview.srv
)
generate_messages()

catkin_package(CATKIN_DEPENDS camera cv_bridge message_runtime rospy sensor_msgs)

catkin_install_python(PROGRAMS scripts/process.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(TARGETS task_sequence_optimizer_native
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
