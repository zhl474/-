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

add_service_files(FILES
  PrepareTask.srv
  GetTaskTarget.srv
  DetectBlockOffset.srv
  DetectBoardOffset.srv
)
generate_messages()

catkin_package(CATKIN_DEPENDS camera cv_bridge message_runtime rospy sensor_msgs)

catkin_install_python(PROGRAMS scripts/process.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
