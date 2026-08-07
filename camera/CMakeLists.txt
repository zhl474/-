cmake_minimum_required(VERSION 3.0.2)
project(camera)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_generation
  rospy
  sensor_msgs
)

add_service_files(FILES
  GetSurfaceHeight.srv
  GetStableWorldPoints.srv
)
generate_messages()

catkin_package(CATKIN_DEPENDS cv_bridge message_runtime rospy sensor_msgs)

catkin_install_python(PROGRAMS scripts/camera_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
