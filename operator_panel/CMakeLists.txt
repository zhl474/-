cmake_minimum_required(VERSION 3.0.2)
project(operator_panel)

find_package(catkin REQUIRED COMPONENTS
  camera
  competition
  control
  cv_bridge
  image_process
  rosgraph
  rosgraph_msgs
  roslaunch
  rosnode
  rospy
  rosservice
  sensor_msgs
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  camera competition control cv_bridge image_process rosgraph rosgraph_msgs
  roslaunch rosnode rospy rosservice sensor_msgs
)

catkin_install_python(PROGRAMS
  scripts/operator_panel.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/start_operator_panel.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config desktop static templates
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
