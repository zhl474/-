cmake_minimum_required(VERSION 3.0.2)
project(competition)

find_package(catkin REQUIRED COMPONENTS
  control
  image_process
  rospy
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS control image_process rospy)

catkin_install_python(PROGRAMS scripts/competition.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
