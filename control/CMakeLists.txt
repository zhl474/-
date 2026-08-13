cmake_minimum_required(VERSION 3.0.2)
project(control)

find_package(catkin REQUIRED COMPONENTS message_generation rospy)

add_service_files(FILES
  ClearArmStop.srv
  GetControlStatus.srv
  MoveArm.srv
  GetActualPose.srv
  RotateTool.srv
  SetSuction.srv
  StopArm.srv
)
generate_messages()

catkin_package(CATKIN_DEPENDS message_runtime rospy)

catkin_install_python(PROGRAMS scripts/controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
