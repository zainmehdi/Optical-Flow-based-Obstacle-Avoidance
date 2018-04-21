cmake_minimum_required(VERSION 2.8.3)
project(optical_flow_obstacle_avoidance)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS

        image_transport
        cv_bridge
        roscpp
        rospy
        std_msgs
       
)
find_package(OpenCV REQUIRED)

message(STATUS" Opencv dir:" ${OpenCV_INCLUDE_DIRS})
#message(STATUS" Opencv lib:"${OpenCV_LIBRARIES})
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})


catkin_package(
        LIBRARIES ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} 
        CATKIN_DEPENDS cv_bridge roscpp rospy std_msgs
       # DEPENDS system_lib
)




add_executable(Obstacle_Avoidance_optical src/Obstacle_avoidance_node.cpp)
target_link_libraries(Obstacle_Avoidance_optical ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
