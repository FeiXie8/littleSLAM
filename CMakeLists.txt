cmake_minimum_required(VERSION 2.4.6)
project(usesophus)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
#find_package(G2O REQUIRED)

add_definitions(${PCL_DEFINITIONS})

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )


include_directories(${EIGEN3_INCLUDE_DIR} 
                    ${Sophus_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Ceres_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
                    ./include
                    )

                    sadfasdfasdf