cmake_minimum_required(VERSION 2.4.6)
project(usesophus)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_definitions("-DUSE_EIGEN")

find_package(OpenCV 4.5 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
find_package(G2O REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(GFlags REQUIRED)
find_package(CSparse REQUIRED)

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
                    ${GLOG_INCLUDE_DIRS}
                    ${GTEST_INCLUDE_DIRS}
                    ${GFLAGS_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR}
                    ./include
                    )

set(THIRD_PARTY_LIBS ${OpenCV_LIBS}
                     ${Sophus_LIBRARIES}
                     ${Pangolin_LIBRARIES} GL GLU GLEW glut
                     g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
                     ${GTEST_BOTH_LIBRARIES}
                     ${GLOG_LIBRARIES}
                     ${GFLAGS_LIBRARIES}
                     pthread
                     ${CSPARSE_LIBRARY}
                     )

enable_testing()

add_subdirectory(src)
#add_subdirectory(test)
