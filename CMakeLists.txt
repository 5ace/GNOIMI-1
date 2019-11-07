project (GNO-IMI)
cmake_minimum_required(VERSION 3.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x -m64 -lpthread -lopenblas")

set(CMAKE_MODULE_PATH ${NovuGNO-IMI_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

SET(YAEL_DIR /home/zhangcunyi/code/yael_v438)
SET(YAEL_LIB /home/zhangcunyi/code/yael_v438/yael)

#add LAPACK
find_package(LAPACK)
link_libraries( ${LAPACK_LIBRARIES} )  

#add OpenBLAS
SET(BLAS_DIR /usr/include/openblas)
SET(BLAS_LIB /usr/lib64)
include_directories( ${BLAS_DIR} )
link_directories( ${BLAS_LIB})

#add MKL
#find_package(MKL REQUIRED)
#include_directories( ${MKL_INCLUDE_DIR})
#link_libraries( ${MKL_LIBRARIES})

#add yael
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
link_libraries( yael )
add_definitions(-DFINTEGER=long)


include_directories( ${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRCS src/*.cpp)
MESSAGE( STATUS "this var key = ${SRCS}.")

ADD_EXECUTABLE (learnGNOIMI ${SRCS} bin/learn_GNOIMI.cpp)
ADD_EXECUTABLE (obtain_assignment ${SRCS} bin/obtain_assignment.cpp)
ADD_EXECUTABLE (generate_initialization_data ${SRCS} bin/generate_initialization_data.cpp)
