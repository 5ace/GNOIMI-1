project (NovuGNO-IMI)
cmake_minimum_required(VERSION 3.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -m64")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++11 -m64")

set(CMAKE_MODULE_PATH ${NovuGNO-IMI_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
message(info: ${CMAKE_MODULE_PATH})
find_package(LAPACK)
find_package(OpenBLAS REQUIRED)
include_directories( ${OpenBLAS_INCLUDE_DIR} )
link_libraries( ${OpenBLAS_LIB})


#SET(YAEL_DIR /home/arbabenko/libs/yael_v438)
SET(YAEL_DIR $ENV{YAEL_DIR})
SET(YAEL_LIB $ENV{YAEL_DIR}/yael)

add_definitions(-DFINTEGER=long)

#########################################################################################
#INCLUDE_DIRECTORIES(${BLAS_DIR})
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})

ADD_EXECUTABLE (learnGNOIMI learn_GNOIMI.cpp)
TARGET_LINK_LIBRARIES (learnGNOIMI yael ${LAPACK_LIBRARIES})

#########################################################################################

ADD_EXECUTABLE (obtain_assignment obtain_assignment.cpp)
TARGET_LINK_LIBRARIES (obtain_assignment yael ${LAPACK_LIBRARIES})
