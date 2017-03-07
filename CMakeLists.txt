project (NovuGNO-IMI)
cmake_minimum_required(VERSION 3.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -m64")

set(CMAKE_MODULE_PATH ${NovuGNO-IMI_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

#add LAPACK
find_package(LAPACK)
link_libraries( ${LAPACK_LIBRARIES} )  

#add OpenBLAS
find_package(OpenBLAS REQUIRED)
include_directories( ${OpenBLAS_INCLUDE_DIR} )
link_libraries( ${OpenBLAS_LIB})

#add MKL
find_package(MKL REQUIRED)
include_directories( ${MKL_INCLUDE_DIR})
link_libraries( ${MKL_LIBRARIES})

#add yael
SET(YAEL_DIR $ENV{YAEL_DIR})
SET(YAEL_LIB $ENV{YAEL_DIR}/yael)
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
link_libraries( yael )
add_definitions(-DFINTEGER=long)


include_directories( ${NovuGNO-IMI_SOURCE_DIR}/include)

file(GLOB_RECURSE SRCS src/*.cpp)

ADD_EXECUTABLE (learnGNOIMI ${SRCS} bin/learn_GNOIMI.cpp)
ADD_EXECUTABLE (obtain_assignment ${SRCS} bin/obtain_assignment.cpp)
ADD_EXECUTABLE (generate_initialization_data ${SRCS} bin/generate_initialization_data.cpp)
