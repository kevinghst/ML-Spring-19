cmake_minimum_required(VERSION 3.14)
project(ML_Spring19_Project)
#set(CMAKE_INSTALL_PREFIX C:/Anaconda/pkgs/)

set(CMAKE_CXX_STANDARD 14)

set(xtl_DIR C:/Anaconda/pkgs/xtl-0.6.4-h1ad3211_0/Library/lib/cmake/xtl)
set(xtensor_DIR C:/Anaconda/pkgs/xtensor-0.20.5-h1ad3211_0/Library/lib/cmake/xtensor)
set(xtensor-blas_DIR C:/Anaconda/pkgs/xtensor-blas-0.16.1-hd41736c_0/Library/lib/cmake/xtensor-blas)
#set(OpenBLAS_DIR C:/Anaconda/pkgs/openblas-0.3.6-h828a276_2/Library/share/cmake/OpenBLAS)

find_package(xtl REQUIRED CONFIG)
find_package(xtensor REQUIRED CONFIG)
find_package(xtensor-blas REQUIRED CONFIG)

set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
set(XBLAS_INCLUDE_DIR ${xtensor_blas_INCLUDE_DIRS})
#message(STATUS "xtensor : " ${XTENSOR_INCLUDE_DIR})
#message(STATUS "xblas   : " ${XBLAS_INCLUDE_DIR})


message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(set_compiler_flag.cmake)

if(CPP17)
    # User requested C++17, but compiler might not oblige.
    set_compiler_flag(
            _cxx_std_flag CXX
            "-std=c++17"  # this should work with GNU, Intel, PGI
            "/std:c++17"  # this should work with MSVC
    )
    if(_cxx_std_flag)
        message(STATUS "Building with C++17")
    endif()
else()
    set_compiler_flag(
            _cxx_std_flag CXX REQUIRED
            "-std=c++14"  # this should work with GNU, Intel, PGI
            "/std:c++14"  # this should work with MSVC
    )
    message(STATUS "Building with C++14")
endif()

if(NOT _cxx_std_flag)
    message(FATAL_ERROR "xtensor-blas needs a C++14-compliant compiler.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=native -Wextra -Wreorder")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /EHsc /MP /bigobj /wd4800")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    else() # We are using clang-cl
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /EHsc /MP /bigobj")
        set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    endif()
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()


add_definitions(-DHAVE_CBLAS=1)
OPTION(USE_OPENBLAS "use OpenBLAS (requires suitable OpenBLASConfig.cmake)" OFF)
if(OpenBLAS_DIR)
    set(USE_OPENBLAS ON)
endif()
if(USE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${OpenBLAS_LIBRARY})
else()
#    set(BLAS_DIR C:/Anaconda/pkgs/openblas-0.3.6-h828a276_2/Library/share/cmake/OpenBLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLAS_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
#message(STATUS "LAPACK LIBRARIES: " ${LAPACK_LIBRARIES})

#set(BLAS_INCLUDE_DIR ${OpenBLAS_INCLUDE_DIRS})

include_directories(.)
include_directories(${xtl_INCLUDE_DIRS})
include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${XBLAS_INCLUDE_DIR})

set(SOURCE_FILES
        optimizer.cpp
        loss_function.cpp
        activation_function.cpp
        layer.cpp
        neural_network.cpp
        autoencoder.cpp
        )
set(HEADER_FILES
        optimizer.h
        loss_function.h
        activation_function.h
        layer.h
        neural_network.h
        )

add_library(NeuralNetwork SHARED
        ${HEADER_FILES}
        ${SOURCE_FILES}
        )

#add_executable(autoencoder
#        autoencoder.cpp)
add_executable(autoencoder
        autoencoder.cpp
        ${HEADER_FILES}
        ${SOURCE_FILES}
        ${XTENSOR_BLAS_HEADERS}
        ${XTENSOR_HEADERS}
        )

#target_include_directories(autoencoder PUBLIC
#        ${XTENSOR_INCLUDE_DIR}
#        ${XBLAS_INCLUDE_DIR}
#        )
link_libraries(xtensor-blas
        xtensor
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        )
target_link_libraries(autoencoder
        NeuralNetwork
#        xtensor
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
#        xtensor-blas
        )


