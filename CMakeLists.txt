cmake_minimum_required(VERSION 3.22)
project(ParAMD VERSION 1.0 LANGUAGES C CXX)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the installation path SuiteSparse is installed
set(SUITESPARSE_ROOT "" CACHE PATH "Path to the SuiteSparse installation")

# Add preprocessor macro if SuiteSparse is installed
if (SUITESPARSE_ROOT)
  message(STATUS "SuiteSparse root directory: ${SUITESPARSE_ROOT}")

  # Define a preprocessor macro USE_SUITESPARSE to indicate that SuiteSparse is being used
  add_definitions(-DUSE_SUITESPARSE)

  # Set SuiteSparse include and library paths based on SUITESPARSE_ROOT
  set(SUITESPARSE_INCLUDE_DIR "${SUITESPARSE_ROOT}/include")
  set(SUITESPARSE_LIB_DIR_64 "${SUITESPARSE_ROOT}/lib64")
  set(SUITESPARSE_LIB_DIR "${SUITESPARSE_ROOT}/lib")


  # Check that the necessary directories exist
  if (NOT EXISTS "${SUITESPARSE_INCLUDE_DIR}/suitesparse/amd.h") 
      message(FATAL_ERROR "SuiteSparse AMD (suitesparse/amd.h) not found!")
  endif()

  if (NOT EXISTS "${SUITESPARSE_LIB_DIR_64}/libamd.so" AND NOT EXISTS "${SUITESPARSE_LIB_DIR_64}/libamd.a")
    if (NOT EXISTS "${SUITESPARSE_LIB_DIR}/libamd.so" AND NOT EXISTS "${SUITESPARSE_LIB_DIR}/libamd.a")
      message(FATAL_ERROR "SuiteSparse library (libamd) not found!")
    endif()
  endif()

  # Find the SuiteSparse AMD library
  find_library(SUITESPARSE_AMD_LIB NAMES amd
    PATHS ${SUITESPARSE_LIB_DIR_64} ${SUITESPARSE_LIB_DIR}
    NO_DEFAULT_PATH
  )

  if (NOT SUITESPARSE_AMD_LIB)
    message(FATAL_ERROR "Could not find SuiteSparse AMD library!")
  endif()

  # Include the SuiteSparse headers
  include_directories(${SUITESPARSE_INCLUDE_DIR})

else()
  message(STATUS "SUITESPARSE_ROOT not specified, SuiteSparse will not be used.")
endif()

include_directories(include)

set(SOURCES
  src/config.cpp
  src/matrix.cpp
  src/paramd.cpp
  src/utils.cpp
)

add_library(paramd_lib SHARED ${SOURCES})

set_target_properties(paramd_lib PROPERTIES
  OUTPUT_NAME paramd
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER include/paramd/paramd.h
)

add_executable(paramd bin/main.cpp)

target_link_libraries(paramd PRIVATE paramd_lib)

if (SUITESPARSE_AMD_LIB)
  target_link_libraries(paramd PRIVATE ${SUITESPARSE_AMD_LIB})
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(paramd_lib PRIVATE -lmetis OpenMP::OpenMP_CXX)
  target_link_libraries(paramd PRIVATE paramd_lib -lmetis OpenMP::OpenMP_CXX)
endif()

# Add to CMakeLists.txt
find_package(METIS QUIET)

if(METIS_FOUND)
    include_directories(${METIS_INCLUDE_DIRS})
    add_definitions(-DUSE_METIS)
    set(METIS_LIBRARIES metis)
    message(STATUS "METIS found: ${METIS_INCLUDE_DIRS}")
else()
    set(METIS_LIBRARIES "")
    message(STATUS "METIS not found. Building without METIS support.")
endif()

# Update target_link_libraries
target_link_libraries(paramd_lib PRIVATE ${METIS_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(paramd PRIVATE paramd_lib ${METIS_LIBRARIES} OpenMP::OpenMP_CXX)

install(TARGETS paramd_lib
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/paramd
)

install(TARGETS paramd
  RUNTIME DESTINATION bin
)