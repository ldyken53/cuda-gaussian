cmake_minimum_required(VERSION 3.20)

option(BUILD_GAUSSIAN_TESTS "Build test executable for Gaussian model" OFF)

project(GaussianModelTest LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set CUDA architectures (adjust for your target GPU)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

# Find required packages
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuBQL
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# CudaRasterizer library (from your original project)
add_library(CudaRasterizer
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

target_compile_options(CudaRasterizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(CudaRasterizer PROPERTIES 
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(CudaRasterizer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
)

target_include_directories(CudaRasterizer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuBQL 
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link CUDA libraries to CudaRasterizer
target_link_libraries(CudaRasterizer 
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

if(BUILD_GAUSSIAN_TESTS)
  # Test executable
  add_executable(test_gaussian_model
      test.cu
  )

  target_compile_options(test_gaussian_model PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  set_target_properties(test_gaussian_model PROPERTIES 
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Link libraries to the test executable
  target_link_libraries(test_gaussian_model
      CudaRasterizer
      ${CUDA_LIBRARIES}
      ${CUDA_CUDART_LIBRARY}
  )

  # Platform-specific settings
  if(UNIX)
      target_link_libraries(test_gaussian_model pthread)
  endif()

  # Custom targets for running tests
  add_custom_target(run_test
      COMMAND test_gaussian_model point_cloud.ply 100
      DEPENDS test_gaussian_model
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Running GaussianModel test with 100 points"
  )
endif()

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")