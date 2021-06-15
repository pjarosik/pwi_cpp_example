cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cpp_example LANGUAGES CXX CUDA)

# Conan generates FindXYZ.cmake files of dependencies into build directory --
# -- add it to CMAKE_MODULE_PATH, so we can use e.g. OpenCV package.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_BINARY_DIR}")
find_package(OpenCV REQUIRED)

# Imaging pipeline source files and target.
# Imaging pipeline is compiled due to the lack of support of C++17 standard
# for nvcc ver. 10.0.
# C++17 standard is required by arrus package.
set(ARRUS_IMAGING_PIPELINE_SRC
    # Imaging
    imaging/kernels/Decimation.cuh
    imaging/kernels/EnvelopeDetection.cuh
    imaging/kernels/FirFilterSingleton.cuh
    imaging/kernels/Kernel.cuh
    imaging/kernels/KernelInitContext.h
    imaging/kernels/KernelInitResult.h
    imaging/kernels/LpFilterSingleton.cuh
    imaging/kernels/QuadratureDemodulation.cuh
    imaging/kernels/ReconstructLriPwi.cuh
    imaging/kernels/RemapToLogicalOrder.cuh
    imaging/kernels/Sum.h
    imaging/kernels/ToBmode.cuh
    imaging/kernels/Transpose.cuh
    imaging/CudaUtils.cuh
    imaging/NdArray.h
    imaging/Pipeline.cuh
    imaging/Pipeline.cu
    imaging/DataType.h)

add_library(arrus-imaging SHARED ${ARRUS_IMAGING_PIPELINE_SRC})
set_target_properties(arrus-imaging
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)
target_compile_definitions(arrus-imaging PRIVATE IMAGING_BUILD_STAGE)

# Main executable file.
# ARRUS package requires a compiler with C++17 support at least.
# Now, all the components should be C++17 at least.
set(CMAKE_CXX_STANDARD 17)
set(MAIN_SRC
    main.cpp
    common.h
    logging/MyCustomLoggerFactory.h logging/MyCustomLogger.h cfg.h Display2D.h)

# TODO change the below path appropriately.
set(ARRUS_INCLUDE_PATH "C:/Users/pjarosik/arrus/include")
# TODO change the below path appropriately.
set(ARRUS_LIB_PATH "C:/Users/pjarosik/arrus/lib64")

add_executable(pwi_example ${MAIN_SRC})
target_link_directories(pwi_example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(pwi_example PRIVATE arrus-core arrus-imaging opencv::opencv)
target_include_directories(pwi_example PRIVATE ${ARRUS_INCLUDE_PATH})

add_executable(async_example async.cpp)
target_link_directories(async_example PRIVATE ${ARRUS_LIB_PATH})
target_link_libraries(async_example PRIVATE arrus-core)
target_include_directories(async_example PRIVATE ${ARRUS_INCLUDE_PATH})


