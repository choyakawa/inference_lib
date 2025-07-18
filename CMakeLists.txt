cmake_minimum_required(VERSION 3.22)
project(HogwildCache CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")

find_package(CUDAToolkit REQUIRED)

option(HOGDILD_TORCH_BINDINGS "Build pytorch bindings" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_subdirectory(src/tpl)

option(HOGDILD_ENABLE_FLOAT "build with float32 kernel support" ON)
option(HOGDILD_ENABLE_HALF "build with float16 kernel support" ON)
option(HOGDILD_ENABLE_BFLOAT "build with bfloat16 kernel support" ON)

add_library(hogwild-kernels STATIC ${HOGWILD_KERNEL_INSTANTIATIONS})
target_include_directories(hogwild-kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(hogwild-kernels PUBLIC -expt-relaxed-constexpr -lineinfo)

if(HOGDILD_ENABLE_FLOAT)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_FLOAT=1)
endif ()

if(HOGDILD_ENABLE_HALF)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_HALF=1)
endif ()

if(HOGDILD_ENABLE_BFLOAT)
    target_compile_definitions(hogwild-kernels PUBLIC HOGDILD_ENABLE_BFLOAT=1)
endif ()


if(HOGDILD_TORCH_BINDINGS)
    include(cmake/pytorch.cmake)
    find_package(Python3 COMPONENTS Development.SABIModule REQUIRED)
    Python3_add_library(hogatt MODULE src/binding.cu WITH_SOABI USE_SABI 3.10)
    if(SKBUILD)
        install(TARGETS hogatt DESTINATION ${SKBUILD_PLATLIB_DIR}/hogwild)
    endif ()
    target_link_libraries(hogatt PRIVATE torch hogwild-kernels)
    target_compile_options(hogatt PUBLIC -expt-relaxed-constexpr -lineinfo)
    target_include_directories(hogatt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif ()
