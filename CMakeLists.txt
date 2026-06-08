cmake_minimum_required(VERSION 3.16)
project(gaussian_blur)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)

FetchContent_Declare(
    OpenCLHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG        v2024.05.08
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(OpenCLHeaders)

set(OPENCL_ICD_LOADER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    OpenCLICDLoader
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG        v2024.05.08
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(OpenCLICDLoader)

set(OPENCL_CLHPP_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPENCL_CLHPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    OpenCLHeadersCpp
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-CLHPP.git
    GIT_TAG        v2024.05.08
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(OpenCLHeadersCpp)

if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/installed/x64-windows")
endif()
find_package(OpenCV REQUIRED)

add_executable(gaussian_blur main.cpp)
target_link_libraries(gaussian_blur OpenCL::OpenCL OpenCL::HeadersCpp ${OpenCV_LIBS})
target_include_directories(gaussian_blur PRIVATE ${OpenCV_INCLUDE_DIRS})

add_custom_command(TARGET gaussian_blur POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:gaussian_blur>
        $<TARGET_FILE_DIR:gaussian_blur>
    COMMAND_EXPAND_LISTS
)
