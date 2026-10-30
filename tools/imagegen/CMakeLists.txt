# imagegen — thin text-to-image wrapper over the sdcpp-hexagon fork of
# stable-diffusion.cpp (MOSS-Image on Qualcomm Hexagon / HTP).
#
# This is a STANDALONE CMake project, deliberately separate from the top-level
# llama.cpp-omni build tree. sdcpp-hexagon vendors its own (patched) ggml with
# the ggml-hexagon backend, whose CMake targets (ggml, ggml-base, ggml-cpu,
# ggml-hexagon, ...) collide with llama.cpp-omni's. Building here with its own
# source/binary dir keeps the two ggml copies isolated; the resulting
# libimagegen.so is loaded next to the llama one at runtime.
#
# Typical invocation (driven from Android gradle, see build.gradle.kts):
#   cmake -S tools/imagegen -B <build> \
#         -DCMAKE_TOOLCHAIN_FILE=<ndk>/android.toolchain.cmake \
#         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
#         -DSDCPP_SOURCE_DIR=<repo>/vendor/sdcpp-hexagon \
#         -DHEXAGON_SDK_ROOT=... -DHEXAGON_TOOLS_ROOT=...
#
cmake_minimum_required(VERSION 3.16)
project(imagegen CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT SDCPP_SOURCE_DIR)
    get_filename_component(SDCPP_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/sdcpp-hexagon" ABSOLUTE)
endif()
if(NOT EXISTS "${SDCPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "sdcpp-hexagon not found at ${SDCPP_SOURCE_DIR}. "
        "Vendor it under vendor/sdcpp-hexagon (git submodule), "
        "or pass -DSDCPP_SOURCE_DIR=<path>.")
endif()
message(STATUS "imagegen: using sdcpp-hexagon at ${SDCPP_SOURCE_DIR}")

# Build stable-diffusion.cpp (and its own ggml + ggml-hexagon) as a shared
# library so the wrapper can link against it while both ggml copies stay in
# separate .so files.
set(SD_BUILD_SHARED_LIBS      ON  CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB  OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES         OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SERVER           OFF CACHE BOOL "" FORCE)
set(SD_WEBP                   OFF CACHE BOOL "" FORCE)
set(SD_WEBM                   OFF CACHE BOOL "" FORCE)
set(SD_GGML_SOURCE_DIR "${SDCPP_SOURCE_DIR}/ggml" CACHE PATH "" FORCE)

# Hexagon backend (HTP/NPU). Requires HEXAGON_SDK_ROOT/HEXAGON_TOOLS_ROOT in
# the environment; the skels (v73/v75/v79/v81) are built by ExternalProject.
if(ANDROID)
    set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
    set(GGML_HEXAGON_HTP_BUILD_TYPE Release CACHE STRING "" FORCE)
    set(GGML_HEXAGON_HTP_DEBUG OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${SDCPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sdcpp EXCLUDE_FROM_ALL)

set(IMAGEGEN_SOURCES imagegen.cpp imagegen.h)

# When building for the Android app, also compile the JNI bridge into the same
# .so so Kotlin only has to load "imagegen".
set(IMAGEGEN_JNI_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../Android-App/MobiInfer-Android/app/src/main/cpp")
if(ANDROID AND EXISTS "${IMAGEGEN_JNI_DIR}/image_jni.cpp")
    list(APPEND IMAGEGEN_SOURCES "${IMAGEGEN_JNI_DIR}/image_jni.cpp")
    message(STATUS "imagegen: including JNI bridge from ${IMAGEGEN_JNI_DIR}")
endif()

add_library(imagegen SHARED ${IMAGEGEN_SOURCES})
target_include_directories(imagegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SDCPP_SOURCE_DIR}/include
    # src/ holds the internal headers (core/util.h): the progress callback is
    # shared by weight loading and denoising, and only the internal flag can
    # tell them apart.
    ${SDCPP_SOURCE_DIR}/src
    ${SDCPP_SOURCE_DIR}/thirdparty)
target_link_libraries(imagegen PRIVATE stable-diffusion)

if(ANDROID)
    target_link_libraries(imagegen PRIVATE log)
endif()

set_target_properties(imagegen PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
