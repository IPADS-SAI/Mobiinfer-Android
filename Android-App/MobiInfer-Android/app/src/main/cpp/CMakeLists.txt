cmake_minimum_required(VERSION 3.22.1)

project("mobiinfer" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    elseif(ANDROID_ABI STREQUAL "x86")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(WARNING "Unsupported ABI: ${ANDROID_ABI}, defaulting to generic CPU")
        set(GGML_SYSTEM_ARCH "")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    endif()
endif()

# The inference engine lives at the top of the enclosing repository (MobiInfer-Android),
# six levels above this file. No git submodule or symlink is involved.
# Allow override via -DLLAMA_SRC=... if you keep the engine elsewhere.
if(NOT DEFINED LLAMA_SRC)
    set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
endif()
get_filename_component(LLAMA_SRC ${LLAMA_SRC} ABSOLUTE)
if(NOT EXISTS ${LLAMA_SRC}/CMakeLists.txt)
    message(FATAL_ERROR
        "Inference engine (llama.cpp-omni) not found at ${LLAMA_SRC}. "
        "Build from within the MobiInfer-Android repository, or pass -DLLAMA_SRC=<path>.")
endif()
message(STATUS "Using llama.cpp-omni source at: ${LLAMA_SRC}")

add_subdirectory(${LLAMA_SRC} ${CMAKE_BINARY_DIR}/build-llama)

# Reuse the exact CLI inference core for MiniCPM 4.7 speculative decoding.
# Compile its queue/context library only; no HTTP server or web UI is built.
set(SERVER_CORE ${LLAMA_SRC}/tools/server)
add_library(minicpm47_core STATIC
        ${SERVER_CORE}/server-chat.cpp
        ${SERVER_CORE}/server-task.cpp
        ${SERVER_CORE}/server-queue.cpp
        ${SERVER_CORE}/server-common.cpp
        ${SERVER_CORE}/server-context.cpp)
set_target_properties(minicpm47_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(minicpm47_core PUBLIC ${SERVER_CORE}
        PRIVATE ${LLAMA_SRC} ${LLAMA_SRC}/vendor ${LLAMA_SRC}/common
        ${LLAMA_SRC}/tools/omni ${LLAMA_SRC}/tools/omni/voxcpm2)
target_link_libraries(minicpm47_core PRIVATE llama-common omni mtmd voxcpm2_runtime)

if(NOT DEFINED LLAMA_INSTALL_VERSION)
    set(LLAMA_INSTALL_VERSION "0.0.0")
endif()

add_library(${CMAKE_PROJECT_NAME} SHARED
        llama_jni.cpp
        minicpm47_runtime.cpp
        omni_jni.cpp)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${LLAMA_SRC}/tools/mtmd
        ${LLAMA_SRC}/tools/omni
        ${LLAMA_SRC}/tools/omni/voxcpm2
        # llama-common/peg-parser pulls in <nlohmann/json_fwd.hpp> via chat.h.
        # The vendored nlohmann headers live under llama.cpp-omni/vendor.
        ${LLAMA_SRC}/vendor)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        minicpm47_core
        llama-common
        mtmd
        voxcpm2_runtime
        android
        log)
