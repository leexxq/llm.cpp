cmake_minimum_required(VERSION 3.14)
project(llm.cpp)

# 如果用户没有指定编译类型，则默认设置为 Debug
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()




set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Eigen3 install

set(Eigen3_DIR ./thirdparty/eigen3/share/eigen3/cmake/)
find_package(Eigen3 3...5 REQUIRED NO_MODULE)

add_executable(eigen3exp tests/eigen3_example.cpp)
target_link_libraries(eigen3exp Eigen3::Eigen)

#---------------

# CLI11 install
include(FetchContent)
FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)

FetchContent_MakeAvailable(cli11_proj)


add_executable(cli11exp tests/cli11_example.cpp)

target_link_libraries(cli11exp CLI11::CLI11)

#---------------


#install fmt
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)



#---------------

#build libgpt2cpp

aux_source_directory(gpt2/ GPT2_SOURCES)

add_library(gpt2cpp SHARED ${GPT2_SOURCES} )


target_link_libraries(gpt2cpp PUBLIC Eigen3::Eigen PUBLIC fmt::fmt)

target_compile_definitions(gpt2cpp PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(gpt2cpp PUBLIC 
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-Wno-nan-infinity-disabled>
    )

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	target_compile_options(gpt2cpp PUBLIC
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:Release>:-Wno-nan-infinity-disabled>
    )
endif()




#-----------------


add_executable(traingpt2 train.cpp)

target_link_libraries(traingpt2 PRIVATE CLI11::CLI11 PRIVATE gpt2cpp )

add_custom_command(TARGET traingpt2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:traingpt2>/data"
)

add_subdirectory(tests)
