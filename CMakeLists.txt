cmake_minimum_required(VERSION 3.14)
project(llm.cpp)

# Default build type is release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
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

add_library(gpt2cpp STATIC ${GPT2_SOURCES} )

target_link_libraries(gpt2cpp PUBLIC Eigen3::Eigen PUBLIC fmt::fmt)


# define Debug macro for debug
target_compile_definitions(gpt2cpp PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)


include(CheckCXXCompilerFlag)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	# --- using follow command install openmp ---
	# --- brew install libomp---

	# ---solove macOS CMake cannot find OpenMP (When using "brew install libomp") ---
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		execute_process(COMMAND brew --prefix libomp
						OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
						OUTPUT_STRIP_TRAILING_WHITESPACE)
		
		set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
		set(OpenMP_CXX_LIB_NAMES "omp")
		set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
	endif()
	# ---------------------------------------------

	target_compile_options(gpt2cpp PUBLIC 
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
	)

	check_cxx_compiler_flag("-Wnan-infinity-disabled" COMPILER_SUPPORTS_NO_NAN_INF)
	if(COMPILER_SUPPORTS_NO_NAN_INF)
		target_compile_options(gpt2cpp PUBLIC $<$<CONFIG:Release>:-Wnan-infinity-disabled>)
	endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	target_compile_options(gpt2cpp PUBLIC
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/fp:fast>
    )
endif()


# find_package(OpenMP REQUIRED CXX)
find_package(OpenMP REQUIRED CXX)
if(OpenMP_CXX_FOUND)
	target_link_libraries(gpt2cpp PUBLIC $<$<CONFIG:Release>:OpenMP::OpenMP_CXX>)
endif()



#-----------------


add_executable(traingpt2 train.cpp)

target_link_libraries(traingpt2 PRIVATE CLI11::CLI11 PRIVATE gpt2cpp)

option(ENABLE_PROFILER "For User who need profiler" OFF)

if(ENABLE_PROFILER)
	if(APPLE)
		find_library(PROFILER_LIBRARY 
			NAMES profiler 
			PATHS /opt/homebrew/lib /usr/local/lib
		)
	if(NOT PROFILER_LIBRARY)
		message(FATAL_ERROR "Connot find profiler! Ensure executed 'brew install gperftools'")
	endif()
	target_link_libraries(traingpt2 PRIVATE ${PROFILER_LIBRARY})
	endif()
endif()


add_custom_command(TARGET traingpt2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:traingpt2>/data"
)

option(ENABLE_TESTS "Build the tests (require GoogleTest)" OFF)

if(ENABLE_TESTS)
	message(STATUS "Testing is enabled. Adding tests")
	add_subdirectory(tests)
else()
	message(STATUS "Testing is disabled. Skipping tests...")
endif()

