cmake_minimum_required(VERSION 3.14)

option(ENABLE_CUDA "Enable CUTLASS" OFF)
option(ENABLE_NVCC_TIME "Enable nvcc --time profiling" OFF)
if(ENABLE_CUDA)
	project(llm.cpp LANGUAGES CXX CUDA)

	find_package(CUDAToolkit REQUIRED)

	set(CMAKE_CUDA_ARCHITECTURES "86")
	set(CUDA_ARCHITECTURES "86")

else()
	project(llm.cpp)
endif()

if(NOT ENABLE_CUDA)
    set(
        ENABLE_NVCC_TIME
        OFF
        CACHE BOOL
        "Enable nvcc --time"
        FORCE
    )
endif()




# Default build type is release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(ENABLE_CUDA)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

endif()


# Eigen3 install

set(Eigen3_DIR ./thirdparty/eigen3/share/eigen3/cmake/)
find_package(Eigen3 3...5 REQUIRED NO_MODULE)

# add_executable(eigen3exp tests/eigen3_example.cpp)
# target_link_libraries(eigen3exp Eigen3::Eigen)

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


# add_executable(cli11exp tests/cli11_example.cpp)

# target_link_libraries(cli11exp CLI11::CLI11)

#---------------


#install fmt
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  ) # 10.2.1
FetchContent_MakeAvailable(fmt)



#---------------

#build libgpt2cpp

aux_source_directory(gpt2/ GPT2_SOURCES)



add_library(gpt2cpp STATIC ${GPT2_SOURCES} )


if(ENABLE_CUDA)
	aux_source_directory(gpt2/cuda GPT2_CUDA_SOURCES)
	add_library(gpt2cuda STATIC ${GPT2_CUDA_SOURCES} )

	target_include_directories(gpt2cuda PRIVATE 
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cutlass/include
		${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cutlass/tools/util/include
		${CMAKE_CURRENT_SOURCE_DIR}/gpt2
	)
    if(ENABLE_NVCC_TIME)
        target_compile_options(
			gpt2cuda	
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--time=nvcc_time.log>
        )
    endif()

	target_link_libraries(gpt2cuda PRIVATE CUDA::cudart PUBLIC fmt::fmt)

	target_compile_options(gpt2cuda PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
	)
endif()

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
	if(ENABLE_CUDA)
		target_compile_options(gpt2cuda PUBLIC 
			$<$<CONFIG:Release>:-O3>
			$<$<CONFIG:Release>:-march=native>
			$<$<CONFIG:Release>:-ffast-math>
		)
	endif()


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


add_executable(traingpt2 train.cpp  utils/dataloader.cpp utils/tokenizer.cpp)

target_link_libraries(traingpt2 PRIVATE CLI11::CLI11 PRIVATE gpt2cpp)

if(ENABLE_CUDA)
	add_executable(traingpt2cuda traincuda.cpp utils/dataloader.cpp utils/tokenizer.cpp gpt2/adamw.cpp)
	target_link_libraries(traingpt2cuda PRIVATE CLI11::CLI11 PRIVATE gpt2cuda PRIVATE CUDA::cudart )
	if(OpenMP_CXX_FOUND)
		target_link_libraries(traingpt2cuda PUBLIC $<$<CONFIG:Release>:OpenMP::OpenMP_CXX>)
	endif()
    if(ENABLE_NVCC_TIME)
        target_compile_options(
			traingpt2cuda
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--time>
        )
    endif()
endif()


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

