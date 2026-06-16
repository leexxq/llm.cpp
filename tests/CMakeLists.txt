
include(FetchContent)
FetchContent_Declare(
        googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG v1.17.0
		PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
        # DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # DOWNLOAD_NO_PROGRESS TRUE
      )
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# include(FetchContent)
# FetchContent_Declare(
#   	googletest
#   # URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# 	URL https://github.com
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

enable_testing()

aux_source_directory(gpt2/ TEST_GPT2_SRC)
if(ENABLE_CUDA)
aux_source_directory(gpt2/cuda TEST_GPT2_SRC)
endif()

message(STATUS "========== TEST_GPT2_SRC ==========")
foreach(file ${TEST_GPT2_SRC})
  message(STATUS "  ${file}")
endforeach()
message(STATUS "==================================")


add_executable(gpt2test EXCLUDE_FROM_ALL ${TEST_GPT2_SRC})

include_directories(../gpt2/)

target_link_libraries(
	gpt2test
	PRIVATE gpt2cpp
	PRIVATE GTest::gtest_main
	PRIVATE Eigen3::Eigen
	PRIVATE CLI11::CLI11
)

include(GoogleTest)

gtest_discover_tests(gpt2test)

#---------------

