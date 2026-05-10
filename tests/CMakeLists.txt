
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
		PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        DOWNLOAD_NO_PROGRESS TRUE
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

file(GLOB_RECURSE TEST_GPT2_SRC gpt2/*.cpp)

add_executable(
	gtestexp
	gtest_example.cpp
)
target_link_libraries(
	gtestexp
	PRIVATE GTest::gtest_main
)

add_executable(gpt2test ${TEST_GPT2_SRC})
target_link_libraries(
	gpt2test
	PRIVATE libgpt2cpp
	PRIVATE GTest::gtest_main
	PRIVATE Eigen3::Eigen
	PRIVATE CLI11::CLI11
)

include(GoogleTest)

gtest_discover_tests(gtestexp)
gtest_discover_tests(gpt2test)

#---------------

