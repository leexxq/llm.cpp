#pragma once
#ifdef DEBUG
#define DEBUG_PRINTLN(...) \
	do { \
		fmt::print("[DEBUG]\t"); \
		fmt::println(__VA_ARGS__); \
	} while (false)
#endif
#ifndef DEBUG
#define DEBUG_PRINTLN(...)
#endif

#define INFO_PRINTLN(...) fmt::println(__VA_ARGS__)
#define INFO_PRINT(...) fmt::print(__VA_ARGS__)

#define ERROR_PRINTLN(...) \
	do { \
		fmt::print(stderr, "[ERROR]\t"); \
		fmt::println(stderr, __VA_ARGS__); \
	} while (false)


#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

