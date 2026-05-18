#pragma once

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <tuple>
#include <vector>

#define FORWARD_NO_DISCARD [[nodiscard("result may be used in forward and backward later")]]
#define BACKWARD_NO_DISCARD [[nodiscard("result may be used in grad update and backward later")]]

#ifdef DEBUG
#define DEBUG_PRINTLN(...) fmt::println(__VA_ARGS__)
#endif
#ifndef DEBUG
#define DEBUG_PRINTLN(...)
#endif

#define INFO_PRINTLN(...) fmt::println(__VA_ARGS__)

#define ERROR_PRINTLN(...) fmt::println(stderr, __VA_ARGS__)

/**
 * BHTC is (batch, head,seq_len,channels) , (seq_len,channels) is matrix
 * Hc is head_channels that's attention head's q dim;
 * contains some
 * **/
using Mati = Eigen::MatrixXi;

using Matf = Eigen::MatrixXf;

using Vecf = Eigen::VectorXf;

using Veci = Eigen::VectorXi;

using VecBTC = std::vector<Matf>;
inline VecBTC makeVecBTC(size_t B, size_t T, size_t C) {
	return VecBTC(B, Matf(T, C));
}

void operator+=(VecBTC &x, const VecBTC &y);

// L is layers
using VecLBTC = std::vector<VecBTC>;

inline VecLBTC makeVecLBTC(size_t L, size_t B, size_t T, size_t C) {
	return VecLBTC(L, makeVecBTC(B, T, C));
}

using VeciBTC = std::vector<Mati>;

using VecBT = std::vector<Vecf>;
inline VecBT makeVecBT(size_t B, size_t T) {
	return VecBT(B, Vecf(T));
}

using VecHTC = VecBTC;

// BHTC is batch , (head , seq_len,channels)
using VecBHTC = std::vector<VecHTC>;
inline VecBHTC makeVecBHTC(size_t B, size_t H, size_t T, size_t C) {
	return makeVecLBTC(B,H,T,C);
}



//Hc is each head's q dim (channels);
using VecBTHc = std::vector<Vecf>;

// Vp is the padded vocab size (for efficiency), V is the "real" vocab size
// example: Vp is 50304 and V is 50257
using VecBTVp = VecBTC;

inline VecBTC makeVecBTVp(size_t B, size_t T, size_t Vp) {
	return makeVecBTC(B, T, Vp);
}
using VecBTV = VecBTC;

inline VecBTC makeVecBTV(size_t B, size_t T, size_t V) {
	return makeVecBTC(B, T, V);
}

inline VecBTC makeZero(size_t B, size_t T, size_t C) {
	return VecBTC(B, Matf::Zero(T, C));
}

inline VecLBTC makeZero(size_t L, size_t B, size_t T, size_t C) {
	return VecLBTC(L, makeZero(B, T, C));
}
using BTCShape = std::tuple<size_t, size_t, size_t>;

inline BTCShape GetShape(const VecBTC shape) {
	return std::make_tuple(shape.size(), shape.front().rows(), shape.front().cols());
}
