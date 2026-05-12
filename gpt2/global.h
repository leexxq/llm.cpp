#pragma once

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#define FORWARD_NO_DISCARD [[nodiscard("result may be used in forward and backward later")]]

#ifdef DEBUG
#define DEBUG_PRINTLN(...) fmt::println(__VA_ARGS__)
#endif
#ifndef DEBUG
#define DEBUG_PRINTLN(...)
#endif

#define INFO_PRINTLN(...) fmt::println(__VA_ARGS__)

#define ERROR_PRINTLN(...) fmt::println(stderr,__VA_ARGS__)


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

using VeciBTC = std::vector<Mati>;

using VecBT = std::vector<Vecf>;

using VecHTC = std::vector<Matf>;

// BHTC is batch , (head , seq_len,channels)
using VecBHTC = std::vector<VecHTC>;

//Hc is each head's q dim (channels);
using VecBTHc = std::vector<Vecf>;

using MaskMat = Matf;
