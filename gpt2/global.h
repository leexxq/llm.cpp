#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#define FORWARD_NO_DISCARD [[nodiscard("result may be used in forward and backward later")]]

/**
 * BHTC is (batch, head,seq_len,channels) , (seq_len,channels) is matrix
 * Hc is head_channels that's attention head's q dim;
 * contains some
 * **/

using Matf = Eigen::MatrixXf;

using Vecf = Eigen::VectorXf;

using Veci = Eigen::VectorXi;

using VecBTC = std::vector<Matf>;

using VecBT = std::vector<Vecf>;

using VecHTC = std::vector<Matf>;

// BHTC is batch , (head , seq_len,channels)
using VecBHTC = std::vector<VecHTC>;

//Hc is each head's q dim (channels);
using VecBTHc = std::vector<Vecf>;

