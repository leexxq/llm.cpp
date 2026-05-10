#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#define FORWARD_NO_DISCARD [[nodiscard("result may be used in forward and backward later")]]

using Matf = Eigen::MatrixXf;

using Vecf = Eigen::VectorXf;

using Veci = Eigen::VectorXi;

using VecBtc = std::vector<Matf>;

using VecBt = std::vector<Vecf>;
