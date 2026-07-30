#include "global.h"
#include "gpt2.h"
#include "cuda/gpt2cuda.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <chrono>


using MatfRow = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using MatiRow = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


