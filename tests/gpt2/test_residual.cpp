
#include "global.h"

#include <gtest/gtest.h>
#include <residual.h>

class TestResidual : public testing::Test {
protected:
	Residual residual;
};

TEST_F(TestResidual, forward1) {
	residual = Residual();

	VecBTC input1(2, Matf(3, 4));

	input1[0] << 0.7475, 0.2059, -1.0990, 0.7776, -1.5552, 0.6648, 0.0580, -1.2327,
			-0.6292, 0.9570, -1.3339, -0.7725;
	input1[1] << 0.5579, -1.2300, -0.8874, 1.8051, -0.3831, 0.5590, 1.1705, -0.3796,
			0.2364, -0.5235, 0.6003, -0.7753;
	VecBTC input2(2, Matf(3, 4));
	input2[0] << -2.1497, 0.9591, 0.1197, 2.0377, -1.6623, -0.4828, 0.5271, -2.5709,
			0.4096, -1.9807, 0.1540, -0.0264;
	input2[1] << 2.3398, -0.1399, 1.1389, -0.7897, 0.2034, 0.2301, -0.7696, 1.2657,
			-0.8282, -1.1220, 0.1352, -0.4819;

	VecBTC res = residual.Forward(input1, input2);

	VecBTC resexp(2, Matf(3, 4));

	resexp[0] << -1.4022, 1.1650, -0.9793, 2.8153, -3.2175, 0.1820, 0.5851, -3.8036,
			-0.2196, -1.0237, -1.1799, -0.7989;

	resexp[1] << 2.8977, -1.3699, 0.2515, 1.0154, -0.1797, 0.7891, 0.4009, 0.8861,
			-0.5918, -1.6455, 0.7355, -1.2572;

	for (int b = 0; b < 2; ++b) {
		EXPECT_TRUE(res[b].isApprox(resexp[b], 0.001));
	}
}
