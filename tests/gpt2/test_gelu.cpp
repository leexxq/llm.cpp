#include "global.h"

#include <gelu.h>
#include <gtest/gtest.h>

class TestGELU : public testing::Test {
protected:
	GELU gelu;
};

TEST_F(TestGELU, forward1) {
	gelu = GELU();

	VecBTC input(2, Matf(3, 4));
	input[0] << -0.3479, 0.6540, 0.9990, -0.3713,
			-1.7410, 0.9143, -0.1482, 0.9791,
			-1.5080, -0.3592, -0.6735, -0.7169;

	input[1] << 0.9676, 0.4098, -0.8678, 0.1405,
			0.6378, 0.8685, -1.6140, -0.7551,
			-2.0414, -0.2937, 0.6089, -0.2142;

	VecBTC res = gelu.Forward(input);
	VecBTC resexp(2, Matf(3, 4));

	resexp[0] << -0.1266, 0.4862, 0.8402, -0.1319,
			-0.0712, 0.7493, -0.0654, 0.8187,
			-0.0994, -0.1292, -0.1686, -0.1698;

	resexp[1] << 0.8062, 0.2701, -0.1674, 0.0781,
			0.4708, 0.7012, -0.0861, -0.1700,
			-0.0419, -0.1129, 0.4437, -0.0890;

	EXPECT_TRUE(res[0].isApprox(resexp[0], 0.001));
	EXPECT_TRUE(res[1].isApprox(resexp[1], 0.001));
}

TEST_F(TestGELU, backward1) {
	gelu = GELU();

	VecBTC d_outputs = VecBTC(2, Matf::Ones(3, 4));

	VecBTC input(2, Matf(3, 4));

	input[0] << -1.4951, 0.0475, 0.5266, -0.4204,
			0.5671, 0.3195, 1.1151, 0.9379,
			-1.7910, 0.0436, 1.8604, 0.5179;
	input[1] << -0.3848, -0.1624, 0.7606, 1.8875,
			-0.6217, -1.2889, -0.3850, -1.8288,
			-0.7550, -0.3807, -0.5675, -0.4625;

	VecBTC res = gelu.Backward(d_outputs, input);
	VecBTC resexp(2, Matf(3, 4));

	resexp[0] << -1.2786e-01, 5.3787e-01, 8.8351e-01, 1.8365e-01,
			9.0714e-01, 7.4641e-01, 1.1062e+00, 1.0665e+00,
			-1.0778e-01, 5.3477e-01, 1.1009e+00, 8.7828e-01;

	resexp[1] << 2.0770e-01, 3.7156e-01, 1.0035e+00, 1.0981e+00,
			6.2837e-02, -1.2524e-01, 2.0756e-01, -1.0409e-01,
			-1.0901e-03, 2.1051e-01, 9.2629e-02, 1.5617e-01;

	EXPECT_TRUE(res[0].isApprox(resexp[0], 0.001));
	EXPECT_TRUE(res[1].isApprox(resexp[1], 0.001));
}
