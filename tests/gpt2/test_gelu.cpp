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
