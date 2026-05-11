#include <matmul.h>

#include <gtest/gtest.h>

class TestMatMul : public testing::Test {
protected:
	MatMul matmul;
};

TEST_F(TestMatMul, forward) {
	VecBTC input{

		Matf{
				{ -0.6913, 1.6103, 0.1138, 0.3595 },
				{ 0.2048, -0.8102, -1.1874, -0.0937 },
				{ 1.1244, 0.5758, -0.0924, -0.8170 },
		},
		Matf{
				{ -1.9811, 1.1705, -0.0687, -1.4556 },
				{ 0.0356, -0.5523, 1.0658, -0.6316 },
				{ 1.2820, 0.3895, 0.0981, 0.0935 },

		},

	};
	matmul.weight = Matf{
		{ 1.6394, -2.9037, -1.0096, -0.4072, 0.5853 },
		{ 0.7580, 0.9567, -0.9137, -0.6394, -0.4959 },
		{ -1.3069, 0.6839, 1.1314, -0.0770, 0.8897 },
		{ -0.0154, 0.7099, -0.7526, 0.8421, -1.2247 },
	};
	matmul.bias = Vecf(5);

	matmul.bias << -1.3034, -0.2438, 0.8006, -1.0919, 1.3184;

	VecBTC resexp{
		Matf{
				{ -1.3704, 3.6371, -0.1147, -1.5461, -0.2238 },
				{ -0.0284, -2.4923, 0.0611, -0.6447, 0.8983 },
				{ 1.1098, -3.6011, -0.3504, -2.5988, 2.6093 },
		},
		Matf{
				{ -3.5517, 5.5484, 2.7489, -2.2541, 1.2999 },
				{ -3.0468, -0.5951, 2.9504, -1.3673, 3.3349 },
				{ 0.9637, -3.4602, -0.8088, -1.7918, 1.8485 },
		}
	};

	VecBTC res = matmul.Forward(input);

	ASSERT_TRUE(res.size() == input.size() && res.front().rows() == input.front().rows() && res.front().cols() == matmul.weight.cols());

	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}
