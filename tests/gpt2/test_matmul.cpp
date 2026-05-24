#include "global.h"

#include <gtest/gtest.h>
#include <matmul.h>

class TestMatMul : public testing::Test {
protected:
	MatMul matmul;
};

TEST_F(TestMatMul, forward1) {
	matmul = MatMul();
	VecBTC inputs{

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

	VecBTC res = matmul.Forward(inputs);

	ASSERT_TRUE(res.size() == inputs.size() && res.front().rows() == inputs.front().rows() && res.front().cols() == matmul.weight.cols());

	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}

TEST_F(TestMatMul, backward1) {
	matmul = MatMul(4, 5);
	VecBTC inputs = makeVecBTC(2, 3, 4);
	inputs[0] << 1.9170, 0.7433, -1.6831, -0.5770,
			0.5002, -0.1631, -0.6377, 0.1790,
			0.8014, 0.9574, 0.5211, 0.6669;

	inputs[1] << 0.0507, -0.7415, 0.7975, -0.1952,
			-0.0915, -0.2075, -0.6837, -1.4144,
			-1.0925, -0.9400, 2.0981, 2.5490;

	matmul.weight << 0.5496, -1.0675, -0.3498, -1.7053, -0.0094,
			0.8118, 0.6951, 1.7514, -0.0827, -0.7742,
			0.6460, -0.5852, 1.0690, 2.4317, -1.1704,
			0.0139, -0.5536, -0.0932, 0.6729, -0.1479;
	matmul.bias << -0.5602, -0.6464, -0.2690, 0.5687, -0.1302;

	VecBTC d_outputs = VecBTC(2, Matf::Ones(3, 5));

	VecBTC res = matmul.Backward(d_outputs, inputs);

	VecBTC resexp = makeVecBTC(2, 3, 4);

	resexp[0] << -2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079;

	resexp[1] << -2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079,
			-2.5823, 2.4014, 2.3912, -0.1079;

	EXPECT_TRUE(res[0].isApprox(resexp[0], 0.001f));
	EXPECT_TRUE(res[1].isApprox(resexp[1], 0.001f));

	Matf d_weightexp(4, 5);
	d_weightexp << 2.0853, 2.0853, 2.0853, 2.0853, 2.0853,
			-0.3514, -0.3514, -0.3514, -0.3514, -0.3514,
			0.4122, 0.4122, 0.4122, 0.4122, 0.4122,
			1.2083, 1.2083, 1.2083, 1.2083, 1.2083;

	Vecf d_biasexp(5);
	d_biasexp << 6., 6., 6., 6., 6.;

	EXPECT_TRUE(matmul.d_weight.isApprox(d_weightexp, 0.001f));
	EXPECT_TRUE(matmul.d_bias.isApprox(d_biasexp, 0.001f));
}

// TEST(TestMatMulGlobalFuction,forward){
//
// }
//
// TEST(TestMatMulGlobalFuction,backward){
//
// }



