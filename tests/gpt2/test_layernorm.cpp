#include "../../gpt2/layernorm.h"

#include <gtest/gtest.h>

class TestLayerNorm : public testing::Test {
protected:
	LayerNorm layernorm;
};

TEST_F(TestLayerNorm, forward1) {
	layernorm = LayerNorm(2, 3, 4);
	VecBtc input{
		Matf{
				{ 0.1262, 0.3419, -0.5983, -0.7482 },
				{ 0.6616, -0.2417, 0.5725, -0.1024 },
				{ -1.3749, -1.0748, -0.1801, -0.0672 } },
		Matf{
				{ 0.0144, 0.3592, -0.8504, -0.4432 },
				{ 1.4541, 0.1279, -0.2581, -0.1408 },
				{ -0.8274, -1.5479, -0.6145, -0.7071 } }
	};

	VecBtc resexp{
		Matf{
				{ 0.7468, 1.2125, -0.8179, -1.1415 },
				{ 1.1008, -1.1638, 0.8775, -0.8146 },
				{ -1.2464, -0.7125, 0.8790, 1.0799 } },

		Matf{
				{ 0.5342, 1.2878, -1.3561, -0.4659 },
				{ 1.6953, -0.2457, -0.8106, -0.6390 },
				{ 0.2633, -1.6952, 0.8418, 0.5901 } }
	};

	VecBtc res = layernorm.forward(input);

	ASSERT_TRUE(res.size() == input.size() && res.front().rows() == input.front().rows() && res.front().cols() == input.front().cols());
	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}

TEST_F(TestLayerNorm, forward2) {
	layernorm = LayerNorm(2, 3, 4);
	VecBtc input{
		Matf{
				{ -0.3190, -0.8180, -1.9410, 0.6254 },
				{ -0.6764, -0.7952, -0.6143, 0.4818 },
				{ -1.9371, 1.9170, -0.3365, 0.7066 },
		},
		Matf{
				{ 1.1318, -0.1032, -0.9494, 2.6337 },
				{ -0.2884, -0.4178, -1.5289, -2.0351 },
				{ -0.9120, 2.2307, 0.4144, 0.4242 },
		}
	};

	VecBtc resexp{
		Matf{
				{ 0.3179, -0.2214, -1.4348, 1.3383 },
				{ -0.5359, -0.7671, -0.4151, 1.7181 },
				{ -1.4308, 1.2929, -0.2997, 0.4375 },

		},
		Matf{
				{ 0.3360, -0.5788, -1.2057, 1.4486 },
				{ 1.0558, 0.8805, -0.6252, -1.3111 },
				{ -1.2986, 1.5134, -0.1118, -0.1030 },
		}
	};

	VecBtc res = layernorm.forward(input);

	ASSERT_TRUE(res.size() == input.size() && res.front().rows() == input.front().rows() && res.front().cols() == input.front().cols());
	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}
