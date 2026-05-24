#include "global.h"

#include <gtest/gtest.h>
#include <layernorm.h>
#include <sys/cdefs.h>

class TestLayerNorm : public testing::Test {
protected:
	LayerNorm layernorm;
};

TEST_F(TestLayerNorm, forward1) {
	layernorm = LayerNorm(2, 3, 4);
	VecBTC inputs{
		Matf{
				{ 0.1262, 0.3419, -0.5983, -0.7482 },
				{ 0.6616, -0.2417, 0.5725, -0.1024 },
				{ -1.3749, -1.0748, -0.1801, -0.0672 } },
		Matf{
				{ 0.0144, 0.3592, -0.8504, -0.4432 },
				{ 1.4541, 0.1279, -0.2581, -0.1408 },
				{ -0.8274, -1.5479, -0.6145, -0.7071 } }
	};

	VecBTC resexp{
		Matf{
				{ 0.7468, 1.2125, -0.8179, -1.1415 },
				{ 1.1008, -1.1638, 0.8775, -0.8146 },
				{ -1.2464, -0.7125, 0.8790, 1.0799 } },

		Matf{
				{ 0.5342, 1.2878, -1.3561, -0.4659 },
				{ 1.6953, -0.2457, -0.8106, -0.6390 },
				{ 0.2633, -1.6952, 0.8418, 0.5901 } }
	};

	VecBTC res = layernorm.Forward(inputs);

	ASSERT_TRUE(res.size() == inputs.size() && res.front().rows() == inputs.front().rows() && res.front().cols() == inputs.front().cols());
	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}

TEST_F(TestLayerNorm, forward2) {
	layernorm = LayerNorm(2, 3, 4);
	VecBTC inputs{
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

	VecBTC resexp{
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

	VecBTC res = layernorm.Forward(inputs);

	ASSERT_TRUE(res.size() == inputs.size() && res.front().rows() == inputs.front().rows() && res.front().cols() == inputs.front().cols());
	for (int i = 0; i < 2; ++i) {
		EXPECT_TRUE(res[i].isApprox(resexp[i], 0.001));
	}
}

TEST_F(TestLayerNorm, backward1) {
	layernorm = LayerNorm(2, 3, 4);

	layernorm.gamma << 16.4667, 10.2935, 11.0739, 12.7998;
	layernorm.beta << -1.7650, -1.4063, 1.6240, 0.3662;
	VecBTC d_outputs = VecBTC(2, Matf::Ones(3, 4));

	VecBTC inputs(2, Matf(3, 4));

	inputs[0] << 0.8351, -52.3470, -120.6479, 79.4512,
			-53.9348, 116.3453, -26.2333, -111.2108,
			95.6808, -122.4990, -139.4510, 52.4164;

	inputs[1] << -48.4344, 102.0699, -32.5717, -21.7057,
			1.3376, 85.8685, 158.5985, 85.5139,
			-41.6447, -225.0275, -35.9537, -88.3156;

	//Used to calculate rstd and mean, which are used in backward ...
	VecBTC output = layernorm.Forward(inputs);
	VecBTC res = layernorm.Backward(d_outputs, inputs);

	VecBTC resexp(2, Matf(3, 4));

	resexp[0] << 0.0470, -0.0262, -0.0012, -0.0196,
			0.0386, -0.0021, -0.0204, -0.0162,
			0.0122, -0.0042, 0.0066, -0.0146;

	resexp[1] << 0.0424, 0.0056, -0.0408, -0.0071,
			0.0168, -0.0406, 0.0195, 0.0042,
			0.0367, -0.0010, -0.0353, -0.0004;

	EXPECT_TRUE(res[0].isApprox(resexp[0], 0.005f)) << "res[0]:" << res[0];
	EXPECT_TRUE(res[1].isApprox(resexp[1], 0.005f)) << "res[1]:" << res[1];

	// std::cout << res[0] << std::endl;
	// std::cout << "------\n";
	// std::cout << res[1] << std::endl;
	// std::cout << "------\n";
	// std::cout << layernorm.d_gamma << std::endl;
	// std::cout << "------\n";
	// std::cout << layernorm.d_beta << std::endl;

	Vecf d_gamma_exp(4);
	d_gamma_exp << -0.4317, 0.4033, -0.8599, 0.8884;
	Vecf d_beta_exp(4);
	d_beta_exp << 6., 6., 6., 6.;

	EXPECT_TRUE(layernorm.d_gamma.isApprox(d_gamma_exp, 0.001f));
	EXPECT_TRUE(layernorm.d_beta.isApprox(d_beta_exp, 0.001f));
}

TEST_F(TestLayerNorm, backward2) {
	layernorm = LayerNorm(2, 3, 4);

	layernorm.gamma << 16.4667, 10.2935, 11.0739, 12.7998;
	layernorm.beta << -1.7650, -1.4063, 1.6240, 0.3662;

	VecBTC d_outputs = makeVecBTC(2, 3, 4);

	d_outputs[0] << 3.7833e+01, 4.0641e-03, 2.0205e-06, 8.8659e+07,
			1.7052e-04, 3.9447e+06, 1.8895e+00, 1.0632e-06,
			6.0932e+07, 2.1885e-05, 3.6637e-05, 3.0876e+04;

	d_outputs[1] << 2.8763e-07, 1.0808e+07, 1.2529e-02, 1.4311e-02,
			5.8303e-12, 4.2980e-01, 1.7804e+07, 2.6734e+00,
			3.1208e+04, 8.4363e-09, 4.0009e+04, 7.0111e+00;

	VecBTC inputs(2, Matf(3, 4));

	inputs[0] << 0.8351, -52.3470, -120.6479, 79.4512,
			-53.9348, 116.3453, -26.2333, -111.2108,
			95.6808, -122.4990, -139.4510, 52.4164;

	inputs[1] << -48.4344, 102.0699, -32.5717, -21.7057,
			1.3376, 85.8685, 158.5985, 85.5139,
			-41.6447, -225.0275, -35.9537, -88.3156;

	//Used to calculate rstd and mean, which are used in backward ...
	VecBTC output = layernorm.Forward(inputs);
	VecBTC res = layernorm.Backward(d_outputs, inputs);

	VecBTC resexp(2, Matf(3, 4));

	resexp[0] << -5.6523e+06, -1.7120e+06, 3.3484e+06, 4.0160e+06,
			-3.9165e+04, 4.8587e+04, -1.0371e+05, 9.4289e+04,
			3.7918e+06, 2.0043e+05, 6.7229e+05, -4.6645e+06,

			resexp[1] << 1.7716e+05, 3.5280e+04, -3.3924e+04, -1.7852e+05,
			8.7831e+05, -9.5110e+05, 1.0162e+06, -9.4342e+05,
			1.8242e+03, 8.9646e+02, 7.1465e+02, -3.4353e+03;

	EXPECT_TRUE(res[0].isApprox(resexp[0], 0.005f)) << "res[0]:" << res[0];
	EXPECT_TRUE(res[1].isApprox(resexp[1], 0.005f)) << "res[1]:" << res[1];

	// std::cout << res[0] << std::endl;
	// std::cout << "------\n";
	// std::cout << res[1] << std::endl;
	// std::cout << "------\n";
	// std::cout << layernorm.d_gamma << std::endl;
	// std::cout << "------\n";
	// std::cout << layernorm.d_beta << std::endl;

	Vecf d_gamma_exp(4);
	d_gamma_exp << 7.2884e+07, 2.4841e+07, 2.4263e+07, 1.2425e+08;
	Vecf d_beta_exp(4);
	d_beta_exp << 60963540., 14752580., 17843962., 88689384.;

	EXPECT_TRUE(layernorm.d_gamma.isApprox(d_gamma_exp, 0.001f));
	EXPECT_TRUE(layernorm.d_beta.isApprox(d_beta_exp, 0.001f));
}
