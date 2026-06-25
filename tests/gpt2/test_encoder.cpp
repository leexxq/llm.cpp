#include <encoder.h>

#include <gtest/gtest.h>

class TestEncoder : public testing::Test {
protected:
	// static void SetUpTestSuite() {
	// 	std::cout << "run before first case..." << std::endl;
	// }
	//
	// static void TearDownTestSuite() {
	// 	std::cout << "run after last case..." << std::endl;
	// }
	//

	Encoder encoder;

	virtual void SetUp() override {
		// std::cout << "enter into SetUp()" << std::endl;
	}
	//
	// virtual void TearDown() override {
	// 	std::cout << "exit from TearDown" << std::endl;
	// }
};

TEST_F(TestEncoder, forward) {
	Mati input(2, 3);
	input << 0, 2, 3,
			1, 3, 2;

	Matf wte(4, 3);
	wte << 1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			1, 1, 0;

	Matf wpe(3, 3);
	wpe << -1, 0, 0,
			0, -1, 0,
			0, 0, -1;

	VecBTC exp_res(2, Matf(3, 3));
	exp_res[0] << 0, 0, 0,
			0, -1, 1,
			1, 1, -1;

	exp_res[1] << -1, 1, 0,
			1, 0, 0,
			0, 0, 0;

	encoder.wpe = std::move(wpe);
	encoder.wte = std::move(wte);
	VecBTC res = encoder.Forward(input);
	for (int i = 0; i < res.size(); ++i) {
		// std::cout << res[i] << std::endl;
		EXPECT_EQ(res[i], exp_res[i]);
	}
}
