#include "global.h"

#include <cross_entropy.h>
#include <gtest/gtest.h>

class TestCrossEntropy : public testing::Test {
protected:
	CrossEntropy cross_entropy;
};

TEST_F(TestCrossEntropy, forward1) {
	cross_entropy = { 2, 7 };

	Mati targets(2, 7);

	targets << 0, 1, 2, 0, 1, 1, 3,
			1, 0, 3, 0, 3, 2, 0;

	CrossEntropy::VecBTV probs(2, Matf(7, 5));
	probs[0] << 0.1281, 0.1188, 0.1115, 0.1407, 0.5010,
			0.2251, 0.0568, 0.1865, 0.1436, 0.3881,
			0.3194, 0.0684, 0.0602, 0.2614, 0.2906,
			0.2173, 0.1671, 0.3586, 0.0530, 0.2039,
			0.0187, 0.3118, 0.4629, 0.0777, 0.1289,
			0.1686, 0.5187, 0.0278, 0.1018, 0.1831,
			0.0379, 0.5042, 0.0586, 0.2128, 0.1864;

	probs[1] << 0.0716, 0.0988, 0.4528, 0.0350, 0.3419,
			0.4144, 0.0210, 0.1202, 0.1327, 0.3117,
			0.1012, 0.0109, 0.1775, 0.7053, 0.0051,
			0.3464, 0.0400, 0.2225, 0.1741, 0.2170,
			0.1679, 0.0997, 0.3373, 0.1473, 0.2478,
			0.0419, 0.1059, 0.0407, 0.2348, 0.5767,
			0.1465, 0.1852, 0.0784, 0.4683, 0.1217;

	float loss = cross_entropy.Forward(probs, targets);

	float lossexp = (1.8042f + 1.6631f) / 2;

	EXPECT_NEAR(cross_entropy.losses[0].sum() / 7, 1.8042f, 1e-3f);
	EXPECT_NEAR(cross_entropy.losses[1].sum() / 7, 1.6631f, 1e-3f);

	EXPECT_NEAR(loss, lossexp, 1e-3f);
}
