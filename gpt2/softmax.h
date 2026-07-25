#pragma once
#include "global.h"

FORWARD_NO_DISCARD
Matf Softmax(const Matf &mat);

void Softmax(const Matf &mat, Matf &output);

FORWARD_NO_DISCARD
VecBTC Softmax(const VecBTC &vmat);
