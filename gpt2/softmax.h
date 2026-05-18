#pragma once
#include "global.h"

Matf softmax(const Matf &mat);

void softmax(const Matf &mat, Matf &output);
VecBTC softmax(const VecBTC &vmat);
