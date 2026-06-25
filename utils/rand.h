#pragma once
#include <random>
#include <vector>
template<class T>
using StdVec = std::vector<T>;
template <class T>
void InitIdentityPermutation(StdVec<T> &data) {
	std::iota(data.begin(), data.end(), 0);
}
template <class T>
void RandomPermutation(StdVec<T> &data, std::mt19937& rng) {
	for (int i = data.size() - 1; i > 0; i--) {
		std::uniform_int_distribution<int> distri{ 0, i };
		// pick an index j in [0, i] with equal probability
		int j = distri(rng);
		// swap i <-> j
		std::swap(data[i], data[j]);
	}
}
