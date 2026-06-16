# llm.cpp
[llm.c](https://github.com/karpathy/llm.c "llm.c") is implemented using modern C++.

- Quickly build using CMake [cmake](https://github.com/kitware/cmake)
- Based on GTest testing framework [googletest](https://github.com/google/googletest)
- Accelerate training using the Eigen library and OpenMP. [eigen](https://github.com/PX4/eigen)
- Optimize performance using pprof [pprof](https://github.com/google/pprof)
- Provides a mindmap to demonstrate training-related architecture and related knowledge. 

![mindmap](https://github.com/leexxq/llm.cpp/blob/main/mindmap.png)



## Quick start


```bash
# Download train datasets
# For Chinese users, this might require a bit of magic.
bash ./download_data_pack.sh

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
./traingpt2
```


## Faster training

### OpenMP (optional)

#### macOS
```bash
brew install libomp
```

#### Ubuntu
```bash
sudo apt-get install libomp-dev
```


## Tests
```bash
cd build
cmake -DENABLE_TESTS=ON ..
make -j gpt2test
./tests/gpt2test
```

## CUDA
### Requirement
1. CUDA version 12.4 or newer
2. [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) 8.0 or newer

### Install CUTLASS Submodule
```bash
git submodule update --init --recursive
```

### Enable CUDA
```bash
cd build
cmake -DENABLE_CUDA=ON ..
```

### CUDA Implement Unit Tests
```bash
cd build
cmake -DENABLE_TESTS=ON ..
make -j gpt2test
# run unit tests
./test/gpt2test
```

