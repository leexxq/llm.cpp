# llm.cpp
[llm.c](https://github.com/karpathy/llm.c "llm.c") is implemented using modern C++.

- Quickly build using CMake [cmake](https://github.com/kitware/cmake)
- Based on GTest testing framework [googletest](https://github.com/google/googletest)
- Accelerate training using the Eigen library and OpenMP. [eigen](https://github.com/PX4/eigen)
- Optimize performance using pprof [pprof](https://github.com/google/pprof)


## Quick start

```bash
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```


## Faster training

OpenMP (optional):

macOS: 
```bash
brew install libomp
```

Ubuntu
```bash
 sudo apt-get install libomp-dev
```


## Tests
```bash
cd build
make -j gpt2test

```


