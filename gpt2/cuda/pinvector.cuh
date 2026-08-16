#pragma once
#include <vector>
#include <initializer_list>
#include <cuda_runtime.h>
namespace gpt2cuda {
    // custom CUDA pinned memory allocator
    template <typename T>
    struct PinAllocator {
        using value_type = T;

        PinAllocator() noexcept = default;

        template <typename U>
        PinAllocator(const PinAllocator<U>&) noexcept {}

        // 分配内存
        T* allocate(std::size_t n) {
            if (n == 0) {
                return nullptr;
            }
            
            // 防止溢出
            if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
                throw std::bad_array_new_length();
            }

            T* ptr = nullptr;
            cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&ptr), n * sizeof(T), cudaHostAllocDefault);
            
            if (err != cudaSuccess) {
                throw std::bad_alloc();
            }
            
            return ptr;
        }

        // 释放内存
        void deallocate(T* p, std::size_t /*n*/) noexcept {
            if (p) {
                cudaFreeHost(p);
            }
        }
    };

    template <typename T, typename U>
    bool operator==(const PinAllocator<T>&, const PinAllocator<U>&) noexcept {
        return true;
    }

    template <typename T, typename U>
    bool operator!=(const PinAllocator<T>&, const PinAllocator<U>&) noexcept {
        return false;
    }

    template <typename T>
    using pinvector = std::vector<T, PinAllocator<T>>;


    template <class T>
    using StdVec = std::vector<T>;
    template <class T>
    using PinVec = pinvector<T>;




    using PinVecf = PinVec<float>;
    using StdVecf = std::vector<float>;
    using PinVeci = PinVec<int>;
    using StdVeci = std::vector<int>;

    // void operator+= (gpt2cuda::StdVecf v1,gpt2cuda::StdVecf v2){

    //     auto n = v1.size();
    //     for(int i =0 ; i < n ; ++ i){
    //         v1[i] += v2[i];
    //     }
    // }


    template<class T>
    inline auto makePinVecConstant(size_t n,const T x){
        return PinVec<T>(n,x);
    }

    template<class T>
    inline auto makePinVecZero(std::size_t n){
        return makePinVecConstant<T>(n,0);
    }

    template<class T>
    inline auto makePinVec(std::size_t n){
        return PinVec<T>(n);
    }


    inline auto makePinVecfConstant(size_t n,const float x){
        return makePinVecConstant<float>(n,x);
    }

    inline auto makePinVecfZero(std::size_t n){
        return makePinVecfConstant(n,0);
    }

    inline auto makePinVecf(size_t n){
        return makePinVec<float>(n);
    }
}



