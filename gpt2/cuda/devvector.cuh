#pragma  once
#include "cutlass/util/device_memory.h"
#include <stdexcept>
#include <vector>
#include "error.cuh"

namespace gpt2cuda {
    template <class T>
    class DevVector{
        private:
        using DAlloc = cutlass::device_memory::allocation<T>;
        DAlloc data_;
        public:
        DevVector():data_(){
        }
        DevVector(size_t n ):data_(n){
        }

        // explicit DevVector(const PinVec<T>& other):data_(other.size()){
        //     data_.copy_from_host(other.data());
        // }
        explicit DevVector(const T* other, size_t n):data_(n){
            if(n > 0){
                data_.copy_from_host(other);
            }
        }

        template<class Allocator>
        explicit DevVector(const std::vector<T,Allocator>& other):data_(other.size()){
            if(other.size() > 0){
                data_.copy_from_host(other.data());
            }
        }

        template<class Allocator>
        DevVector<T>& operator=(const std::vector<T,Allocator>& other){
            if(other.size() != data_.size()){
                throw std::invalid_argument("DevVector size mismatch");
            }
            if(size() > 0){
                data_.copy_from_host(other.data());
            }
            return *this;
        }
        

        DevVector<T>& operator=(const DevVector<T>& other){
            data_ = other.data_;
            return *this;
        }

        DevVector<T>& operator=(DevVector<T>&& other){
            data_ = other.data_;
            return *this;
        }
        
        DevVector(const DevVector<T>& other):data_(other.data_){
        }
        DevVector(DevVector<T>&& other):data_(other.data_){

        }

        T* data() noexcept {
            return data_.get();
        }
        const T* data() const noexcept{
            return data_.get();
        }

        size_t size() const noexcept{
            return data_.size();
        }
        bool empty() const noexcept{
            return size() == 0;
        }

        void zero(){
            CUDA_CHECK(cudaMemset(data(), 0, size()*sizeof(T)));
        }
        void zero(cudaStream_t stream){
            CUDA_CHECK(cudaMemsetAsync(data(), 0, size()*sizeof(T),stream));
        }

        template<class Allocator>
        void to(std::vector<T,Allocator>& other)const{
            data_.copy_to_host(other.data());
        }
        
        template<class Allocator>
        void to(std::vector<T,Allocator>& other,cudaStream_t stream)const{
            to(other.data(),stream);
        }

        void to(T* dst_data,cudaStream_t stream = 0)const {
            CUDA_CHECK(cudaMemcpyAsync(dst_data,this->data(),size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
        }
        void from(const T* src_data,cudaStream_t stream = 0){
            CUDA_CHECK(cudaMemcpyAsync(this->data(),src_data,size()*sizeof(T),cudaMemcpyHostToDevice,stream));
        }

        void from(const DevVector<T>& other,cudaStream_t stream = 0){
            CUDA_CHECK(cudaMemcpyAsync(this->data(),other.data(),size()*sizeof(T),cudaMemcpyDeviceToDevice,stream));
        }
    };

    using DevVecf = DevVector<float>;
    using DevVeci = DevVector<int>;
    template<class T>
    using DevVec = DevVector<T>;

    inline auto makeDevVecf(size_t n){
        return DevVecf(n);
    }

    inline auto makeDevVecfZero(size_t n){
        auto v = DevVecf(n);
        v.zero();
        return v;
    }



    inline auto makeDevVeci(size_t n){
        return DevVeci(n);
    }

    template<class T>
    inline auto makeDevVec(size_t n ){
        return DevVector<T>(n);
    }
}
