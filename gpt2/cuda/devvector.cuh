#pragma  once
#include "cutlass/util/device_memory.h"
#include "pinvector.cuh"
#include <vector>
namespace gpt2cuda {
    template <class T>
    class DevVector{
        private:
        using DAlloc = cutlass::device_memory::allocation<T>;
        DAlloc _data;
        public:
        DevVector():_data(){
        }
        DevVector(size_t n ):_data(n){
        }

        explicit DevVector(const PinVec<T>& other):_data(other.size()){
            _data.copy_from_host(other);
        }

        template<class Allocator>
        explicit DevVector(const std::vector<T,Allocator>& other):_data(other.size()){
            _data.copy_from_host(other);
        }
        

        DevVector<T>& operator=(const DevVector<T>& other){
            _data = other._data;
            return *this;
        }
        DevVector<T>& operator=(DevVector<T>&& other){
            _data = other._data;
            return *this;
        }
        
        DevVector(const DevVector<T>& other):_data(other){
        }
        DevVector(DevVector<T>&& other):_data(other){

        }

        T* data() const{
            return _data.get();
        }

    };

    using DevVecf = DevVector<float>;
    using DevVeci = DevVector<int>;
    template<class T>
    using DevVec = DevVector<T>;

    auto makeVecfDevice(size_t n){
        return DevVecf(n);
    }


    auto makeVeciDevice(size_t n){
        return DevVeci(n);
    }

    template<class T>
    auto makeVecDevice(size_t n ){
        return DevVector<T>(n);
    }
}
