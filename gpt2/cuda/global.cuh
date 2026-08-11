#pragma  once
#include <numeric>
#include <iostream>

#include <vector>
#include <initializer_list>
namespace gpt2cuda {

    template <class T>
    using StdVec = std::vector<T>;

    using StdVecf = std::vector<float>;
    using StdVeci = std::vector<int>;

    // void operator+= (gpt2cuda::StdVecf v1,gpt2cuda::StdVecf v2){

    //     auto n = v1.size();
    //     for(int i =0 ; i < n ; ++ i){
    //         v1[i] += v2[i];
    //     }
    // }

    template<class T,class V>
    inline StdVec<T> makeVecConstant(std::initializer_list<V> args,const T x){
        int res = 1;
        for(auto& x : args){
            res *= x;

        }
        return StdVec<T>(res,x);
    }

    template<class T,class V>
    inline StdVec<T> makeZero(std::initializer_list<V> args){
        return makeVecConstant<T>(args,0);
    }

    template<class T,class V>
    inline StdVec<T> makeVec(std::initializer_list<V> args){
        int res = 1;
        for(auto& x : args){
            res *= x;

        }
        return StdVec<T>(res);
    }

}



