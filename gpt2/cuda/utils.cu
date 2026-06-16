#include "utils.cuh"


__global__ void ReduceCuda(float * dst, float const *  src , int const length,  int const stride){
    // using namespace cute;
    // auto gs = make_tensor(make_gmem_ptr(src),make_layout(make_shape(Int<1>{},length)));
    // auto gd = make_tensor(make_gmem_ptr(dst),make_layout(make_shape(Int<1>{},stride)));

    const int bidx = blockIdx.x;
    const int tidx = threadIdx.x; 
    const int warpidx = tidx / warpSize;
    const int lane = tidx % warpSize;
    assert(blockDim.x%warpSize == 0);
    const int warp_nums = blockDim.x / warpSize;
    const int wave_nums = (length + warpSize * stride - 1)/(warpSize * stride);
    const u_int32_t mask = 0xFFFFFFFFU;
    for(int wave = 0 ; wave < wave_nums ; ++wave){
        const int warp_offest = bidx * warp_nums + warpidx ;
        const int wave_offest = warp_offest+ wave * warpSize * stride;
        const int thread_offest = wave_offest +  lane * stride;
        float val = 0;
        if(thread_offest < length){
            val = src[thread_offest];
        }
        for(int offest = warpSize / 2 ; offest > 0 ; offest/=2){
            val += __shfl_down_sync(mask,val,offest);
        }
        if(lane == 0){
            dst[warp_offest] += val;
        }
    }
    
}

void ReduceKernel(dim3 blocks,dim3 threads, float*dst,float const* src , int length,int stride){
    ReduceCuda<<<blocks,threads>>>(dst,src,length,stride);
}