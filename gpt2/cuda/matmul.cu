#include "cute/arch/copy.hpp"
#include "cutlass/util/device_memory.h"
#include "matmul.cuh"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cute/tensor.hpp>
namespace gpt2cuda{
namespace kernel{
    using namespace cute;
    template<class STensor>
    __forceinline__ __device__ auto convert_to_rowcol(STensor& tensor){
        Layout flat_layout = flatten(tensor.layout());//(x,y, M,N)
        auto row_layout = append(select<1,2>(flat_layout),select<0,3>(flat_layout));
        return group_modes<0,2>(make_tensor(tensor.data(),row_layout));
    }
    template <int threads = 128,int Br = 64, int Bc = 32>
    __global__ void MatrixColSumKernel(float* output , float const* input,int M, int N){
        Layout I_layout = make_layout(make_shape(M,N),make_stride(N,_1{}));//(M,N)
        Tensor I_tensor = make_tensor(make_gmem_ptr(input),I_layout);

        Layout O_layout = make_layout(make_shape(N),make_stride(_1{}));
        Tensor O_tensor = make_tensor(make_gmem_ptr(output),O_layout);//(N)



        using threads_per_col =  Int<32 / (128 / (sizeof(float) * 8))>;
        using threads_per_row = Int<threads / threads_per_col{}>;

        using S2RCopyAtom = Copy_Atom<AutoVectorizingCopy,float>;
        using S2RCpThrLayout = Layout<Shape<threads_per_row,threads_per_col>,Stride<_1,threads_per_row>>; 

        TiledCopy s2r_tiled_cp = make_tiled_copy(S2RCopyAtom{},S2RCpThrLayout{},Layout<Shape<_4,_1>>{});

        
        __shared__ float smemI[Br*Bc];
        __shared__ float smemO[Bc];

        Tensor sI = make_tensor(make_smem_ptr(smemI),make_layout(make_shape(Int<Br>{},Int<Bc>{}),LayoutRight{}));//(Br,Bc)

        using SwzFn = Swizzle<5,0,5>;
        auto sI_swz_layout = composition(SwzFn{},sI.layout());
        Tensor sI_swz = make_tensor(make_smem_ptr(smemI),sI_swz_layout);//(Br,Bc)


        Tensor gI_first = local_tile(I_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(_0{},blockIdx.x));//(Br,Bc)

        using G2SCpThrLayout = Layout<Shape<Int<threads/32>,_32>,Stride<_32,_1>>;
        Tensor tsI = local_partition(sI_swz,G2SCpThrLayout{},threadIdx.x);
        Tensor tgI_first = local_partition(gI_first, G2SCpThrLayout{},threadIdx.x);
        
        
        copy(AutoCopyAsync{},tgI_first,tsI);
        // copy(tgI_first,tsI);
        cp_async_fence();


        Tensor I_identity_tensor = make_identity_tensor(make_shape(Int<Br>{},Int<Bc>{}));
        ThrCopy s2r_thr_cp = s2r_tiled_cp.get_slice(threadIdx.x);
        Tensor tOsI = s2r_thr_cp.partition_S(sI_swz);
        Tensor tOrI = s2r_thr_cp.retile_D(make_fragment_like<float>(shape(tOsI)));
        Tensor I_identity_frag = s2r_thr_cp.partition_S(I_identity_tensor);

        Tensor col_tOrI = convert_to_rowcol(tOrI);
        Tensor col_sums = make_tensor<float>(size<1>(col_tOrI));
        Tensor col_I_identity_frag= convert_to_rowcol(I_identity_frag);

        clear(col_sums);

        int Tr = M/Br;
        // if(thread0()){
        //     print("cur Tr = ");
        //     print(Tr);
        //     print("\n");
        // }
        for(int k =0 ; k < Tr ; ++k){

            cp_async_wait<0>();
            __syncthreads();


            //copy smem to register
            copy(s2r_tiled_cp,tOsI,tOrI);

            // prefetch next block
            {
                if(k < Tr - 1){
                    Tensor gI = local_tile(I_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(k + 1,blockIdx.x));
                    Tensor tgI = local_partition(gI, G2SCpThrLayout{},threadIdx.x);
                    // copy(AutoCopyAsync{},tgI,tsI);
                    copy(AutoCopyAsync{},tgI,tsI);
                    cp_async_fence();
                }
            }

            // compute sum
            for(int c =0 ; c < size<1>(col_tOrI); ++c){
                for(int r =0 ; r < size<0>(col_tOrI); ++r){
                    col_sums(c) += col_tOrI(r,c);
                }
                for(int off = 1 ; off < threads_per_row{};off<<=1){
                    col_sums(c) += __shfl_xor_sync(0xffffffffU,col_sums(c),off);
                }
            }

            // {
            //     __syncthreads();
            //     if(thread0()){
            //         print_tensor(local_tile(I_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(k,blockIdx.x)));
            //         print_tensor(sI_swz);
            //         print("col_sums:") ;
            //         print_tensor(col_sums);
            //         print_tensor(tOrI);
            //         print_tensor(col_tOrI);
            //         print_tensor(col_I_identity_frag);
            //     }
            //     __syncthreads();
            // }

            if(get<0>(col_I_identity_frag(_0{})) != _0{}){
                clear(col_sums);
            }

        }
        Tensor sO = make_tensor(make_smem_ptr(smemO),make_layout(make_shape(Int<Bc>{})));

        if(get<0>(col_I_identity_frag(0)) == _0{}){
            for(int c= 0 ; c < size(col_sums); ++c){
                sO(get<1>(col_I_identity_frag(0,c))) = col_sums(c);
            }
        }
        __syncthreads();

        // {
        //     if(thread0()){
        //         print_tensor(local_tile(I_tensor,make_tile(Int<Br>{},Int<Bc>{}),make_coord(Tr-1,blockIdx.x)));
        //         print_tensor(sI_swz);
        //         print("col_sums:") ; print_tensor(col_sums);
        //         print_tensor(tOrI);
        //         print_tensor(col_tOrI);
        //         print_tensor(col_I_identity_frag);
        //     }
        //     __syncthreads();
        //      print_tensor(sO);
        // }
        

        Tensor gO = local_tile(O_tensor,make_tile(Int<Bc>{}),make_coord(blockIdx.x));
        Tensor local_O_identity_tensor = make_identity_tensor(make_shape(Int<Bc>{}));

        using S2GCopyAtom = Copy_Atom<UniversalCopy<uint128_t>,float>;
        using S2GCpThrLayout = Layout<Shape<Int<threads>>>; 
        TiledCopy s2g_tiled_cp = make_tiled_copy(S2GCopyAtom{},S2GCpThrLayout{},Layout<Shape<_4>>{});

        ThrCopy s2g_thr_cp = s2g_tiled_cp.get_slice(threadIdx.x);
        Tensor tsO = s2g_thr_cp.partition_S(sO);
        Tensor tgO = s2g_thr_cp.partition_D(gO);
        

        Tensor local_O_identity  = s2g_thr_cp.partition_D(local_O_identity_tensor);
        CUTE_UNROLL 
        for(int i = 0 ; i < size<1>(tsO) ; ++i){
            if(elem_less(get<0>(local_O_identity(_0{},i)),Bc)){
                copy(s2g_tiled_cp,tsO,tgO);
            }
        }
        // if(thread0()){
        //     print_tensor(O_tensor);
        // }
    }
    template <int Br = 64 , int Bc = 32>
    void MatrixColSumCUDA(float* output , float const* input,int M, int N,cudaStream_t stream = cudaStreamDefault){
        constexpr int threads = 128;
        assert(M > 0 && N % Bc == 0);
        assert(N > 0 && M % Br ==0);

        kernel::MatrixColSumKernel<128,Br,Bc><<<cute::ceil_div(N, Bc), threads,0,stream>>>(output, input, M, N);
        CUDA_CHECK_LAST();
    }
}

}

    

template<typename LayoutA = cutlass::layout::RowMajor,typename LayoutB = cutlass::layout::RowMajor>
void BatchMatmulForward(float * outputs_d, float const *  inputs_d , float const * weight_d, float const * bias_d ,int B, int T,int  C,int Oc,cudaStream_t stream = 0){
    // // The code section below describes datatype for input, output matrices and computation between
    // // elements in input matrices.
    using ElementAccumulator = float;                   // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = float;              // <- data type of elements in input matrix A
    using ElementInputB = float;              // <- data type of elements in input matrix B
    using ElementOutput = float;                        // <- data type of elements in output matrix D


    using ThreadblockShape = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::ThreadblockShape;

    using WarpShape = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::WarpShape;
    /// Warp-level tile size (concept: GemmShape)

    using InstructionShape= cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::InstructionShape;
    if(bias_d != nullptr){
        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 
        1,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;


        using Gemm = cutlass::gemm::device::GemmBatched<
            ElementInputA, LayoutA,
            ElementInputB, LayoutB,
            ElementOutput, cutlass::layout::RowMajor,
            ElementAccumulator,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp
        >;

        Gemm gemm;

        cutlass::gemm::GemmCoord problem_size(T, Oc, C);

        
        auto A_m = cutlass::make_TensorRef(inputs_d,LayoutA::packed({T,C}));
        auto B_m = cutlass::make_TensorRef(weight_d,LayoutB::packed({C,Oc}));
        auto C_m = cutlass::make_TensorRef(bias_d, cutlass::layout::RowMajor());
        auto D_m = cutlass::make_TensorRef(outputs_d, cutlass::layout::RowMajor::packed({T,Oc}));

        typename Gemm::Arguments argument{
            problem_size,
            A_m,
            T*C,
            B_m,
            0,
            C_m,
            0,
            D_m,
            T*Oc,
            {1},
            B
        };

        gemm(argument,nullptr,stream);
    }else {

        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 
        1,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::Nothing>;


        using Gemm = cutlass::gemm::device::GemmBatched<
            ElementInputA, LayoutA,
            ElementInputB, LayoutB,
            ElementOutput, cutlass::layout::RowMajor,
            ElementAccumulator,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp
        >;

        Gemm gemm;

        cutlass::gemm::GemmCoord problem_size(T, Oc, C);

        
        auto A_m = cutlass::make_TensorRef(inputs_d,LayoutA::packed({T,C}));
        auto B_m = cutlass::make_TensorRef(weight_d,LayoutB::packed({C,Oc}));
        auto D_m = cutlass::make_TensorRef(outputs_d, cutlass::layout::RowMajor::packed({T,Oc}));


        typename Gemm::Arguments argument{
            problem_size,
            A_m,
            T*C,
            B_m,
            0,
            D_m,
            T*Oc,
            D_m,
            T*Oc,
            {1},
            B
        };

        gemm(argument,nullptr,stream);
    }
}

template<typename LayoutA = cutlass::layout::RowMajor,typename LayoutB = cutlass::layout::RowMajor>
void BatchMatmulBackward(float * d_inputs_d, float* d_weight_d, float* d_bias_d,float const * d_outputs_d, float const *  inputs_d , float const * weight_d,int B, int T,int  C,int Oc,cudaStream_t stream = 0){

    // // The code section below describes datatype for input, output matrices and computation between
    // // elements in input matrices.
    using ElementAccumulator = float;                   // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = float;              // <- data type of elements in input matrix A
    using ElementInputB = float;              // <- data type of elements in input matrix B
    using ElementOutput = float;                        // <- data type of elements in output matrix D


    using ThreadblockShape = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::ThreadblockShape;

    using WarpShape = cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::WarpShape;
    /// Warp-level tile size (concept: GemmShape)

    using InstructionShape= cutlass::gemm::device::DefaultGemmConfiguration<
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
        ElementAccumulator>::InstructionShape;
    /// Epilogue output operator
    using alloctor = cutlass::device_memory::allocation<float>;

    {
        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 
            1,
            ElementAccumulator,
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::Nothing>;


        using GemmABT = cutlass::gemm::device::GemmBatched<
            ElementInputA, LayoutA,
            ElementInputB, typename  cutlass::layout::LayoutTranspose<LayoutB>::type,
            ElementOutput, cutlass::layout::RowMajor,
            ElementAccumulator,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp
        >;

        GemmABT gemm;

        cutlass::gemm::GemmCoord problem_size(T, C, Oc);

        
        auto A_m = cutlass::make_TensorRef(d_outputs_d,LayoutA::packed({T,Oc}));
        //d_input = d_output @ weight's transpose
        auto B_m = cutlass::make_TensorRef(weight_d,cutlass::layout::LayoutTranspose<LayoutB>::type::packed({Oc,C}));
        auto C_m = cutlass::make_TensorRef(d_inputs_d,cutlass::layout::RowMajor::packed({T,C}));

        typename GemmABT::Arguments argument{
            problem_size,
            A_m,
            T*Oc,
            B_m,
            0,
            C_m,
            T*C,
            C_m,
            T*C,
            {1},
            B
        };

        auto status = gemm(argument,nullptr,stream);
        if(status != cutlass::Status::kSuccess){
            std::cerr << cutlass::cutlassGetStatusString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }


    {
        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementOutput, 
            1,
            ElementAccumulator,
            ElementComputeEpilogue,
            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;


        using GemmATB = cutlass::gemm::device::Gemm<
            ElementInputA, typename  cutlass::layout::LayoutTranspose<LayoutA>::type,
            ElementInputB, LayoutB,
            ElementOutput, cutlass::layout::RowMajor,
            ElementAccumulator,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp
        >;

        GemmATB gemm;

        cutlass::gemm::GemmCoord problem_size(C, Oc, T);

        
        //d_weight= sum( input's transpose @ d_output )

        for(int i = 0 ;i < B ; ++i){


            auto A_m = cutlass::make_TensorRef(inputs_d +i *T*C ,cutlass::layout::LayoutTranspose<LayoutA>::type::packed({C,T}));
            auto B_m = cutlass::make_TensorRef(d_outputs_d + i * T*Oc,LayoutB::packed({T,Oc}));
            auto C_m = cutlass::make_TensorRef(d_weight_d,cutlass::layout::RowMajor::packed({C,Oc}));

            typename GemmATB::Arguments argument{
                problem_size,
                A_m,
                B_m,
                C_m,
                C_m,
                {1},
            };
            auto status = gemm(argument,nullptr,stream);

            if(status != cutlass::Status::kSuccess){
                std::cerr << cutlass::cutlassGetStatusString(status) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    }

    if(d_bias_d != nullptr){
        gpt2cuda::kernel::MatrixColSumCUDA(d_bias_d, d_outputs_d, B*T,Oc,stream);
    }
}

void gpt2cuda::BatchMatmulNNForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc){

    using alloctor = cutlass::device_memory::allocation<float>;
    alloctor  inputs_d(B*T *C);
    alloctor  weight_d(C*Oc);
    alloctor  outputs_d(B*T *Oc);
    alloctor  bias_d;
    inputs_d.copy_from_host(inputs);
    weight_d.copy_from_host(weight);

    if(bias != nullptr){
        bias_d = alloctor(Oc);
        bias_d.copy_from_host(bias);
        BatchMatmulForward(outputs_d.get(), inputs_d.get(), weight_d.get(), bias_d.get(), B, T, C, Oc);
    }else{
        BatchMatmulForward(outputs_d.get(), inputs_d.get(), weight_d.get(), nullptr, B, T, C, Oc);
    }
    outputs_d.copy_to_host(outputs);
}
void gpt2cuda::BatchMatmulNTForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc){
    using alloctor = cutlass::device_memory::allocation<float>;
    alloctor  inputs_d(B*T *C);
    alloctor  weight_d(C*Oc);
    alloctor  outputs_d(B*T *Oc);
    alloctor  bias_d;
    inputs_d.copy_from_host(inputs);
    weight_d.copy_from_host(weight);

    if(bias != nullptr){
        bias_d = alloctor(Oc);
        bias_d.copy_from_host(bias);
        BatchMatmulForward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs_d.get(), inputs_d.get(), weight_d.get(), bias_d.get(), B, T, C, Oc);
    }else{

        BatchMatmulForward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs_d.get(), inputs_d.get(), weight_d.get(), nullptr, B, T, C, Oc);
    }
    outputs_d.copy_to_host(outputs);

}
void gpt2cuda::BatchMatmulNNBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc){

    using alloctor = cutlass::device_memory::allocation<float>;
    alloctor  weight_d;
    alloctor  inputs_d;
    alloctor  d_outputs_d;
    alloctor  d_inputs_d;
    alloctor  d_weight_d;
    try {
        weight_d  = alloctor(C*Oc);
        inputs_d  = alloctor(B*T*C);
        d_outputs_d = alloctor(B*T *Oc);
        d_inputs_d = alloctor(B*T *C);
        d_weight_d = alloctor(C*Oc);
    } catch (const cutlass::cuda_exception& e) {
        std::cerr<<"CUDA memory allocll error: " << e.what() << std::endl;
    }


    d_outputs_d.copy_from_host(d_outputs);
    inputs_d.copy_from_host(inputs);
    weight_d.copy_from_host(weight);
    d_inputs_d.copy_from_host(d_inputs);
    d_weight_d.copy_from_host(d_weight);
    if(d_bias != nullptr){
        alloctor d_bias_d(Oc);
        d_bias_d.copy_from_host(d_bias);
        BatchMatmulBackward(d_inputs_d.get(),d_weight_d.get(),d_bias_d.get(),d_outputs_d.get(),inputs_d.get(),weight_d.get(),B,T,C,Oc);
        d_bias_d.copy_to_host(d_bias);
    }else{
        BatchMatmulBackward(d_inputs_d.get(),d_weight_d.get(),nullptr,d_outputs_d.get(),inputs_d.get(),weight_d.get(),B,T,C,Oc);
    }
    d_inputs_d.copy_to_host(d_inputs);
    d_weight_d.copy_to_host(d_weight);

}
void gpt2cuda::BatchMatmulNTBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc){


    using alloctor = cutlass::device_memory::allocation<float>;
    alloctor  weight_d;
    alloctor  inputs_d;
    alloctor  d_outputs_d;
    alloctor  d_inputs_d;
    alloctor  d_weight_d;
    try {
        weight_d  = alloctor(C*Oc);
        inputs_d  = alloctor(B*T*C);
        d_outputs_d = alloctor(B*T *Oc);
        d_inputs_d = alloctor(B*T *C);
        d_weight_d = alloctor(C*Oc);
    } catch (const cutlass::cuda_exception& e) {
        std::cerr<<"CUDA memory allocll error: " << e.what() << std::endl;
    }


    d_outputs_d.copy_from_host(d_outputs);
    inputs_d.copy_from_host(inputs);
    weight_d.copy_from_host(weight);
    d_inputs_d.copy_from_host(d_inputs);
    d_weight_d.copy_from_host(d_weight);
    if(d_bias != nullptr){
        alloctor d_bias_d(Oc);
        d_bias_d.copy_from_host(d_bias);
        BatchMatmulBackward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(d_inputs_d.get(),d_weight_d.get(),d_bias_d.get(),d_outputs_d.get(),inputs_d.get(),weight_d.get(),B,T,C,Oc);
        d_bias_d.copy_to_host(d_bias);
    }else{
        BatchMatmulBackward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(d_inputs_d.get(),d_weight_d.get(),nullptr,d_outputs_d.get(),inputs_d.get(),weight_d.get(),B,T,C,Oc);
    }
    d_inputs_d.copy_to_host(d_inputs);
    d_weight_d.copy_to_host(d_weight);
}

void gpt2cuda::BatchMatmulGeluForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T,int  C,int Oc){
    if(bias !=nullptr){
        // // The code section below describes datatype for input, output matrices and computation between
        // // elements in input matrices.
        using ElementAccumulator = float;                   // <- data type of accumulator
        using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
        using ElementInputA = float;              // <- data type of elements in input matrix A
        using ElementInputB = float;              // <- data type of elements in input matrix B
        using ElementOutput = float;                        // <- data type of elements in output matrix D


        using ThreadblockShape = cutlass::gemm::device::DefaultGemmConfiguration<
            cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
            ElementAccumulator>::ThreadblockShape;

        using WarpShape = cutlass::gemm::device::DefaultGemmConfiguration<
            cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
            ElementAccumulator>::WarpShape;
        /// Warp-level tile size (concept: GemmShape)

        using InstructionShape= cutlass::gemm::device::DefaultGemmConfiguration<
            cutlass::arch::OpClassSimt, cutlass::arch::Sm80, ElementInputA, ElementInputB, ElementOutput,
            ElementAccumulator>::InstructionShape;
        /// Epilogue output operator
        using alloctor = cutlass::device_memory::allocation<float>;

        alloctor  inputs_d(B*T *C);
        alloctor  weight_d(C*Oc);
        alloctor  outputs_d(B*T *Oc);
        alloctor  bias_d(Oc);

        inputs_d.copy_from_host(inputs);
        weight_d.copy_from_host(weight);
        bias_d.copy_from_host(bias);

        using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGELU<ElementOutput, 
        1,
        ElementAccumulator,
        ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;


        using Gemm = cutlass::gemm::device::GemmBatched<
            ElementInputA, cutlass::layout::RowMajor,
            ElementInputB, cutlass::layout::RowMajor,
            ElementOutput, cutlass::layout::RowMajor,
            ElementAccumulator,cutlass::arch::OpClassSimt,cutlass::arch::Sm80,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueOutputOp
        >;

        Gemm gemm;

        cutlass::gemm::GemmCoord problem_size(T, Oc, C);

        
        auto A_m = cutlass::make_TensorRef(inputs_d.get(),cutlass::layout::RowMajor::packed({T,C}));
        auto B_m = cutlass::make_TensorRef(weight_d.get(),cutlass::layout::RowMajor::packed({C,Oc}));
        auto C_m = cutlass::make_TensorRef(bias_d.get(), cutlass::layout::RowMajor());
        auto D_m = cutlass::make_TensorRef(outputs_d.get(), cutlass::layout::RowMajor::packed({T,Oc}));

        typename Gemm::Arguments argument{
            problem_size,
            A_m,
            T*C,
            B_m,
            0,
            C_m,
            0,
            D_m,
            T*Oc,
            {1},
            B
        };

        gemm(argument);

        outputs_d.copy_to_host(outputs);
    }
    
}

void gpt2cuda::BatchMatmulGeluBackward(float * d_inputs, float* d_weight, float* d_bias,float const * d_outputs, float const *  inputs , float const * weight,int B, int T,int  C,int Oc){



}

// N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
void gpt2cuda::BatchMatmulNTForward(DevVecf& outputs,const DevVecf& inputs , const DevVecf& weight, const DevVecf& bias , int B, int T,int  C,int Oc,cudaStream_t stream){

    if(bias.size() > 0){
        BatchMatmulForward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs.data(), inputs.data(), weight.data(), bias.data(), B, T, C, Oc,stream);
    }else{
        BatchMatmulForward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs.data(), inputs.data(), weight.data(), nullptr, B, T, C, Oc,stream);
    }

}
// N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
void gpt2cuda::BatchMatmulNTBackward(DevVecf& d_inputs,  DevVecf& d_weight, DevVecf& d_bias,const DevVecf& d_outputs, const DevVecf& inputs , const DevVecf& weight,int B, int T,int  C,int Oc,cudaStream_t stream){
    if(d_bias.size() > 0){
        BatchMatmulBackward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(d_inputs.data(),d_weight.data(),d_bias.data(),d_outputs.data(),inputs.data(),weight.data(),B,T,C,Oc,stream);
    }else{
        BatchMatmulBackward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(d_inputs.data(),d_weight.data(),nullptr      ,d_outputs.data(),inputs.data(),weight.data(),B,T,C,Oc,stream);
    }
}

