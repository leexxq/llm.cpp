#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "matmul.cuh"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "utils.cuh"
    

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

        gemm(argument,stream);
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

        gemm(argument,stream);
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

        auto status = gemm(argument,stream);
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
            auto status = gemm(argument);

            if(status != cutlass::Status::kSuccess){
                std::cerr << cutlass::cutlassGetStatusString(status) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

    }

    if(d_bias_d != nullptr){
        gpt2cuda::Reduce1D<>(d_bias_d, d_outputs_d, B*T*Oc, Oc);
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

    BatchMatmulForward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs.data(), inputs.data(), weight.data(), bias.data(), B, T, C, Oc,stream);

}
// N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
void gpt2cuda::BatchMatmulNTBackward(DevVecf& d_inputs, const DevVecf& d_weight, const DevVecf& d_bias,const DevVecf& d_outputs, const DevVecf& inputs , const DevVecf& weight,int B, int T,int  C,int Oc,cudaStream_t stream){
    
    BatchMatmulBackward<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(d_inputs.data(),d_weight.data(),d_bias.data(),d_outputs.data(),inputs.data(),weight.data(),B,T,C,Oc,stream);
}

