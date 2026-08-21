#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "gemm_with_softmax.h"
#include "gemm_with_epilogue_visitor.h"
namespace gpt2cuda {
namespace kernel{
    template<class LayoutInput = cutlass::layout::RowMajor,class LayoutWeight = cutlass::layout::RowMajor>
    struct MatmulSoftmaxConfig{
        using ElementA = float;
        using ElementB = float;
        using ElementC = float;
        using ElementCompute = float;
        using ElementD = ElementC;
        using ElementSoftmax = ElementC;

        using LayoutA = LayoutInput;
        using LayoutB = LayoutWeight;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
        using WarpShape        = cutlass::gemm::GemmShape<64, 64, 16>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using ArchTag = cutlass::arch::Sm80;

        // ApplyShape impacts the final Softmax performance a lot.
        // Set ApplyShape::kColumn to be the next multiple of 32 number that is after
        // (gemm_N / alignment).
        // Set ApplyShape::kRow to max(1, 128 / ApplyShape::kColumn).
        using ApplyShape = cutlass::MatrixShape<1, 1024>;

        static constexpr  int const kStages = 3;

        /// Linear scaling operator
        using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementCompute,
            ElementCompute
        >;

        static constexpr int AlignmentA = 1;
        static constexpr int AlignmentB = 1;
        static constexpr int AlignmentSoftmax = 1;


        using GemmSoftmax = cutlass::GemmSoftmax<
            ElementA, LayoutA,
            ElementB, LayoutB,
            ElementC,
            ElementCompute,
            OperatorClass,
            ArchTag,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            EpilogueFunctorOp,
            kStages,
            ApplyShape,
            AlignmentA,
            AlignmentB,
            AlignmentSoftmax
        >;


        using ElementNorm = typename GemmSoftmax::ElementNorm;
        using ElementSum = typename GemmSoftmax::ElementSum;
        using LayoutC = typename GemmSoftmax::LayoutC;
        using LayoutN = typename GemmSoftmax::LayoutN;
        using LayoutS = typename GemmSoftmax::LayoutS;
        using MatrixCoord = typename LayoutC::TensorCoord;
    };

}

}