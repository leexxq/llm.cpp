#include "kernel/gemm_fusion/gemm_softmax/config.cuh"
#include "matmul_softmax.cuh"

namespace gpt2cuda{
namespace kernel{

    template < class LayoutInput = cutlass::layout::RowMajor , class LayoutWeight = cutlass::layout::RowMajor> 
    void MatmulSoftmaxCUDA(float * outputs_d, float * softmax_norms ,float * softmax_sums , float const *  inputs_d , float const * weight_d, float const * bias_d , int B, int T,int  C,int Vp,int V, cudaStream_t stream = 0){
        using Config = MatmulSoftmaxConfig<LayoutInput,LayoutWeight>;
        cutlass::gemm::GemmCoord problem{T,V,C};
        int batch_count = B;


        int64_t lda;
        int64_t ldb;
        int64_t ldc = 0;
        int64_t ldd = Vp;

        if constexpr ( cutlass::platform::is_same_v<LayoutInput,cutlass::layout::RowMajor>){
            lda = C;
        }else{
            lda = T;
        }
        if constexpr( cutlass::platform::is_same_v<LayoutWeight,cutlass::layout::RowMajor>){
            ldb= Vp;
        }else{
            ldb = C;
        }

        // fixed rowmajor for norm and sum
        int64_t ldn = T; 
        int64_t lds = ldn;

        int64_t total_elements_A_per_batch = T * C;
        int64_t total_elements_B_per_batch = 0;
        int64_t total_elements_C_per_batch = 0;
        int64_t total_elements_D_per_batch = T*Vp;
        int64_t total_elements_partial_norm_per_batch = GetMatmulSoftmaxWorkSpaceSize(B, T, V) / B /2;

        int64_t total_elements_A = total_elements_A_per_batch * batch_count;
        int64_t total_elements_B = total_elements_B_per_batch * batch_count;
        int64_t total_elements_C = total_elements_C_per_batch * batch_count;
        int64_t total_elements_D = total_elements_D_per_batch * batch_count;
        int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * batch_count;
    
        cutlass::Status status = cutlass::Status::kSuccess;

        //
        // Setup arguments
        //
        
        

        typename Config::GemmSoftmax::Arguments args(
            problem,
            batch_count,
            {const_cast<float*>(inputs_d), lda},
            {const_cast<float*>(weight_d), ldb},
            {const_cast<float*>(bias_d), ldc},
            {outputs_d, ldd},
            {
                typename  Config::ElementCompute(1.f),
                typename Config::ElementCompute(1.f)
            },
            {softmax_norms, ldn},
            {softmax_sums, lds},
            {outputs_d, ldd},
            total_elements_A_per_batch,
            total_elements_B_per_batch,
            total_elements_C_per_batch,
            total_elements_D_per_batch,
            total_elements_partial_norm_per_batch,
            total_elements_partial_norm_per_batch,
            total_elements_D_per_batch
        );

        //
        // Launch
        //

        typename Config::GemmSoftmax gemm_softmax;

        // Initialize
        status = gemm_softmax.initialize(args);
        if(status != cutlass::Status::kSuccess){
            std::cerr << cutlass::cutlassGetStatusString(status) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Run
        status = gemm_softmax(stream);
        if(status != cutlass::Status::kSuccess){
            std::cerr << cutlass::cutlassGetStatusString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

}

    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTSoftmaxForward(DevVecf& outputs,DevVecf& workspace, const DevVecf& inputs , const DevVecf& weight, const DevVecf& bias , int B, int T,int C , int  Vp,int V,cudaStream_t stream){
        auto workspace_size = GetMatmulSoftmaxWorkSpaceSize(B,T,V);
        float * softmax_norms = workspace.data();
        float * softmax_sums = workspace.data() + workspace_size / 2;
        if(bias.size() > 0){
            kernel::MatmulSoftmaxCUDA<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs.data(), softmax_norms,softmax_sums,inputs.data(), weight.data(), bias.data(), B, T, C,Vp,V,stream);
        }else{
            kernel::MatmulSoftmaxCUDA<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs.data(), softmax_norms,softmax_sums,inputs.data(), weight.data(), bias.data(), B, T, C,Vp,V,stream);
        }
    }
    // N is row-major ,T is column-major, Return is row-major , C is considered intermediate dimension of matrix multiplication
    void BatchMatmulNTSoftmaxForward(float * outputs,float const *  inputs , float const * weight, float const * bias , int B, int T, int C , int  Vp,int V){
        using alloctor = cutlass::device_memory::allocation<float>;
        alloctor  inputs_d(B*T *C);
        alloctor  weight_d(C*Vp);
        alloctor  outputs_d(B*T *Vp);
        alloctor  bias_d;
        auto workspace_size = GetMatmulSoftmaxWorkSpaceSize(B,T,V);
        alloctor  softmax_norms_d(workspace_size/2);
        alloctor  softmax_sums_d(workspace_size/2);
        inputs_d.copy_from_host(inputs);
        weight_d.copy_from_host(weight);


        if(bias != nullptr){
            bias_d = alloctor(V);
            bias_d.copy_from_host(bias);
            kernel::MatmulSoftmaxCUDA<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs_d.get(),softmax_norms_d.get(),softmax_sums_d.get() ,inputs_d.get(), weight_d.get(), bias_d.get(), B, T, C, Vp,V);
        }else{
            kernel::MatmulSoftmaxCUDA<cutlass::layout::RowMajor,cutlass::layout::ColumnMajor>(outputs_d.get(),softmax_norms_d.get(),softmax_sums_d.get() ,inputs_d.get(), weight_d.get(), bias_d.get(), B, T, C, Vp,V);
        }
        outputs_d.copy_to_host(outputs);

    }
    std::size_t GetMatmulSoftmaxWorkSpaceSize(int B,int T, int V){
        using Config = kernel::MatmulSoftmaxConfig<>;
        int block_num = (V+ Config::GemmSoftmax::ThreadblockShape::kN - 1) / Config::GemmSoftmax::ThreadblockShape::kN;
        int64_t total_elements_partial_norm_per_batch = block_num * T;
        int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * B;
        return  2 * total_elements_partial_norm;
    }

}
