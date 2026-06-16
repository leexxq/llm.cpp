It's not hard to see that I used the CPU in my CUDA implememnation tests.
Howerver, we konw that Eigen's default matrix major is column-major,which presents some problems some problems when using CUTLASS. 
Therefore, we considerd using row-major in CUTLASS implementation.
This has two advantagets:
1. We can better utilize the gemm-related code in CUTLASS and perform better operator fusion.
2. Row-major make understanding CUDA operators easier.