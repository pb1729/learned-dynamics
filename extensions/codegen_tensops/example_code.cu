#define WARPSZ 32


template<int inds_dim>
__global__
void example_tenslinear_kern( // <<<(batch), (WARPSZ, 8)>>>
    const float* W,
    const float* x,
    float* __restrict__ y,
    int batch, int dim_in, int rank
) {
    for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {
        for (int idx_out = threadIdx.y; idx_out < rank; idx_out += blockDim.y) {
            float accum[inds_dim];
            for (int i = 0; i < inds_dim; i++) {
                accum[i] = 0.0;
            }
            // matmul
            for (int idx_in = threadIdx.x; idx_in < dim_in; idx_in += blockDim.x) {
                float W_oi = W[idx_out*dim_in + idx_in];
                for (int i = 0; i < inds_dim; i++) {
                    accum[i] += W_oi*x[((idx_batch)*dim_in + idx_in)*inds_dim + i];
                }
            }
            // reduce across threads
            // warp-level primitives correspond to blockDim.x
            for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
                for (int i = 0; i < inds_dim; i++) {
                    accum[i] += __shfl_down_sync(0xffffffff, accum[i], offset);
                }
            }
            // write to output
            if (threadIdx.x == 0) {
                for (int i = 0; i < inds_dim; i++) {
                    y[((idx_batch)*rank + idx_out)*inds_dim + i] = accum[i];
                }
            }
        }
    }
}
