

#define WARPSZ 32


template<int inds_dim>
__global__ void tensorLinearKern( // <<<(gridDim.x, gridDim.y), (WARPSZ)>>>
    const float* W,
    const float* x,
    float* __restrict__ out,
    int batch, int dim_in, int dim_out,
    int stride_W_0, int stride_W_1
) {
    for (int idx_batch = blockIdx.y; idx_batch < batch; idx_batch += gridDim.y) {
        for (int idx_out = blockIdx.x; idx_out < dim_out; idx_out += gridDim.x) {
            // initialize accumulator
            float accum[inds_dim];
            for (int i = 0; i < inds_dim; i++) {
                accum[i] = 0.0;
            }
            // matmul
            for (int idx_in = threadIdx.x; idx_in < dim_in; idx_in += blockDim.x) {
                float W_oi = W[idx_out*stride_W_0 + idx_in*stride_W_1];
                for (int i = 0; i < inds_dim; i++) {
                    accum[i] += W_oi*x[(idx_batch*dim_in + idx_in)*inds_dim + i];
                }
            }
            // reduce across threads
            for (int offset = WARPSZ/2; offset >= 1; offset /=2) {
                for (int i = 0; i < inds_dim; i++) {
                    accum[i] += __shfl_down_sync(0xffffffff, accum[i], offset);
                }
            }
            // write to output
            if (threadIdx.x == 0) {
                for (int i = 0; i < inds_dim; i++) {
                    out[(idx_batch*dim_out + idx_out)*inds_dim + i] = accum[i];
                }
            }
        }
    }
}


template<int inds_dim>
__global__ void tensorLinearBackwardKern( // <<<(gridDim.x, gridDim.y), (WARPSZ)>>>
    const float* x,
    const float* dout,
    float* __restrict__ dW,
    int batch, int dim_in, int dim_out
) {
    for (int idx_out = blockIdx.y; idx_out < dim_out; idx_out += gridDim.y) {
        for (int idx_in = blockIdx.x; idx_in < dim_in; idx_in += gridDim.x) {
            float accum = 0.0;
            // backwards
            for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
                for (int i = 0; i < inds_dim; i++) {
                    accum += dout[(idx_batch*dim_out + idx_out)*inds_dim + i]*x[(idx_batch*dim_in + idx_in)*inds_dim + i];
                }
            }
            // reduce across threads
            for (int offset = WARPSZ/2; offset >= 1; offset /=2) {
                accum += __shfl_down_sync(0xffffffff, accum, offset);
            }
            // write to output
            if (threadIdx.x == 0) {
                dW[idx_out*dim_in + idx_in] = accum;
            }
        }
    }
}



template<int inds_dim>
void tensorLinear(
    const float* W,
    const float* x,
    float* out,
    int batch, int dim_in, int dim_out,
    int stride_W_0, int stride_W_1
) {
    tensorLinearKern<inds_dim><<<dim3(dim_out, batch), WARPSZ>>>(
        W, x, out,
        batch, dim_in, dim_out,
        stride_W_0, stride_W_1);
}

template<int inds_dim>
void tensorLinearBackward(
    const float* x,
    const float* dout,
    float* dW,
    int batch, int dim_in, int dim_out
) {
    tensorLinearBackwardKern<inds_dim><<<dim3(dim_in, dim_out), WARPSZ>>>(
        x, dout, dW,
        batch, dim_in, dim_out);
}


// Compiler wants us to tell it in advance which ones we're going to need.
template void tensorLinear<1>(const float*, const float*, float*, int, int, int, int, int);
template void tensorLinear<3>(const float*, const float*, float*, int, int, int, int, int);
template void tensorLinear<9>(const float*, const float*, float*, int, int, int, int, int);
template void tensorLinearBackward<1>(const float*, const float*, float*, int, int, int);
template void tensorLinearBackward<3>(const float*, const float*, float*, int, int, int);
template void tensorLinearBackward<9>(const float*, const float*, float*, int, int, int);
