
#define WARPSZ 32
#define MODWARP(X) (X & 0x1f)



__global__
void fused_tensor_prods_example_kern(
    // <<<(batch), (WARPSZ, dim_l)>>>
    int p_0, int p_1_base, int p_1, int p_2_base, int p_2, 
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* __restrict__ y_0, float* __restrict__ y_1, float* __restrict__ y_2) {
  extern __shared__ float s[];
  float* product_000 = &s[0*p_0]; // size = 1*p_0
  float* product_011 = &s[p_1_base + 0*p_1]; // size = 3*p_1
  float* product_101 = &s[1*p_0]; // size = 3*p_0
  float* product_110 = &s[p_1_base + 3*p_1]; // size = 1*p_1
  float* product_220 = &s[p_2_base + 0*p_2]; // size = 1*p_2
  float* product_222 = &s[p_2_base + 1*p_2]; // size = 9*p_2
  float* product_211 = &s[p_1_base + 4*p_1]; // size = 3*p_1
  float* product_111 = &s[p_1_base + 7*p_1]; // size = 3*p_1
  float* product_212 = &s[p_1_base + 10*p_1]; // size = 9*p_1
  for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {
    { // compute tensor products
      float l_000_0 = left_000[((idx_batch)*dim_l + threadIdx.y)*1 + 0];
      for (int idx_chan_in_000 = threadIdx.x; idx_chan_in_000 < dim_0; idx_chan_in_000 += blockDim.x) {
        product_000[((threadIdx.y)*dim_0 + idx_chan_in_000)*1 + 0] = l_000_0*x_0[((idx_batch)*dim_0 + idx_chan_in_000)*1 + 0];
      }
      float l_011_0 = left_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0];
      for (int idx_chan_in_011 = threadIdx.x; idx_chan_in_011 < dim_1; idx_chan_in_011 += blockDim.x) {
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 0] = l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 0];
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 1] = l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 1];
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 2] = l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 2];
      }
      float l_101_0 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_101_1 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_101_2 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_in_101 = threadIdx.x; idx_chan_in_101 < dim_0; idx_chan_in_101 += blockDim.x) {
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 0] = l_101_0*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0];
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 1] = l_101_1*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0];
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 2] = l_101_2*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0];
      }
      float l_110_0 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_110_1 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_110_2 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_in_110 = threadIdx.x; idx_chan_in_110 < dim_1; idx_chan_in_110 += blockDim.x) {
        product_110[((threadIdx.y)*dim_1 + idx_chan_in_110)*1 + 0] = l_110_0*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 0] + l_110_1*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 1] + l_110_2*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 2];
      }
      float l_220_0 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_220_1 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_220_2 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_220_3 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_220_4 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_220_5 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_220_6 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_220_7 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_220_8 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_in_220 = threadIdx.x; idx_chan_in_220 < dim_2; idx_chan_in_220 += blockDim.x) {
        product_220[((threadIdx.y)*dim_2 + idx_chan_in_220)*1 + 0] = l_220_0*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 0] + l_220_1*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 1] + l_220_2*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 2] + l_220_3*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 3] + l_220_4*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 4] + l_220_5*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 5] + l_220_6*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 6] + l_220_7*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 7] + l_220_8*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 8];
      }
      float l_222_0 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_222_1 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_222_2 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_222_3 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_222_4 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_222_5 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_222_6 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_222_7 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_222_8 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_in_222 = threadIdx.x; idx_chan_in_222 < dim_2; idx_chan_in_222 += blockDim.x) {
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 0] = l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0] + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1] + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 1] = l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3] + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4] + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 2] = l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6] + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7] + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 3] = l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0] + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1] + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 4] = l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3] + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4] + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 5] = l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6] + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7] + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 6] = l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0] + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1] + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 7] = l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3] + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4] + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5];
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 8] = l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6] + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7] + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8];
      }
      float l_211_0 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_211_1 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_211_2 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_211_3 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_211_4 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_211_5 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_211_6 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_211_7 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_211_8 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_in_211 = threadIdx.x; idx_chan_in_211 < dim_1; idx_chan_in_211 += blockDim.x) {
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 0] = l_211_0*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0] + l_211_1*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1] + l_211_2*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2];
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 1] = l_211_3*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0] + l_211_4*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1] + l_211_5*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2];
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 2] = l_211_6*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0] + l_211_7*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1] + l_211_8*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2];
      }
      float l_111_0 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_111_1 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_111_2 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_in_111 = threadIdx.x; idx_chan_in_111 < dim_1; idx_chan_in_111 += blockDim.x) {
        product_111[((threadIdx.y)*dim_1 + idx_chan_in_111)*3 + 0] = l_111_1*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2] + (-1)*l_111_2*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1];
        product_111[((threadIdx.y)*dim_1 + idx_chan_in_111)*3 + 1] = (-1)*l_111_0*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2] + l_111_2*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0];
        product_111[((threadIdx.y)*dim_1 + idx_chan_in_111)*3 + 2] = l_111_0*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1] + (-1)*l_111_1*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0];
      }
      float l_212_0 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_212_1 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_212_2 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_212_3 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_212_4 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_212_5 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_212_6 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_212_7 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_212_8 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_in_212 = threadIdx.x; idx_chan_in_212 < dim_1; idx_chan_in_212 += blockDim.x) {
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 0] = l_212_1*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + (-1)*l_212_2*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 1] = (-1)*l_212_0*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + l_212_2*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 2] = l_212_0*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1] + (-1)*l_212_1*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 3] = l_212_4*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + (-1)*l_212_5*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 4] = (-1)*l_212_3*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + l_212_5*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 5] = l_212_3*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1] + (-1)*l_212_4*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 6] = l_212_7*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + (-1)*l_212_8*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 7] = (-1)*l_212_6*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2] + l_212_8*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
        product_212[((threadIdx.y)*dim_1 + idx_chan_in_212)*9 + 8] = l_212_6*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1] + (-1)*l_212_7*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0];
      }
    }
    __syncthreads();
    { // linear transforms to compute the outputs
      for (int idx_chan_out_0 = threadIdx.y; idx_chan_out_0 < dim_0; idx_chan_out_0 += blockDim.y) {
        float y_o_0_0 = 0.0;
        float accum_000_0 = 0.0;
        for (int idx_chan_in_000 = threadIdx.x; idx_chan_in_000 < dim_l*dim_0; idx_chan_in_000 += blockDim.x) {
          float P_oi_000 = P_000[(idx_chan_out_0)*dim_l*dim_0 + idx_chan_in_000];
          accum_000_0 += P_oi_000*product_000[(idx_chan_in_000)*1 + 0];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_000_0 += __shfl_down_sync(0xffffffff, accum_000_0, offset);
        }
        if (threadIdx.x == 0) {
          y_o_0_0 += accum_000_0;
        }
        float accum_110_0 = 0.0;
        for (int idx_chan_in_110 = threadIdx.x; idx_chan_in_110 < dim_l*dim_1; idx_chan_in_110 += blockDim.x) {
          float P_oi_110 = P_110[(idx_chan_out_0)*dim_l*dim_1 + idx_chan_in_110];
          accum_110_0 += P_oi_110*product_110[(idx_chan_in_110)*1 + 0];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_110_0 += __shfl_down_sync(0xffffffff, accum_110_0, offset);
        }
        if (threadIdx.x == 0) {
          y_o_0_0 += accum_110_0;
        }
        float accum_220_0 = 0.0;
        for (int idx_chan_in_220 = threadIdx.x; idx_chan_in_220 < dim_l*dim_2; idx_chan_in_220 += blockDim.x) {
          float P_oi_220 = P_220[(idx_chan_out_0)*dim_l*dim_2 + idx_chan_in_220];
          accum_220_0 += P_oi_220*product_220[(idx_chan_in_220)*1 + 0];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_220_0 += __shfl_down_sync(0xffffffff, accum_220_0, offset);
        }
        if (threadIdx.x == 0) {
          y_o_0_0 += accum_220_0;
        }
        if (threadIdx.x == 0) {
          y_0[((idx_batch)*dim_0 + idx_chan_out_0)*1 + 0] = y_o_0_0;
        }
      }
      for (int idx_chan_out_1 = threadIdx.y; idx_chan_out_1 < dim_1; idx_chan_out_1 += blockDim.y) {
        float y_o_1_0 = 0.0;
        float y_o_1_1 = 0.0;
        float y_o_1_2 = 0.0;
        float accum_011_0 = 0.0;
        float accum_011_1 = 0.0;
        float accum_011_2 = 0.0;
        for (int idx_chan_in_011 = threadIdx.x; idx_chan_in_011 < dim_l*dim_1; idx_chan_in_011 += blockDim.x) {
          float P_oi_011 = P_011[(idx_chan_out_1)*dim_l*dim_1 + idx_chan_in_011];
          accum_011_0 += P_oi_011*product_011[(idx_chan_in_011)*3 + 0];
          accum_011_1 += P_oi_011*product_011[(idx_chan_in_011)*3 + 1];
          accum_011_2 += P_oi_011*product_011[(idx_chan_in_011)*3 + 2];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_011_0 += __shfl_down_sync(0xffffffff, accum_011_0, offset);
          accum_011_1 += __shfl_down_sync(0xffffffff, accum_011_1, offset);
          accum_011_2 += __shfl_down_sync(0xffffffff, accum_011_2, offset);
        }
        if (threadIdx.x == 0) {
          y_o_1_0 += accum_011_0;
          y_o_1_1 += accum_011_1;
          y_o_1_2 += accum_011_2;
        }
        float accum_101_0 = 0.0;
        float accum_101_1 = 0.0;
        float accum_101_2 = 0.0;
        for (int idx_chan_in_101 = threadIdx.x; idx_chan_in_101 < dim_l*dim_0; idx_chan_in_101 += blockDim.x) {
          float P_oi_101 = P_101[(idx_chan_out_1)*dim_l*dim_0 + idx_chan_in_101];
          accum_101_0 += P_oi_101*product_101[(idx_chan_in_101)*3 + 0];
          accum_101_1 += P_oi_101*product_101[(idx_chan_in_101)*3 + 1];
          accum_101_2 += P_oi_101*product_101[(idx_chan_in_101)*3 + 2];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_101_0 += __shfl_down_sync(0xffffffff, accum_101_0, offset);
          accum_101_1 += __shfl_down_sync(0xffffffff, accum_101_1, offset);
          accum_101_2 += __shfl_down_sync(0xffffffff, accum_101_2, offset);
        }
        if (threadIdx.x == 0) {
          y_o_1_0 += accum_101_0;
          y_o_1_1 += accum_101_1;
          y_o_1_2 += accum_101_2;
        }
        float accum_211_0 = 0.0;
        float accum_211_1 = 0.0;
        float accum_211_2 = 0.0;
        for (int idx_chan_in_211 = threadIdx.x; idx_chan_in_211 < dim_l*dim_1; idx_chan_in_211 += blockDim.x) {
          float P_oi_211 = P_211[(idx_chan_out_1)*dim_l*dim_1 + idx_chan_in_211];
          accum_211_0 += P_oi_211*product_211[(idx_chan_in_211)*3 + 0];
          accum_211_1 += P_oi_211*product_211[(idx_chan_in_211)*3 + 1];
          accum_211_2 += P_oi_211*product_211[(idx_chan_in_211)*3 + 2];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_211_0 += __shfl_down_sync(0xffffffff, accum_211_0, offset);
          accum_211_1 += __shfl_down_sync(0xffffffff, accum_211_1, offset);
          accum_211_2 += __shfl_down_sync(0xffffffff, accum_211_2, offset);
        }
        if (threadIdx.x == 0) {
          y_o_1_0 += accum_211_0;
          y_o_1_1 += accum_211_1;
          y_o_1_2 += accum_211_2;
        }
        float accum_111_0 = 0.0;
        float accum_111_1 = 0.0;
        float accum_111_2 = 0.0;
        for (int idx_chan_in_111 = threadIdx.x; idx_chan_in_111 < dim_l*dim_1; idx_chan_in_111 += blockDim.x) {
          float P_oi_111 = P_111[(idx_chan_out_1)*dim_l*dim_1 + idx_chan_in_111];
          accum_111_0 += P_oi_111*product_111[(idx_chan_in_111)*3 + 0];
          accum_111_1 += P_oi_111*product_111[(idx_chan_in_111)*3 + 1];
          accum_111_2 += P_oi_111*product_111[(idx_chan_in_111)*3 + 2];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_111_0 += __shfl_down_sync(0xffffffff, accum_111_0, offset);
          accum_111_1 += __shfl_down_sync(0xffffffff, accum_111_1, offset);
          accum_111_2 += __shfl_down_sync(0xffffffff, accum_111_2, offset);
        }
        if (threadIdx.x == 0) {
          y_o_1_0 += accum_111_0;
          y_o_1_1 += accum_111_1;
          y_o_1_2 += accum_111_2;
        }
        if (threadIdx.x == 0) {
          y_1[((idx_batch)*dim_1 + idx_chan_out_1)*3 + 0] = y_o_1_0;
          y_1[((idx_batch)*dim_1 + idx_chan_out_1)*3 + 1] = y_o_1_1;
          y_1[((idx_batch)*dim_1 + idx_chan_out_1)*3 + 2] = y_o_1_2;
        }
      }
      for (int idx_chan_out_2 = threadIdx.y; idx_chan_out_2 < dim_2; idx_chan_out_2 += blockDim.y) {
        float y_o_2_0 = 0.0;
        float y_o_2_1 = 0.0;
        float y_o_2_2 = 0.0;
        float y_o_2_3 = 0.0;
        float y_o_2_4 = 0.0;
        float y_o_2_5 = 0.0;
        float y_o_2_6 = 0.0;
        float y_o_2_7 = 0.0;
        float y_o_2_8 = 0.0;
        float accum_222_0 = 0.0;
        float accum_222_1 = 0.0;
        float accum_222_2 = 0.0;
        float accum_222_3 = 0.0;
        float accum_222_4 = 0.0;
        float accum_222_5 = 0.0;
        float accum_222_6 = 0.0;
        float accum_222_7 = 0.0;
        float accum_222_8 = 0.0;
        for (int idx_chan_in_222 = threadIdx.x; idx_chan_in_222 < dim_l*dim_2; idx_chan_in_222 += blockDim.x) {
          float P_oi_222 = P_222[(idx_chan_out_2)*dim_l*dim_2 + idx_chan_in_222];
          accum_222_0 += P_oi_222*product_222[(idx_chan_in_222)*9 + 0];
          accum_222_1 += P_oi_222*product_222[(idx_chan_in_222)*9 + 1];
          accum_222_2 += P_oi_222*product_222[(idx_chan_in_222)*9 + 2];
          accum_222_3 += P_oi_222*product_222[(idx_chan_in_222)*9 + 3];
          accum_222_4 += P_oi_222*product_222[(idx_chan_in_222)*9 + 4];
          accum_222_5 += P_oi_222*product_222[(idx_chan_in_222)*9 + 5];
          accum_222_6 += P_oi_222*product_222[(idx_chan_in_222)*9 + 6];
          accum_222_7 += P_oi_222*product_222[(idx_chan_in_222)*9 + 7];
          accum_222_8 += P_oi_222*product_222[(idx_chan_in_222)*9 + 8];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_222_0 += __shfl_down_sync(0xffffffff, accum_222_0, offset);
          accum_222_1 += __shfl_down_sync(0xffffffff, accum_222_1, offset);
          accum_222_2 += __shfl_down_sync(0xffffffff, accum_222_2, offset);
          accum_222_3 += __shfl_down_sync(0xffffffff, accum_222_3, offset);
          accum_222_4 += __shfl_down_sync(0xffffffff, accum_222_4, offset);
          accum_222_5 += __shfl_down_sync(0xffffffff, accum_222_5, offset);
          accum_222_6 += __shfl_down_sync(0xffffffff, accum_222_6, offset);
          accum_222_7 += __shfl_down_sync(0xffffffff, accum_222_7, offset);
          accum_222_8 += __shfl_down_sync(0xffffffff, accum_222_8, offset);
        }
        if (threadIdx.x == 0) {
          y_o_2_0 += accum_222_0;
          y_o_2_1 += accum_222_1;
          y_o_2_2 += accum_222_2;
          y_o_2_3 += accum_222_3;
          y_o_2_4 += accum_222_4;
          y_o_2_5 += accum_222_5;
          y_o_2_6 += accum_222_6;
          y_o_2_7 += accum_222_7;
          y_o_2_8 += accum_222_8;
        }
        float accum_212_0 = 0.0;
        float accum_212_1 = 0.0;
        float accum_212_2 = 0.0;
        float accum_212_3 = 0.0;
        float accum_212_4 = 0.0;
        float accum_212_5 = 0.0;
        float accum_212_6 = 0.0;
        float accum_212_7 = 0.0;
        float accum_212_8 = 0.0;
        for (int idx_chan_in_212 = threadIdx.x; idx_chan_in_212 < dim_l*dim_1; idx_chan_in_212 += blockDim.x) {
          float P_oi_212 = P_212[(idx_chan_out_2)*dim_l*dim_1 + idx_chan_in_212];
          accum_212_0 += P_oi_212*product_212[(idx_chan_in_212)*9 + 0];
          accum_212_1 += P_oi_212*product_212[(idx_chan_in_212)*9 + 1];
          accum_212_2 += P_oi_212*product_212[(idx_chan_in_212)*9 + 2];
          accum_212_3 += P_oi_212*product_212[(idx_chan_in_212)*9 + 3];
          accum_212_4 += P_oi_212*product_212[(idx_chan_in_212)*9 + 4];
          accum_212_5 += P_oi_212*product_212[(idx_chan_in_212)*9 + 5];
          accum_212_6 += P_oi_212*product_212[(idx_chan_in_212)*9 + 6];
          accum_212_7 += P_oi_212*product_212[(idx_chan_in_212)*9 + 7];
          accum_212_8 += P_oi_212*product_212[(idx_chan_in_212)*9 + 8];
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_212_0 += __shfl_down_sync(0xffffffff, accum_212_0, offset);
          accum_212_1 += __shfl_down_sync(0xffffffff, accum_212_1, offset);
          accum_212_2 += __shfl_down_sync(0xffffffff, accum_212_2, offset);
          accum_212_3 += __shfl_down_sync(0xffffffff, accum_212_3, offset);
          accum_212_4 += __shfl_down_sync(0xffffffff, accum_212_4, offset);
          accum_212_5 += __shfl_down_sync(0xffffffff, accum_212_5, offset);
          accum_212_6 += __shfl_down_sync(0xffffffff, accum_212_6, offset);
          accum_212_7 += __shfl_down_sync(0xffffffff, accum_212_7, offset);
          accum_212_8 += __shfl_down_sync(0xffffffff, accum_212_8, offset);
        }
        if (threadIdx.x == 0) {
          y_o_2_0 += accum_212_0;
          y_o_2_1 += accum_212_1;
          y_o_2_2 += accum_212_2;
          y_o_2_3 += accum_212_3;
          y_o_2_4 += accum_212_4;
          y_o_2_5 += accum_212_5;
          y_o_2_6 += accum_212_6;
          y_o_2_7 += accum_212_7;
          y_o_2_8 += accum_212_8;
        }
        if (threadIdx.x == 0) {
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 0] = y_o_2_0;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 1] = y_o_2_1;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 2] = y_o_2_2;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 3] = y_o_2_3;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 4] = y_o_2_4;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 5] = y_o_2_5;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 6] = y_o_2_6;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 7] = y_o_2_7;
          y_2[((idx_batch)*dim_2 + idx_chan_out_2)*9 + 8] = y_o_2_8;
        }
      }
    }
  }
}


void fused_tensor_prods_example(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* y_0, float* y_1, float* y_2) {
  
  int p_0 = dim_l*dim_0;
  int p_1 = dim_l*dim_1;
  int p_2 = dim_l*dim_2;
  int sharedmemsz = 0;
  sharedmemsz += 4*p_0;
  int p_1_base = sharedmemsz;
  sharedmemsz += 19*p_1;
  int p_2_base = sharedmemsz;
  sharedmemsz += 10*p_2;
  dim3 gridsz = dim3(batch);
  dim3 blocksz = dim3(WARPSZ, dim_l);
  fused_tensor_prods_example_kern<<<gridsz, blocksz, sharedmemsz*sizeof(float)>>>(
      p_0, p_1_base, p_1, p_2_base, p_2, 
      batch, dim_l, dim_0, dim_1, dim_2,
      x_0, x_1, x_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212,
      y_0, y_1, y_2);
  
}


__global__
void fused_tensor_prods_example_backward_kern(
    // <<<(batch), (WARPSZ, dim_l)>>>
    int p_0, int p_1_base, int p_1, int p_2_base, int p_2, 
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* __restrict__ dx_0, float* __restrict__ dx_1, float* __restrict__ dx_2) {
  extern __shared__ float s[];
  float* dproduct_000 = &s[0*p_0]; // size = 1*p_0
  float* dproduct_011 = &s[p_1_base + 0*p_1]; // size = 3*p_1
  float* dproduct_101 = &s[p_1_base + 3*p_1]; // size = 1*p_1
  float* dproduct_110 = &s[1*p_0]; // size = 3*p_0
  float* dproduct_220 = &s[4*p_0]; // size = 9*p_0
  float* dproduct_222 = &s[p_2_base + 0*p_2]; // size = 9*p_2
  float* dproduct_211 = &s[p_1_base + 4*p_1]; // size = 3*p_1
  float* dproduct_111 = &s[p_1_base + 7*p_1]; // size = 3*p_1
  float* dproduct_212 = &s[p_2_base + 9*p_2]; // size = 3*p_2
  for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {
    { // compute tensor products
      float l_000_0 = left_000[((idx_batch)*dim_l + threadIdx.y)*1 + 0];
      for (int idx_chan_out_000 = threadIdx.x; idx_chan_out_000 < dim_0; idx_chan_out_000 += blockDim.x) {
        dproduct_000[((threadIdx.y)*dim_0 + idx_chan_out_000)*1 + 0] = l_000_0*dy_0[((idx_batch)*dim_0 + idx_chan_out_000)*1 + 0];
      }
      float l_011_0 = left_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0];
      for (int idx_chan_out_011 = threadIdx.x; idx_chan_out_011 < dim_1; idx_chan_out_011 += blockDim.x) {
        dproduct_011[((threadIdx.y)*dim_1 + idx_chan_out_011)*3 + 0] = l_011_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 0];
        dproduct_011[((threadIdx.y)*dim_1 + idx_chan_out_011)*3 + 1] = l_011_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 1];
        dproduct_011[((threadIdx.y)*dim_1 + idx_chan_out_011)*3 + 2] = l_011_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 2];
      }
      float l_101_0 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_101_1 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_101_2 = left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_out_101 = threadIdx.x; idx_chan_out_101 < dim_1; idx_chan_out_101 += blockDim.x) {
        dproduct_101[((threadIdx.y)*dim_1 + idx_chan_out_101)*1 + 0] = l_101_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 0] + l_101_1*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 1] + l_101_2*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 2];
      }
      float l_110_0 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_110_1 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_110_2 = left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_out_110 = threadIdx.x; idx_chan_out_110 < dim_0; idx_chan_out_110 += blockDim.x) {
        dproduct_110[((threadIdx.y)*dim_0 + idx_chan_out_110)*3 + 0] = l_110_0*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
        dproduct_110[((threadIdx.y)*dim_0 + idx_chan_out_110)*3 + 1] = l_110_1*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
        dproduct_110[((threadIdx.y)*dim_0 + idx_chan_out_110)*3 + 2] = l_110_2*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
      }
      float l_220_0 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_220_1 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_220_2 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_220_3 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_220_4 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_220_5 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_220_6 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_220_7 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_220_8 = left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_out_220 = threadIdx.x; idx_chan_out_220 < dim_0; idx_chan_out_220 += blockDim.x) {
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 0] = l_220_0*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 1] = l_220_1*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 2] = l_220_2*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 3] = l_220_3*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 4] = l_220_4*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 5] = l_220_5*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 6] = l_220_6*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 7] = l_220_7*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
        dproduct_220[((threadIdx.y)*dim_0 + idx_chan_out_220)*9 + 8] = l_220_8*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
      }
      float l_222_0 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_222_1 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_222_2 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_222_3 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_222_4 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_222_5 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_222_6 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_222_7 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_222_8 = left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_out_222 = threadIdx.x; idx_chan_out_222 < dim_2; idx_chan_out_222 += blockDim.x) {
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 0] = l_222_0*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + l_222_3*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + l_222_6*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 1] = l_222_1*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + l_222_4*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + l_222_7*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 2] = l_222_2*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + l_222_5*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + l_222_8*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 3] = l_222_0*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + l_222_3*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + l_222_6*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 4] = l_222_1*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + l_222_4*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + l_222_7*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 5] = l_222_2*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + l_222_5*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + l_222_8*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 6] = l_222_0*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + l_222_3*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + l_222_6*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 7] = l_222_1*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + l_222_4*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + l_222_7*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
        dproduct_222[((threadIdx.y)*dim_2 + idx_chan_out_222)*9 + 8] = l_222_2*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + l_222_5*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + l_222_8*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
      }
      float l_211_0 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_211_1 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_211_2 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_211_3 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_211_4 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_211_5 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_211_6 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_211_7 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_211_8 = left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_out_211 = threadIdx.x; idx_chan_out_211 < dim_1; idx_chan_out_211 += blockDim.x) {
        dproduct_211[((threadIdx.y)*dim_1 + idx_chan_out_211)*3 + 0] = l_211_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + l_211_3*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + l_211_6*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
        dproduct_211[((threadIdx.y)*dim_1 + idx_chan_out_211)*3 + 1] = l_211_1*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + l_211_4*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + l_211_7*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
        dproduct_211[((threadIdx.y)*dim_1 + idx_chan_out_211)*3 + 2] = l_211_2*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + l_211_5*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + l_211_8*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
      }
      float l_111_0 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 0];
      float l_111_1 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 1];
      float l_111_2 = left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 2];
      for (int idx_chan_out_111 = threadIdx.x; idx_chan_out_111 < dim_1; idx_chan_out_111 += blockDim.x) {
        dproduct_111[((threadIdx.y)*dim_1 + idx_chan_out_111)*3 + 0] = l_111_2*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1] + (-1)*l_111_1*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2];
        dproduct_111[((threadIdx.y)*dim_1 + idx_chan_out_111)*3 + 1] = (-1)*l_111_2*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + l_111_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2];
        dproduct_111[((threadIdx.y)*dim_1 + idx_chan_out_111)*3 + 2] = l_111_1*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + (-1)*l_111_0*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1];
      }
      float l_212_0 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 0];
      float l_212_1 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 1];
      float l_212_2 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 2];
      float l_212_3 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 3];
      float l_212_4 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 4];
      float l_212_5 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 5];
      float l_212_6 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 6];
      float l_212_7 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 7];
      float l_212_8 = left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 8];
      for (int idx_chan_out_212 = threadIdx.x; idx_chan_out_212 < dim_2; idx_chan_out_212 += blockDim.x) {
        dproduct_212[((threadIdx.y)*dim_2 + idx_chan_out_212)*3 + 0] = l_212_2*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1] + (-1)*l_212_1*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2] + l_212_5*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4] + (-1)*l_212_4*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5] + l_212_8*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7] + (-1)*l_212_7*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8];
        dproduct_212[((threadIdx.y)*dim_2 + idx_chan_out_212)*3 + 1] = (-1)*l_212_2*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + l_212_0*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2] + (-1)*l_212_5*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + l_212_3*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5] + (-1)*l_212_8*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + l_212_6*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8];
        dproduct_212[((threadIdx.y)*dim_2 + idx_chan_out_212)*3 + 2] = l_212_1*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + (-1)*l_212_0*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1] + l_212_4*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + (-1)*l_212_3*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4] + l_212_7*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + (-1)*l_212_6*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7];
      }
    }
    __syncthreads();
    { // linear transforms to compute dx
      for (int idx_chan_in_0 = threadIdx.y; idx_chan_in_0 < dim_0; idx_chan_in_0 += blockDim.y) {
        float dx_o_0_0 = 0.0;
        float accum_000_0 = 0.0;
        for (int idx_l_000 = 0; idx_l_000 < dim_l; idx_l_000 += 1) {
          for (int idx_chan_out_000 = threadIdx.x; idx_chan_out_000 < dim_0; idx_chan_out_000 += blockDim.x) {
            float P_oi_000 = P_000[((idx_chan_out_000)*dim_l + idx_l_000)*dim_0 + idx_chan_in_0];
            accum_000_0 += P_oi_000*dproduct_000[((idx_l_000)*dim_0 + idx_chan_out_000)*1 + 0];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_000_0 += __shfl_down_sync(0xffffffff, accum_000_0, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_0_0 += accum_000_0;
        }
        float accum_101_0 = 0.0;
        for (int idx_l_101 = 0; idx_l_101 < dim_l; idx_l_101 += 1) {
          for (int idx_chan_out_101 = threadIdx.x; idx_chan_out_101 < dim_1; idx_chan_out_101 += blockDim.x) {
            float P_oi_101 = P_101[((idx_chan_out_101)*dim_l + idx_l_101)*dim_0 + idx_chan_in_0];
            accum_101_0 += P_oi_101*dproduct_101[((idx_l_101)*dim_1 + idx_chan_out_101)*1 + 0];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_101_0 += __shfl_down_sync(0xffffffff, accum_101_0, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_0_0 += accum_101_0;
        }
        if (threadIdx.x == 0) {
          dx_0[((idx_batch)*dim_0 + idx_chan_in_0)*1 + 0] = dx_o_0_0;
        }
      }
      for (int idx_chan_in_1 = threadIdx.y; idx_chan_in_1 < dim_1; idx_chan_in_1 += blockDim.y) {
        float dx_o_1_0 = 0.0;
        float dx_o_1_1 = 0.0;
        float dx_o_1_2 = 0.0;
        float accum_011_0 = 0.0;
        float accum_011_1 = 0.0;
        float accum_011_2 = 0.0;
        for (int idx_l_011 = 0; idx_l_011 < dim_l; idx_l_011 += 1) {
          for (int idx_chan_out_011 = threadIdx.x; idx_chan_out_011 < dim_1; idx_chan_out_011 += blockDim.x) {
            float P_oi_011 = P_011[((idx_chan_out_011)*dim_l + idx_l_011)*dim_1 + idx_chan_in_1];
            accum_011_0 += P_oi_011*dproduct_011[((idx_l_011)*dim_1 + idx_chan_out_011)*3 + 0];
            accum_011_1 += P_oi_011*dproduct_011[((idx_l_011)*dim_1 + idx_chan_out_011)*3 + 1];
            accum_011_2 += P_oi_011*dproduct_011[((idx_l_011)*dim_1 + idx_chan_out_011)*3 + 2];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_011_0 += __shfl_down_sync(0xffffffff, accum_011_0, offset);
          accum_011_1 += __shfl_down_sync(0xffffffff, accum_011_1, offset);
          accum_011_2 += __shfl_down_sync(0xffffffff, accum_011_2, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_1_0 += accum_011_0;
          dx_o_1_1 += accum_011_1;
          dx_o_1_2 += accum_011_2;
        }
        float accum_110_0 = 0.0;
        float accum_110_1 = 0.0;
        float accum_110_2 = 0.0;
        for (int idx_l_110 = 0; idx_l_110 < dim_l; idx_l_110 += 1) {
          for (int idx_chan_out_110 = threadIdx.x; idx_chan_out_110 < dim_0; idx_chan_out_110 += blockDim.x) {
            float P_oi_110 = P_110[((idx_chan_out_110)*dim_l + idx_l_110)*dim_1 + idx_chan_in_1];
            accum_110_0 += P_oi_110*dproduct_110[((idx_l_110)*dim_0 + idx_chan_out_110)*3 + 0];
            accum_110_1 += P_oi_110*dproduct_110[((idx_l_110)*dim_0 + idx_chan_out_110)*3 + 1];
            accum_110_2 += P_oi_110*dproduct_110[((idx_l_110)*dim_0 + idx_chan_out_110)*3 + 2];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_110_0 += __shfl_down_sync(0xffffffff, accum_110_0, offset);
          accum_110_1 += __shfl_down_sync(0xffffffff, accum_110_1, offset);
          accum_110_2 += __shfl_down_sync(0xffffffff, accum_110_2, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_1_0 += accum_110_0;
          dx_o_1_1 += accum_110_1;
          dx_o_1_2 += accum_110_2;
        }
        float accum_211_0 = 0.0;
        float accum_211_1 = 0.0;
        float accum_211_2 = 0.0;
        for (int idx_l_211 = 0; idx_l_211 < dim_l; idx_l_211 += 1) {
          for (int idx_chan_out_211 = threadIdx.x; idx_chan_out_211 < dim_1; idx_chan_out_211 += blockDim.x) {
            float P_oi_211 = P_211[((idx_chan_out_211)*dim_l + idx_l_211)*dim_1 + idx_chan_in_1];
            accum_211_0 += P_oi_211*dproduct_211[((idx_l_211)*dim_1 + idx_chan_out_211)*3 + 0];
            accum_211_1 += P_oi_211*dproduct_211[((idx_l_211)*dim_1 + idx_chan_out_211)*3 + 1];
            accum_211_2 += P_oi_211*dproduct_211[((idx_l_211)*dim_1 + idx_chan_out_211)*3 + 2];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_211_0 += __shfl_down_sync(0xffffffff, accum_211_0, offset);
          accum_211_1 += __shfl_down_sync(0xffffffff, accum_211_1, offset);
          accum_211_2 += __shfl_down_sync(0xffffffff, accum_211_2, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_1_0 += accum_211_0;
          dx_o_1_1 += accum_211_1;
          dx_o_1_2 += accum_211_2;
        }
        float accum_111_0 = 0.0;
        float accum_111_1 = 0.0;
        float accum_111_2 = 0.0;
        for (int idx_l_111 = 0; idx_l_111 < dim_l; idx_l_111 += 1) {
          for (int idx_chan_out_111 = threadIdx.x; idx_chan_out_111 < dim_1; idx_chan_out_111 += blockDim.x) {
            float P_oi_111 = P_111[((idx_chan_out_111)*dim_l + idx_l_111)*dim_1 + idx_chan_in_1];
            accum_111_0 += P_oi_111*dproduct_111[((idx_l_111)*dim_1 + idx_chan_out_111)*3 + 0];
            accum_111_1 += P_oi_111*dproduct_111[((idx_l_111)*dim_1 + idx_chan_out_111)*3 + 1];
            accum_111_2 += P_oi_111*dproduct_111[((idx_l_111)*dim_1 + idx_chan_out_111)*3 + 2];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_111_0 += __shfl_down_sync(0xffffffff, accum_111_0, offset);
          accum_111_1 += __shfl_down_sync(0xffffffff, accum_111_1, offset);
          accum_111_2 += __shfl_down_sync(0xffffffff, accum_111_2, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_1_0 += accum_111_0;
          dx_o_1_1 += accum_111_1;
          dx_o_1_2 += accum_111_2;
        }
        float accum_212_0 = 0.0;
        float accum_212_1 = 0.0;
        float accum_212_2 = 0.0;
        for (int idx_l_212 = 0; idx_l_212 < dim_l; idx_l_212 += 1) {
          for (int idx_chan_out_212 = threadIdx.x; idx_chan_out_212 < dim_2; idx_chan_out_212 += blockDim.x) {
            float P_oi_212 = P_212[((idx_chan_out_212)*dim_l + idx_l_212)*dim_1 + idx_chan_in_1];
            accum_212_0 += P_oi_212*dproduct_212[((idx_l_212)*dim_2 + idx_chan_out_212)*3 + 0];
            accum_212_1 += P_oi_212*dproduct_212[((idx_l_212)*dim_2 + idx_chan_out_212)*3 + 1];
            accum_212_2 += P_oi_212*dproduct_212[((idx_l_212)*dim_2 + idx_chan_out_212)*3 + 2];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_212_0 += __shfl_down_sync(0xffffffff, accum_212_0, offset);
          accum_212_1 += __shfl_down_sync(0xffffffff, accum_212_1, offset);
          accum_212_2 += __shfl_down_sync(0xffffffff, accum_212_2, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_1_0 += accum_212_0;
          dx_o_1_1 += accum_212_1;
          dx_o_1_2 += accum_212_2;
        }
        if (threadIdx.x == 0) {
          dx_1[((idx_batch)*dim_1 + idx_chan_in_1)*3 + 0] = dx_o_1_0;
          dx_1[((idx_batch)*dim_1 + idx_chan_in_1)*3 + 1] = dx_o_1_1;
          dx_1[((idx_batch)*dim_1 + idx_chan_in_1)*3 + 2] = dx_o_1_2;
        }
      }
      for (int idx_chan_in_2 = threadIdx.y; idx_chan_in_2 < dim_2; idx_chan_in_2 += blockDim.y) {
        float dx_o_2_0 = 0.0;
        float dx_o_2_1 = 0.0;
        float dx_o_2_2 = 0.0;
        float dx_o_2_3 = 0.0;
        float dx_o_2_4 = 0.0;
        float dx_o_2_5 = 0.0;
        float dx_o_2_6 = 0.0;
        float dx_o_2_7 = 0.0;
        float dx_o_2_8 = 0.0;
        float accum_220_0 = 0.0;
        float accum_220_1 = 0.0;
        float accum_220_2 = 0.0;
        float accum_220_3 = 0.0;
        float accum_220_4 = 0.0;
        float accum_220_5 = 0.0;
        float accum_220_6 = 0.0;
        float accum_220_7 = 0.0;
        float accum_220_8 = 0.0;
        for (int idx_l_220 = 0; idx_l_220 < dim_l; idx_l_220 += 1) {
          for (int idx_chan_out_220 = threadIdx.x; idx_chan_out_220 < dim_0; idx_chan_out_220 += blockDim.x) {
            float P_oi_220 = P_220[((idx_chan_out_220)*dim_l + idx_l_220)*dim_2 + idx_chan_in_2];
            accum_220_0 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 0];
            accum_220_1 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 1];
            accum_220_2 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 2];
            accum_220_3 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 3];
            accum_220_4 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 4];
            accum_220_5 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 5];
            accum_220_6 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 6];
            accum_220_7 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 7];
            accum_220_8 += P_oi_220*dproduct_220[((idx_l_220)*dim_0 + idx_chan_out_220)*9 + 8];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_220_0 += __shfl_down_sync(0xffffffff, accum_220_0, offset);
          accum_220_1 += __shfl_down_sync(0xffffffff, accum_220_1, offset);
          accum_220_2 += __shfl_down_sync(0xffffffff, accum_220_2, offset);
          accum_220_3 += __shfl_down_sync(0xffffffff, accum_220_3, offset);
          accum_220_4 += __shfl_down_sync(0xffffffff, accum_220_4, offset);
          accum_220_5 += __shfl_down_sync(0xffffffff, accum_220_5, offset);
          accum_220_6 += __shfl_down_sync(0xffffffff, accum_220_6, offset);
          accum_220_7 += __shfl_down_sync(0xffffffff, accum_220_7, offset);
          accum_220_8 += __shfl_down_sync(0xffffffff, accum_220_8, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_2_0 += accum_220_0;
          dx_o_2_1 += accum_220_1;
          dx_o_2_2 += accum_220_2;
          dx_o_2_3 += accum_220_3;
          dx_o_2_4 += accum_220_4;
          dx_o_2_5 += accum_220_5;
          dx_o_2_6 += accum_220_6;
          dx_o_2_7 += accum_220_7;
          dx_o_2_8 += accum_220_8;
        }
        float accum_222_0 = 0.0;
        float accum_222_1 = 0.0;
        float accum_222_2 = 0.0;
        float accum_222_3 = 0.0;
        float accum_222_4 = 0.0;
        float accum_222_5 = 0.0;
        float accum_222_6 = 0.0;
        float accum_222_7 = 0.0;
        float accum_222_8 = 0.0;
        for (int idx_l_222 = 0; idx_l_222 < dim_l; idx_l_222 += 1) {
          for (int idx_chan_out_222 = threadIdx.x; idx_chan_out_222 < dim_2; idx_chan_out_222 += blockDim.x) {
            float P_oi_222 = P_222[((idx_chan_out_222)*dim_l + idx_l_222)*dim_2 + idx_chan_in_2];
            accum_222_0 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 0];
            accum_222_1 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 1];
            accum_222_2 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 2];
            accum_222_3 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 3];
            accum_222_4 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 4];
            accum_222_5 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 5];
            accum_222_6 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 6];
            accum_222_7 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 7];
            accum_222_8 += P_oi_222*dproduct_222[((idx_l_222)*dim_2 + idx_chan_out_222)*9 + 8];
          }
        }
        // reduce across the warp so that first thread in warp will have the sum 
        for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
          accum_222_0 += __shfl_down_sync(0xffffffff, accum_222_0, offset);
          accum_222_1 += __shfl_down_sync(0xffffffff, accum_222_1, offset);
          accum_222_2 += __shfl_down_sync(0xffffffff, accum_222_2, offset);
          accum_222_3 += __shfl_down_sync(0xffffffff, accum_222_3, offset);
          accum_222_4 += __shfl_down_sync(0xffffffff, accum_222_4, offset);
          accum_222_5 += __shfl_down_sync(0xffffffff, accum_222_5, offset);
          accum_222_6 += __shfl_down_sync(0xffffffff, accum_222_6, offset);
          accum_222_7 += __shfl_down_sync(0xffffffff, accum_222_7, offset);
          accum_222_8 += __shfl_down_sync(0xffffffff, accum_222_8, offset);
        }
        if (threadIdx.x == 0) {
          dx_o_2_0 += accum_222_0;
          dx_o_2_1 += accum_222_1;
          dx_o_2_2 += accum_222_2;
          dx_o_2_3 += accum_222_3;
          dx_o_2_4 += accum_222_4;
          dx_o_2_5 += accum_222_5;
          dx_o_2_6 += accum_222_6;
          dx_o_2_7 += accum_222_7;
          dx_o_2_8 += accum_222_8;
        }
        if (threadIdx.x == 0) {
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 0] = dx_o_2_0;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 1] = dx_o_2_1;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 2] = dx_o_2_2;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 3] = dx_o_2_3;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 4] = dx_o_2_4;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 5] = dx_o_2_5;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 6] = dx_o_2_6;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 7] = dx_o_2_7;
          dx_2[((idx_batch)*dim_2 + idx_chan_in_2)*9 + 8] = dx_o_2_8;
        }
      }
    }
  }
}


void fused_tensor_prods_example_backward(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* dx_0, float* dx_1, float* dx_2) {
  
  int p_0 = dim_l*dim_0;
  int p_1 = dim_l*dim_1;
  int p_2 = dim_l*dim_2;
  int sharedmemsz = 0;
  sharedmemsz += 13*p_0;
  int p_1_base = sharedmemsz;
  sharedmemsz += 10*p_1;
  int p_2_base = sharedmemsz;
  sharedmemsz += 12*p_2;
  dim3 gridsz = dim3(batch);
  dim3 blocksz = dim3(WARPSZ, dim_l);
  fused_tensor_prods_example_backward_kern<<<gridsz, blocksz, sharedmemsz*sizeof(float)>>>(
      p_0, p_1_base, p_1, p_2_base, p_2, 
      batch, dim_l, dim_0, dim_1, dim_2,
      dy_0, dy_1, dy_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212,
      dx_0, dx_1, dx_2);
  
}


__global__
void fused_tensor_prods_example_backleft_kern(
    // <<<(batch), (WARPSZ, dim_l)>>>
    
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* P_011, const float* P_101, const float* P_110, const float* P_220, const float* P_222, const float* P_211, const float* P_111, const float* P_212,
    float* __restrict__ dleft_000, float* __restrict__ dleft_011, float* __restrict__ dleft_101, float* __restrict__ dleft_110, float* __restrict__ dleft_220, float* __restrict__ dleft_222, float* __restrict__ dleft_211, float* __restrict__ dleft_111, float* __restrict__ dleft_212) {
  extern __shared__ float s[];
  for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {
    { // compute left derivative tensor products
      float accum_000_0 = 0.0;
      for (int idx_chan_in_000 = threadIdx.x; idx_chan_in_000 < dim_0; idx_chan_in_000 += blockDim.x) {
        for (int idx_chan_out_000 = 0; idx_chan_out_000 < dim_0; idx_chan_out_000 += 1) {
          float l_000_0 = x_0[((idx_batch)*dim_0 + idx_chan_in_000)*1 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_000)*1 + 0];
          float P_oi_000 = P_000[((idx_chan_out_000)*blockDim.y + threadIdx.y)*dim_0 + idx_chan_in_000];
          accum_000_0 += P_oi_000*l_000_0;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_000_0 += __shfl_down_sync(0xffffffff, accum_000_0, offset);
      }
      if (threadIdx.x == 0) {
        dleft_000[((idx_batch)*dim_l + threadIdx.y)*1 + 0] = accum_000_0;
      }
      float accum_011_0 = 0.0;
      for (int idx_chan_in_011 = threadIdx.x; idx_chan_in_011 < dim_1; idx_chan_in_011 += blockDim.x) {
        for (int idx_chan_out_011 = 0; idx_chan_out_011 < dim_1; idx_chan_out_011 += 1) {
          float l_011_0 = x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 0] + x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 1] + x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 2];
          float P_oi_011 = P_011[((idx_chan_out_011)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_011];
          accum_011_0 += P_oi_011*l_011_0;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_011_0 += __shfl_down_sync(0xffffffff, accum_011_0, offset);
      }
      if (threadIdx.x == 0) {
        dleft_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0] = accum_011_0;
      }
      float accum_101_0 = 0.0;
      float accum_101_1 = 0.0;
      float accum_101_2 = 0.0;
      for (int idx_chan_in_101 = threadIdx.x; idx_chan_in_101 < dim_0; idx_chan_in_101 += blockDim.x) {
        for (int idx_chan_out_101 = 0; idx_chan_out_101 < dim_1; idx_chan_out_101 += 1) {
          float l_101_0 = x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 0];
          float l_101_1 = x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 1];
          float l_101_2 = x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 2];
          float P_oi_101 = P_101[((idx_chan_out_101)*blockDim.y + threadIdx.y)*dim_0 + idx_chan_in_101];
          accum_101_0 += P_oi_101*l_101_0;
          accum_101_1 += P_oi_101*l_101_1;
          accum_101_2 += P_oi_101*l_101_2;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_101_0 += __shfl_down_sync(0xffffffff, accum_101_0, offset);
        accum_101_1 += __shfl_down_sync(0xffffffff, accum_101_1, offset);
        accum_101_2 += __shfl_down_sync(0xffffffff, accum_101_2, offset);
      }
      if (threadIdx.x == 0) {
        dleft_101[((idx_batch)*dim_l + threadIdx.y)*3 + 0] = accum_101_0;
        dleft_101[((idx_batch)*dim_l + threadIdx.y)*3 + 1] = accum_101_1;
        dleft_101[((idx_batch)*dim_l + threadIdx.y)*3 + 2] = accum_101_2;
      }
      float accum_110_0 = 0.0;
      float accum_110_1 = 0.0;
      float accum_110_2 = 0.0;
      for (int idx_chan_in_110 = threadIdx.x; idx_chan_in_110 < dim_1; idx_chan_in_110 += blockDim.x) {
        for (int idx_chan_out_110 = 0; idx_chan_out_110 < dim_0; idx_chan_out_110 += 1) {
          float l_110_0 = x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
          float l_110_1 = x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 1]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
          float l_110_2 = x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 2]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0];
          float P_oi_110 = P_110[((idx_chan_out_110)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_110];
          accum_110_0 += P_oi_110*l_110_0;
          accum_110_1 += P_oi_110*l_110_1;
          accum_110_2 += P_oi_110*l_110_2;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_110_0 += __shfl_down_sync(0xffffffff, accum_110_0, offset);
        accum_110_1 += __shfl_down_sync(0xffffffff, accum_110_1, offset);
        accum_110_2 += __shfl_down_sync(0xffffffff, accum_110_2, offset);
      }
      if (threadIdx.x == 0) {
        dleft_110[((idx_batch)*dim_l + threadIdx.y)*3 + 0] = accum_110_0;
        dleft_110[((idx_batch)*dim_l + threadIdx.y)*3 + 1] = accum_110_1;
        dleft_110[((idx_batch)*dim_l + threadIdx.y)*3 + 2] = accum_110_2;
      }
      float accum_220_0 = 0.0;
      float accum_220_1 = 0.0;
      float accum_220_2 = 0.0;
      float accum_220_3 = 0.0;
      float accum_220_4 = 0.0;
      float accum_220_5 = 0.0;
      float accum_220_6 = 0.0;
      float accum_220_7 = 0.0;
      float accum_220_8 = 0.0;
      for (int idx_chan_in_220 = threadIdx.x; idx_chan_in_220 < dim_2; idx_chan_in_220 += blockDim.x) {
        for (int idx_chan_out_220 = 0; idx_chan_out_220 < dim_0; idx_chan_out_220 += 1) {
          float l_220_0 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_1 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 1]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_2 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 2]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_3 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 3]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_4 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 4]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_5 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 5]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_6 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 6]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_7 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 7]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float l_220_8 = x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 8]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0];
          float P_oi_220 = P_220[((idx_chan_out_220)*blockDim.y + threadIdx.y)*dim_2 + idx_chan_in_220];
          accum_220_0 += P_oi_220*l_220_0;
          accum_220_1 += P_oi_220*l_220_1;
          accum_220_2 += P_oi_220*l_220_2;
          accum_220_3 += P_oi_220*l_220_3;
          accum_220_4 += P_oi_220*l_220_4;
          accum_220_5 += P_oi_220*l_220_5;
          accum_220_6 += P_oi_220*l_220_6;
          accum_220_7 += P_oi_220*l_220_7;
          accum_220_8 += P_oi_220*l_220_8;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_220_0 += __shfl_down_sync(0xffffffff, accum_220_0, offset);
        accum_220_1 += __shfl_down_sync(0xffffffff, accum_220_1, offset);
        accum_220_2 += __shfl_down_sync(0xffffffff, accum_220_2, offset);
        accum_220_3 += __shfl_down_sync(0xffffffff, accum_220_3, offset);
        accum_220_4 += __shfl_down_sync(0xffffffff, accum_220_4, offset);
        accum_220_5 += __shfl_down_sync(0xffffffff, accum_220_5, offset);
        accum_220_6 += __shfl_down_sync(0xffffffff, accum_220_6, offset);
        accum_220_7 += __shfl_down_sync(0xffffffff, accum_220_7, offset);
        accum_220_8 += __shfl_down_sync(0xffffffff, accum_220_8, offset);
      }
      if (threadIdx.x == 0) {
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 0] = accum_220_0;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 1] = accum_220_1;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 2] = accum_220_2;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 3] = accum_220_3;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 4] = accum_220_4;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 5] = accum_220_5;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 6] = accum_220_6;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 7] = accum_220_7;
        dleft_220[((idx_batch)*dim_l + threadIdx.y)*9 + 8] = accum_220_8;
      }
      float accum_222_0 = 0.0;
      float accum_222_1 = 0.0;
      float accum_222_2 = 0.0;
      float accum_222_3 = 0.0;
      float accum_222_4 = 0.0;
      float accum_222_5 = 0.0;
      float accum_222_6 = 0.0;
      float accum_222_7 = 0.0;
      float accum_222_8 = 0.0;
      for (int idx_chan_in_222 = threadIdx.x; idx_chan_in_222 < dim_2; idx_chan_in_222 += blockDim.x) {
        for (int idx_chan_out_222 = 0; idx_chan_out_222 < dim_2; idx_chan_out_222 += 1) {
          float l_222_0 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2];
          float l_222_1 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2];
          float l_222_2 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2];
          float l_222_3 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5];
          float l_222_4 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5];
          float l_222_5 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5];
          float l_222_6 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
          float l_222_7 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
          float l_222_8 = x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8];
          float P_oi_222 = P_222[((idx_chan_out_222)*blockDim.y + threadIdx.y)*dim_2 + idx_chan_in_222];
          accum_222_0 += P_oi_222*l_222_0;
          accum_222_1 += P_oi_222*l_222_1;
          accum_222_2 += P_oi_222*l_222_2;
          accum_222_3 += P_oi_222*l_222_3;
          accum_222_4 += P_oi_222*l_222_4;
          accum_222_5 += P_oi_222*l_222_5;
          accum_222_6 += P_oi_222*l_222_6;
          accum_222_7 += P_oi_222*l_222_7;
          accum_222_8 += P_oi_222*l_222_8;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_222_0 += __shfl_down_sync(0xffffffff, accum_222_0, offset);
        accum_222_1 += __shfl_down_sync(0xffffffff, accum_222_1, offset);
        accum_222_2 += __shfl_down_sync(0xffffffff, accum_222_2, offset);
        accum_222_3 += __shfl_down_sync(0xffffffff, accum_222_3, offset);
        accum_222_4 += __shfl_down_sync(0xffffffff, accum_222_4, offset);
        accum_222_5 += __shfl_down_sync(0xffffffff, accum_222_5, offset);
        accum_222_6 += __shfl_down_sync(0xffffffff, accum_222_6, offset);
        accum_222_7 += __shfl_down_sync(0xffffffff, accum_222_7, offset);
        accum_222_8 += __shfl_down_sync(0xffffffff, accum_222_8, offset);
      }
      if (threadIdx.x == 0) {
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0] = accum_222_0;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1] = accum_222_1;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2] = accum_222_2;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3] = accum_222_3;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4] = accum_222_4;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5] = accum_222_5;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6] = accum_222_6;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7] = accum_222_7;
        dleft_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8] = accum_222_8;
      }
      float accum_211_0 = 0.0;
      float accum_211_1 = 0.0;
      float accum_211_2 = 0.0;
      float accum_211_3 = 0.0;
      float accum_211_4 = 0.0;
      float accum_211_5 = 0.0;
      float accum_211_6 = 0.0;
      float accum_211_7 = 0.0;
      float accum_211_8 = 0.0;
      for (int idx_chan_in_211 = threadIdx.x; idx_chan_in_211 < dim_1; idx_chan_in_211 += blockDim.x) {
        for (int idx_chan_out_211 = 0; idx_chan_out_211 < dim_1; idx_chan_out_211 += 1) {
          float l_211_0 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0];
          float l_211_1 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0];
          float l_211_2 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0];
          float l_211_3 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1];
          float l_211_4 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1];
          float l_211_5 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1];
          float l_211_6 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
          float l_211_7 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
          float l_211_8 = x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2];
          float P_oi_211 = P_211[((idx_chan_out_211)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_211];
          accum_211_0 += P_oi_211*l_211_0;
          accum_211_1 += P_oi_211*l_211_1;
          accum_211_2 += P_oi_211*l_211_2;
          accum_211_3 += P_oi_211*l_211_3;
          accum_211_4 += P_oi_211*l_211_4;
          accum_211_5 += P_oi_211*l_211_5;
          accum_211_6 += P_oi_211*l_211_6;
          accum_211_7 += P_oi_211*l_211_7;
          accum_211_8 += P_oi_211*l_211_8;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_211_0 += __shfl_down_sync(0xffffffff, accum_211_0, offset);
        accum_211_1 += __shfl_down_sync(0xffffffff, accum_211_1, offset);
        accum_211_2 += __shfl_down_sync(0xffffffff, accum_211_2, offset);
        accum_211_3 += __shfl_down_sync(0xffffffff, accum_211_3, offset);
        accum_211_4 += __shfl_down_sync(0xffffffff, accum_211_4, offset);
        accum_211_5 += __shfl_down_sync(0xffffffff, accum_211_5, offset);
        accum_211_6 += __shfl_down_sync(0xffffffff, accum_211_6, offset);
        accum_211_7 += __shfl_down_sync(0xffffffff, accum_211_7, offset);
        accum_211_8 += __shfl_down_sync(0xffffffff, accum_211_8, offset);
      }
      if (threadIdx.x == 0) {
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 0] = accum_211_0;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 1] = accum_211_1;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 2] = accum_211_2;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 3] = accum_211_3;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 4] = accum_211_4;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 5] = accum_211_5;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 6] = accum_211_6;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 7] = accum_211_7;
        dleft_211[((idx_batch)*dim_l + threadIdx.y)*9 + 8] = accum_211_8;
      }
      float accum_111_0 = 0.0;
      float accum_111_1 = 0.0;
      float accum_111_2 = 0.0;
      for (int idx_chan_in_111 = threadIdx.x; idx_chan_in_111 < dim_1; idx_chan_in_111 += blockDim.x) {
        for (int idx_chan_out_111 = 0; idx_chan_out_111 < dim_1; idx_chan_out_111 += 1) {
          float l_111_0 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1] + x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2];
          float l_111_1 = x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2];
          float l_111_2 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1];
          float P_oi_111 = P_111[((idx_chan_out_111)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_111];
          accum_111_0 += P_oi_111*l_111_0;
          accum_111_1 += P_oi_111*l_111_1;
          accum_111_2 += P_oi_111*l_111_2;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_111_0 += __shfl_down_sync(0xffffffff, accum_111_0, offset);
        accum_111_1 += __shfl_down_sync(0xffffffff, accum_111_1, offset);
        accum_111_2 += __shfl_down_sync(0xffffffff, accum_111_2, offset);
      }
      if (threadIdx.x == 0) {
        dleft_111[((idx_batch)*dim_l + threadIdx.y)*3 + 0] = accum_111_0;
        dleft_111[((idx_batch)*dim_l + threadIdx.y)*3 + 1] = accum_111_1;
        dleft_111[((idx_batch)*dim_l + threadIdx.y)*3 + 2] = accum_111_2;
      }
      float accum_212_0 = 0.0;
      float accum_212_1 = 0.0;
      float accum_212_2 = 0.0;
      float accum_212_3 = 0.0;
      float accum_212_4 = 0.0;
      float accum_212_5 = 0.0;
      float accum_212_6 = 0.0;
      float accum_212_7 = 0.0;
      float accum_212_8 = 0.0;
      for (int idx_chan_in_212 = threadIdx.x; idx_chan_in_212 < dim_1; idx_chan_in_212 += blockDim.x) {
        for (int idx_chan_out_212 = 0; idx_chan_out_212 < dim_2; idx_chan_out_212 += 1) {
          float l_212_0 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2];
          float l_212_1 = x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2];
          float l_212_2 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1];
          float l_212_3 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5];
          float l_212_4 = x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5];
          float l_212_5 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4];
          float l_212_6 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8];
          float l_212_7 = x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8];
          float l_212_8 = (-1)*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7];
          float P_oi_212 = P_212[((idx_chan_out_212)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_212];
          accum_212_0 += P_oi_212*l_212_0;
          accum_212_1 += P_oi_212*l_212_1;
          accum_212_2 += P_oi_212*l_212_2;
          accum_212_3 += P_oi_212*l_212_3;
          accum_212_4 += P_oi_212*l_212_4;
          accum_212_5 += P_oi_212*l_212_5;
          accum_212_6 += P_oi_212*l_212_6;
          accum_212_7 += P_oi_212*l_212_7;
          accum_212_8 += P_oi_212*l_212_8;
        }
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_212_0 += __shfl_down_sync(0xffffffff, accum_212_0, offset);
        accum_212_1 += __shfl_down_sync(0xffffffff, accum_212_1, offset);
        accum_212_2 += __shfl_down_sync(0xffffffff, accum_212_2, offset);
        accum_212_3 += __shfl_down_sync(0xffffffff, accum_212_3, offset);
        accum_212_4 += __shfl_down_sync(0xffffffff, accum_212_4, offset);
        accum_212_5 += __shfl_down_sync(0xffffffff, accum_212_5, offset);
        accum_212_6 += __shfl_down_sync(0xffffffff, accum_212_6, offset);
        accum_212_7 += __shfl_down_sync(0xffffffff, accum_212_7, offset);
        accum_212_8 += __shfl_down_sync(0xffffffff, accum_212_8, offset);
      }
      if (threadIdx.x == 0) {
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 0] = accum_212_0;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 1] = accum_212_1;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 2] = accum_212_2;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 3] = accum_212_3;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 4] = accum_212_4;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 5] = accum_212_5;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 6] = accum_212_6;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 7] = accum_212_7;
        dleft_212[((idx_batch)*dim_l + threadIdx.y)*9 + 8] = accum_212_8;
      }
    }
  }
}


void fused_tensor_prods_example_backleft(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* P_011, const float* P_101, const float* P_110, const float* P_220, const float* P_222, const float* P_211, const float* P_111, const float* P_212,
    float* dleft_000, float* dleft_011, float* dleft_101, float* dleft_110, float* dleft_220, float* dleft_222, float* dleft_211, float* dleft_111, float* dleft_212) {
  
  
  int sharedmemsz = 0;
  dim3 gridsz = dim3(batch);
  dim3 blocksz = dim3(WARPSZ, dim_l);
  fused_tensor_prods_example_backleft_kern<<<gridsz, blocksz, sharedmemsz*sizeof(float)>>>(
      
      batch, dim_l, dim_0, dim_1, dim_2,
      x_0, x_1, x_2, dy_0, dy_1, dy_2, P_000, P_011, P_101, P_110, P_220, P_222, P_211, P_111, P_212,
      dleft_000, dleft_011, dleft_101, dleft_110, dleft_220, dleft_222, dleft_211, dleft_111, dleft_212);
  
}


__global__
void fused_tensor_prods_example_wtsback_kern(
    // <<<(WARPSZ, WARPSZ), (WARPSZ, dim_l)>>>
    
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* left_000, const float* left_011, const float* left_101, const float* left_110, const float* left_220, const float* left_222, const float* left_211, const float* left_111, const float* left_212,
    float* __restrict__ dP_000, float* __restrict__ dP_011, float* __restrict__ dP_101, float* __restrict__ dP_110, float* __restrict__ dP_220, float* __restrict__ dP_222, float* __restrict__ dP_211, float* __restrict__ dP_111, float* __restrict__ dP_212) {
  extern __shared__ float s[];
  for (int idx_chan_in_000 = blockIdx.x; idx_chan_in_000 < dim_0; idx_chan_in_000 += gridDim.x) {
    for (int idx_chan_out_000 = blockIdx.y; idx_chan_out_000 < dim_0; idx_chan_out_000 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_000[((idx_batch)*dim_l + threadIdx.y)*1 + 0]*x_0[((idx_batch)*dim_0 + idx_chan_in_000)*1 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_000)*1 + 0]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_000[((idx_chan_out_000)*blockDim.y + threadIdx.y)*dim_0 + idx_chan_in_000] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_011 = blockIdx.x; idx_chan_in_011 < dim_1; idx_chan_in_011 += gridDim.x) {
    for (int idx_chan_out_011 = blockIdx.y; idx_chan_out_011 < dim_1; idx_chan_out_011 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 0] + left_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 1] + left_011[((idx_batch)*dim_l + threadIdx.y)*1 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_011)*3 + 2]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_011[((idx_chan_out_011)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_011] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_101 = blockIdx.x; idx_chan_in_101 < dim_0; idx_chan_in_101 += gridDim.x) {
    for (int idx_chan_out_101 = blockIdx.y; idx_chan_out_101 < dim_1; idx_chan_out_101 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 0]*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 0] + left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 1]*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 1] + left_101[((idx_batch)*dim_l + threadIdx.y)*3 + 2]*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_101)*3 + 2]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_101[((idx_chan_out_101)*blockDim.y + threadIdx.y)*dim_0 + idx_chan_in_101] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_110 = blockIdx.x; idx_chan_in_110 < dim_1; idx_chan_in_110 += gridDim.x) {
    for (int idx_chan_out_110 = blockIdx.y; idx_chan_out_110 < dim_0; idx_chan_out_110 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0] + left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 1]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0] + left_110[((idx_batch)*dim_l + threadIdx.y)*3 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 2]*dy_0[((idx_batch)*dim_0 + idx_chan_out_110)*1 + 0]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_110[((idx_chan_out_110)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_110] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_220 = blockIdx.x; idx_chan_in_220 < dim_2; idx_chan_in_220 += gridDim.x) {
    for (int idx_chan_out_220 = blockIdx.y; idx_chan_out_220 < dim_0; idx_chan_out_220 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 0]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 1]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 2]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 3]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 4]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 5]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 6]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 7]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0] + left_220[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 8]*dy_0[((idx_batch)*dim_0 + idx_chan_out_220)*1 + 0]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_220[((idx_chan_out_220)*blockDim.y + threadIdx.y)*dim_2 + idx_chan_in_220] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_222 = blockIdx.x; idx_chan_in_222 < dim_2; idx_chan_in_222 += gridDim.x) {
    for (int idx_chan_out_222 = blockIdx.y; idx_chan_out_222 < dim_2; idx_chan_out_222 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 0] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 1] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 2] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 3] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 4] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 5] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 6] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 7] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8] + left_222[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]*dy_2[((idx_batch)*dim_2 + idx_chan_out_222)*9 + 8]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_222[((idx_chan_out_222)*blockDim.y + threadIdx.y)*dim_2 + idx_chan_in_222] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_211 = blockIdx.x; idx_chan_in_211 < dim_1; idx_chan_in_211 += gridDim.x) {
    for (int idx_chan_out_211 = blockIdx.y; idx_chan_out_211 < dim_1; idx_chan_out_211 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 0] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 1] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2] + left_211[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_211)*3 + 2]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_211[((idx_chan_out_211)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_211] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_111 = blockIdx.x; idx_chan_in_111 < dim_1; idx_chan_in_111 += gridDim.x) {
    for (int idx_chan_out_111 = blockIdx.y; idx_chan_out_111 < dim_1; idx_chan_out_111 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + (-1)*left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 0] + (-1)*left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 2]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1] + left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 1] + left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 1]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2] + (-1)*left_111[((idx_batch)*dim_l + threadIdx.y)*3 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_111)*3 + 0]*dy_1[((idx_batch)*dim_1 + idx_chan_out_111)*3 + 2]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_111[((idx_chan_out_111)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_111] = dP_oi;
      }
    }
  }
  for (int idx_chan_in_212 = blockIdx.x; idx_chan_in_212 < dim_1; idx_chan_in_212 += gridDim.x) {
    for (int idx_chan_out_212 = blockIdx.y; idx_chan_out_212 < dim_2; idx_chan_out_212 += gridDim.y) {
      float dP_oi = 0.0;
      for (int idx_batch = threadIdx.x; idx_batch < batch; idx_batch += blockDim.x) {
        dP_oi += (left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 0] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 2]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 1] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 0]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 1]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 2] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 3] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 5]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 4] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 3]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 4]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 5] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 6] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 2]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 8]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 7] + left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 6]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 1]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8] + (-1)*left_212[((idx_batch)*dim_l + threadIdx.y)*9 + 7]*x_1[((idx_batch)*dim_1 + idx_chan_in_212)*3 + 0]*dy_2[((idx_batch)*dim_2 + idx_chan_out_212)*9 + 8]);
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        dP_oi += __shfl_down_sync(0xffffffff, dP_oi, offset);
      }
      if (threadIdx.x == 0) {
        dP_212[((idx_chan_out_212)*blockDim.y + threadIdx.y)*dim_1 + idx_chan_in_212] = dP_oi;
      }
    }
  }
}


void fused_tensor_prods_example_wtsback(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* left_000, const float* left_011, const float* left_101, const float* left_110, const float* left_220, const float* left_222, const float* left_211, const float* left_111, const float* left_212,
    float* dP_000, float* dP_011, float* dP_101, float* dP_110, float* dP_220, float* dP_222, float* dP_211, float* dP_111, float* dP_212) {
  
  
  int sharedmemsz = 0;
  dim3 gridsz = dim3(WARPSZ, WARPSZ);
  dim3 blocksz = dim3(WARPSZ, dim_l);
  fused_tensor_prods_example_wtsback_kern<<<gridsz, blocksz, sharedmemsz*sizeof(float)>>>(
      
      batch, dim_l, dim_0, dim_1, dim_2,
      x_0, x_1, x_2, dy_0, dy_1, dy_2, left_000, left_011, left_101, left_110, left_220, left_222, left_211, left_111, left_212,
      dP_000, dP_011, dP_101, dP_110, dP_220, dP_222, dP_211, dP_111, dP_212);
  
}


