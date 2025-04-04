
#define WARPSZ 32



__global__
void fused_tensor_prods_example_kern(
    // <<<(batch), (WARPSZ, dim_l)>>>
    int dim_l, int p_0_base, int p_0, int p_1_base, int p_1, int p_2_base, int p_2,
    int batch, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* W_000, const float* W_011, const float* W_101, const float* W_110, const float* W_220, const float* W_222, const float* W_211, const float* P_000, const float* P_110, const float* P_220, const float* P_011, const float* P_101, const float* P_211, const float* P_222,
    float* __restrict__ y_0, float* __restrict__ y_1, float* __restrict__ y_2) {
  extern __shared__ float s[];
  float* left_000 = &s[0*dim_l]; // size = 1*dim_l
  float* left_011 = &s[1*dim_l]; // size = 1*dim_l
  float* left_101 = &s[2*dim_l]; // size = 3*dim_l
  float* left_110 = &s[5*dim_l]; // size = 3*dim_l
  float* left_220 = &s[8*dim_l]; // size = 9*dim_l
  float* left_222 = &s[17*dim_l]; // size = 9*dim_l
  float* left_211 = &s[26*dim_l]; // size = 9*dim_l
  float* product_000 = &s[p_0_base + 0*p_0]; // size = 1*p_0
  float* product_011 = &s[p_1_base + 0*p_1]; // size = 3*p_1
  float* product_101 = &s[p_0_base + 1*p_0]; // size = 3*p_0
  float* product_110 = &s[p_1_base + 3*p_1]; // size = 1*p_1
  float* product_220 = &s[p_2_base + 0*p_2]; // size = 1*p_2
  float* product_222 = &s[p_2_base + 1*p_2]; // size = 9*p_2
  float* product_211 = &s[p_1_base + 4*p_1]; // size = 3*p_1
  for (int idx_batch = blockIdx.x; idx_batch < batch; idx_batch += gridDim.x) {
    { // linear transform to compute the left sides of the products
      float accum_000_0 = 0.0;
      for (int idx_chan_in_000 = threadIdx.x; idx_chan_in_000 < dim_0; idx_chan_in_000 += blockDim.x) {
        float W_oi_000 = W_000[(threadIdx.y)*dim_0 + idx_chan_in_000];
        accum_000_0 += W_oi_000*x_0[((idx_batch)*dim_0 + idx_chan_in_000)*1 + 0];
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_000_0 += __shfl_down_sync(0xffffffff, accum_000_0, offset);
      }
      if (threadIdx.x == 0) {
        left_000[(threadIdx.y)*1 + 0] = accum_000_0;
      }
      float accum_011_0 = 0.0;
      for (int idx_chan_in_011 = threadIdx.x; idx_chan_in_011 < dim_0; idx_chan_in_011 += blockDim.x) {
        float W_oi_011 = W_011[(threadIdx.y)*dim_0 + idx_chan_in_011];
        accum_011_0 += W_oi_011*x_0[((idx_batch)*dim_0 + idx_chan_in_011)*1 + 0];
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_011_0 += __shfl_down_sync(0xffffffff, accum_011_0, offset);
      }
      if (threadIdx.x == 0) {
        left_011[(threadIdx.y)*1 + 0] = accum_011_0;
      }
      float accum_101_0 = 0.0;
      float accum_101_1 = 0.0;
      float accum_101_2 = 0.0;
      for (int idx_chan_in_101 = threadIdx.x; idx_chan_in_101 < dim_1; idx_chan_in_101 += blockDim.x) {
        float W_oi_101 = W_101[(threadIdx.y)*dim_1 + idx_chan_in_101];
        accum_101_0 += W_oi_101*x_1[((idx_batch)*dim_1 + idx_chan_in_101)*3 + 0];
        accum_101_1 += W_oi_101*x_1[((idx_batch)*dim_1 + idx_chan_in_101)*3 + 1];
        accum_101_2 += W_oi_101*x_1[((idx_batch)*dim_1 + idx_chan_in_101)*3 + 2];
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_101_0 += __shfl_down_sync(0xffffffff, accum_101_0, offset);
        accum_101_1 += __shfl_down_sync(0xffffffff, accum_101_1, offset);
        accum_101_2 += __shfl_down_sync(0xffffffff, accum_101_2, offset);
      }
      if (threadIdx.x == 0) {
        left_101[(threadIdx.y)*3 + 0] = accum_101_0;
        left_101[(threadIdx.y)*3 + 1] = accum_101_1;
        left_101[(threadIdx.y)*3 + 2] = accum_101_2;
      }
      float accum_110_0 = 0.0;
      float accum_110_1 = 0.0;
      float accum_110_2 = 0.0;
      for (int idx_chan_in_110 = threadIdx.x; idx_chan_in_110 < dim_1; idx_chan_in_110 += blockDim.x) {
        float W_oi_110 = W_110[(threadIdx.y)*dim_1 + idx_chan_in_110];
        accum_110_0 += W_oi_110*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 0];
        accum_110_1 += W_oi_110*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 1];
        accum_110_2 += W_oi_110*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 2];
      }
      // reduce across the warp so that first thread in warp will have the sum 
      for (int offset = WARPSZ/2; offset >= 1; offset >>= 1) {
        accum_110_0 += __shfl_down_sync(0xffffffff, accum_110_0, offset);
        accum_110_1 += __shfl_down_sync(0xffffffff, accum_110_1, offset);
        accum_110_2 += __shfl_down_sync(0xffffffff, accum_110_2, offset);
      }
      if (threadIdx.x == 0) {
        left_110[(threadIdx.y)*3 + 0] = accum_110_0;
        left_110[(threadIdx.y)*3 + 1] = accum_110_1;
        left_110[(threadIdx.y)*3 + 2] = accum_110_2;
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
        float W_oi_220 = W_220[(threadIdx.y)*dim_2 + idx_chan_in_220];
        accum_220_0 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 0];
        accum_220_1 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 1];
        accum_220_2 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 2];
        accum_220_3 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 3];
        accum_220_4 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 4];
        accum_220_5 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 5];
        accum_220_6 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 6];
        accum_220_7 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 7];
        accum_220_8 += W_oi_220*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 8];
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
        left_220[(threadIdx.y)*9 + 0] = accum_220_0;
        left_220[(threadIdx.y)*9 + 1] = accum_220_1;
        left_220[(threadIdx.y)*9 + 2] = accum_220_2;
        left_220[(threadIdx.y)*9 + 3] = accum_220_3;
        left_220[(threadIdx.y)*9 + 4] = accum_220_4;
        left_220[(threadIdx.y)*9 + 5] = accum_220_5;
        left_220[(threadIdx.y)*9 + 6] = accum_220_6;
        left_220[(threadIdx.y)*9 + 7] = accum_220_7;
        left_220[(threadIdx.y)*9 + 8] = accum_220_8;
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
        float W_oi_222 = W_222[(threadIdx.y)*dim_2 + idx_chan_in_222];
        accum_222_0 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0];
        accum_222_1 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1];
        accum_222_2 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2];
        accum_222_3 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3];
        accum_222_4 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4];
        accum_222_5 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5];
        accum_222_6 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6];
        accum_222_7 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7];
        accum_222_8 += W_oi_222*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8];
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
        left_222[(threadIdx.y)*9 + 0] = accum_222_0;
        left_222[(threadIdx.y)*9 + 1] = accum_222_1;
        left_222[(threadIdx.y)*9 + 2] = accum_222_2;
        left_222[(threadIdx.y)*9 + 3] = accum_222_3;
        left_222[(threadIdx.y)*9 + 4] = accum_222_4;
        left_222[(threadIdx.y)*9 + 5] = accum_222_5;
        left_222[(threadIdx.y)*9 + 6] = accum_222_6;
        left_222[(threadIdx.y)*9 + 7] = accum_222_7;
        left_222[(threadIdx.y)*9 + 8] = accum_222_8;
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
      for (int idx_chan_in_211 = threadIdx.x; idx_chan_in_211 < dim_2; idx_chan_in_211 += blockDim.x) {
        float W_oi_211 = W_211[(threadIdx.y)*dim_2 + idx_chan_in_211];
        accum_211_0 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 0];
        accum_211_1 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 1];
        accum_211_2 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 2];
        accum_211_3 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 3];
        accum_211_4 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 4];
        accum_211_5 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 5];
        accum_211_6 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 6];
        accum_211_7 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 7];
        accum_211_8 += W_oi_211*x_2[((idx_batch)*dim_2 + idx_chan_in_211)*9 + 8];
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
        left_211[(threadIdx.y)*9 + 0] = accum_211_0;
        left_211[(threadIdx.y)*9 + 1] = accum_211_1;
        left_211[(threadIdx.y)*9 + 2] = accum_211_2;
        left_211[(threadIdx.y)*9 + 3] = accum_211_3;
        left_211[(threadIdx.y)*9 + 4] = accum_211_4;
        left_211[(threadIdx.y)*9 + 5] = accum_211_5;
        left_211[(threadIdx.y)*9 + 6] = accum_211_6;
        left_211[(threadIdx.y)*9 + 7] = accum_211_7;
        left_211[(threadIdx.y)*9 + 8] = accum_211_8;
      }
    }
    __syncthreads();
    { // compute tensor products
      float l_000_0 = left_000[(threadIdx.y)*1 + 0];
      for (int idx_chan_in_000 = threadIdx.x; idx_chan_in_000 < dim_0; idx_chan_in_000 += blockDim.x) {
        product_000[((threadIdx.y)*dim_0 + idx_chan_in_000)*1 + 0] = (l_000_0*x_0[((idx_batch)*dim_0 + idx_chan_in_000)*1 + 0]);
      }
      float l_011_0 = left_011[(threadIdx.y)*1 + 0];
      for (int idx_chan_in_011 = threadIdx.x; idx_chan_in_011 < dim_1; idx_chan_in_011 += blockDim.x) {
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 0] = (l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 0]);
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 1] = (l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 1]);
        product_011[((threadIdx.y)*dim_1 + idx_chan_in_011)*3 + 2] = (l_011_0*x_1[((idx_batch)*dim_1 + idx_chan_in_011)*3 + 2]);
      }
      float l_101_0 = left_101[(threadIdx.y)*3 + 0];
      float l_101_1 = left_101[(threadIdx.y)*3 + 1];
      float l_101_2 = left_101[(threadIdx.y)*3 + 2];
      for (int idx_chan_in_101 = threadIdx.x; idx_chan_in_101 < dim_0; idx_chan_in_101 += blockDim.x) {
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 0] = (l_101_0*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]);
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 1] = (l_101_1*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]);
        product_101[((threadIdx.y)*dim_0 + idx_chan_in_101)*3 + 2] = (l_101_2*x_0[((idx_batch)*dim_0 + idx_chan_in_101)*1 + 0]);
      }
      float l_110_0 = left_110[(threadIdx.y)*3 + 0];
      float l_110_1 = left_110[(threadIdx.y)*3 + 1];
      float l_110_2 = left_110[(threadIdx.y)*3 + 2];
      for (int idx_chan_in_110 = threadIdx.x; idx_chan_in_110 < dim_1; idx_chan_in_110 += blockDim.x) {
        product_110[((threadIdx.y)*dim_1 + idx_chan_in_110)*1 + 0] = (l_110_0*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 0]
            + l_110_1*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 1]
            + l_110_2*x_1[((idx_batch)*dim_1 + idx_chan_in_110)*3 + 2]);
      }
      float l_220_0 = left_220[(threadIdx.y)*9 + 0];
      float l_220_1 = left_220[(threadIdx.y)*9 + 1];
      float l_220_2 = left_220[(threadIdx.y)*9 + 2];
      float l_220_3 = left_220[(threadIdx.y)*9 + 3];
      float l_220_4 = left_220[(threadIdx.y)*9 + 4];
      float l_220_5 = left_220[(threadIdx.y)*9 + 5];
      float l_220_6 = left_220[(threadIdx.y)*9 + 6];
      float l_220_7 = left_220[(threadIdx.y)*9 + 7];
      float l_220_8 = left_220[(threadIdx.y)*9 + 8];
      for (int idx_chan_in_220 = threadIdx.x; idx_chan_in_220 < dim_2; idx_chan_in_220 += blockDim.x) {
        product_220[((threadIdx.y)*dim_2 + idx_chan_in_220)*1 + 0] = (l_220_0*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 0]
            + l_220_1*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 1]
            + l_220_2*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 2]
            + l_220_3*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 3]
            + l_220_4*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 4]
            + l_220_5*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 5]
            + l_220_6*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 6]
            + l_220_7*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 7]
            + l_220_8*x_2[((idx_batch)*dim_2 + idx_chan_in_220)*9 + 8]);
      }
      float l_222_0 = left_222[(threadIdx.y)*9 + 0];
      float l_222_1 = left_222[(threadIdx.y)*9 + 1];
      float l_222_2 = left_222[(threadIdx.y)*9 + 2];
      float l_222_3 = left_222[(threadIdx.y)*9 + 3];
      float l_222_4 = left_222[(threadIdx.y)*9 + 4];
      float l_222_5 = left_222[(threadIdx.y)*9 + 5];
      float l_222_6 = left_222[(threadIdx.y)*9 + 6];
      float l_222_7 = left_222[(threadIdx.y)*9 + 7];
      float l_222_8 = left_222[(threadIdx.y)*9 + 8];
      for (int idx_chan_in_222 = threadIdx.x; idx_chan_in_222 < dim_2; idx_chan_in_222 += blockDim.x) {
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 0] = (l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]
            + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]
            + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 1] = (l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]
            + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]
            + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 2] = (l_222_0*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]
            + l_222_1*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]
            + l_222_2*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 3] = (l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]
            + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]
            + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 4] = (l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]
            + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]
            + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 5] = (l_222_3*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]
            + l_222_4*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]
            + l_222_5*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 6] = (l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 0]
            + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 1]
            + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 2]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 7] = (l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 3]
            + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 4]
            + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 5]);
        product_222[((threadIdx.y)*dim_2 + idx_chan_in_222)*9 + 8] = (l_222_6*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 6]
            + l_222_7*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 7]
            + l_222_8*x_2[((idx_batch)*dim_2 + idx_chan_in_222)*9 + 8]);
      }
      float l_211_0 = left_211[(threadIdx.y)*9 + 0];
      float l_211_1 = left_211[(threadIdx.y)*9 + 1];
      float l_211_2 = left_211[(threadIdx.y)*9 + 2];
      float l_211_3 = left_211[(threadIdx.y)*9 + 3];
      float l_211_4 = left_211[(threadIdx.y)*9 + 4];
      float l_211_5 = left_211[(threadIdx.y)*9 + 5];
      float l_211_6 = left_211[(threadIdx.y)*9 + 6];
      float l_211_7 = left_211[(threadIdx.y)*9 + 7];
      float l_211_8 = left_211[(threadIdx.y)*9 + 8];
      for (int idx_chan_in_211 = threadIdx.x; idx_chan_in_211 < dim_1; idx_chan_in_211 += blockDim.x) {
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 0] = (l_211_0*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]
            + l_211_1*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]
            + l_211_2*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]);
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 1] = (l_211_3*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]
            + l_211_4*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]
            + l_211_5*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]);
        product_211[((threadIdx.y)*dim_1 + idx_chan_in_211)*3 + 2] = (l_211_6*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 0]
            + l_211_7*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 1]
            + l_211_8*x_1[((idx_batch)*dim_1 + idx_chan_in_211)*3 + 2]);
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
    int batch, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* W_000, const float* W_011, const float* W_101, const float* W_110, const float* W_220, const float* W_222, const float* W_211, const float* P_000, const float* P_110, const float* P_220, const float* P_011, const float* P_101, const float* P_211, const float* P_222,
    float* y_0, float* y_1, float* y_2) {
  
  int dim_l = 8;
  int p_0 = dim_l*dim_0;
  int p_1 = dim_l*dim_1;
  int p_2 = dim_l*dim_2;
  int sharedmemsz = 0;
  sharedmemsz += 35*dim_l;
  int p_0_base = sharedmemsz;
  sharedmemsz += 4*p_0;
  int p_1_base = sharedmemsz;
  sharedmemsz += 7*p_1;
  int p_2_base = sharedmemsz;
  sharedmemsz += 10*p_2;
  dim3 gridsz = dim3(batch);
  dim3 blocksz = dim3(WARPSZ, dim_l);
  fused_tensor_prods_example_kern<<<gridsz, blocksz, sharedmemsz*sizeof(float)>>>(
      dim_l, p_0_base, p_0, p_1_base, p_1, p_2_base, p_2,
      batch, dim_0, dim_1, dim_2,
      x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222,
      y_0, y_1, y_2);
  
}


