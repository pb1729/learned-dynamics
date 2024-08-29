#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define REPEL_SCALE 20.0
#define K 20.0
#define R0 0.8
#define LJ_EPSILON 0.2

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0);


__global__ void init_curand_states(curandState* states, unsigned long seed)
{ // Initialize random states
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}


// Helpful device functions for computing the force
__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float a, float3 v)
{
    return make_float3(a*v.x, a*v.y, a*v.z);
}

__device__ void operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float3 curand_randn_float3(curandState* state)
{
    return make_float3(curand_normal(state), curand_normal(state), curand_normal(state));
}

__device__ float sq(float x)
{
    return x*x;
}

__device__ float pow6(float x)
{
    float y1 = x*x;
    float y2 = y1*y1;
    return y1*y2;
}

__device__ float3 bond_force(float3 delta_x) {
    float factor = 1 - R0/sqrt(sq(delta_x.x) + sq(delta_x.y) + sq(delta_x.z));
    return (-K*factor)*delta_x;
}

__device__ float3 repel_force(float3 delta_x) {
    float sum_sq = sq(delta_x.x) + sq(delta_x.y) + sq(delta_x.z);
    float factor = REPEL_SCALE/pow6(sum_sq + LJ_EPSILON);
    return factor*delta_x;
}

__device__ float3 repel5_ff(int i, const float3* x, int n) {
    float3 delta_x;
    float3 force = make_float3(0.0, 0.0, 0.0);

    // Bond forces
    if (i > 0) {
        delta_x = x[i] - x[i - 1];
        force += bond_force(delta_x);
    }
    if (i < n - 1) {
        delta_x = x[i] - x[i + 1];
        force += bond_force(delta_x);
    }

    // Repulsive forces
    for (int j = 3; j < n; j++) { // non-bonded
        int k = (i + j - 1) % n;
        delta_x = x[i] - x[k];
        force += repel_force(delta_x);
    }

    return force;
}

__global__ void repel5_sim_kern(
    float3* __restrict__ x, float3* __restrict__ v, float3* __restrict__ a,
    const float* drag, float T, float dt,
    int nsteps, int batch_size, int n,
    curandState* states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n) return;

    int batch_idx = idx / n;
    int coord_idx = idx % n;

    float drag_i = drag[coord_idx];
    float noise_i = sqrt(2*drag_i*T*dt);

    // Initial half step for velocity
    float3 acc_i = repel5_ff(coord_idx, &(x[batch_idx*n]), n);
    v[idx] = v[idx] + 0.5*(dt*(acc_i - drag_i*v[idx]) + (sqrt(0.5)*noise_i)*curand_randn_float3(&(states[idx])));


    // Main loop
    for (int i = 0; i < nsteps - 1; i++)
    {
        x[idx] += dt * v[idx];
        __syncthreads(); // must sync threads after updating positions!
        acc_i = repel5_ff(coord_idx, &(x[batch_idx*n]), n);
        v[idx] += dt*(acc_i - drag_i*v[idx]) + noise_i*curand_randn_float3(&(states[idx]));
    }

    // Final position update
    x[idx] += dt * v[idx];
    __syncthreads(); // must sync threads after updating positions!

    // Final half step for velocity
    acc_i = repel5_ff(coord_idx, &(x[batch_idx*n]), n);
    v[idx] = v[idx] + 0.5*(dt*(acc_i - drag_i*v[idx]) + (sqrt(0.5)*noise_i)*curand_randn_float3(&(states[idx])));
}

void repel5_sim(float3* x, float3* v, float* drag, float T, float dt, int nsteps, int batch_size, int n, unsigned long seed)
{
    dim3 block_size(256);
    dim3 num_blocks((batch_size * n + block_size.x - 1) / block_size.x);

    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, batch_size * n * sizeof(curandState)));

    // Initialize random states
    init_curand_states<<<num_blocks, block_size>>>(d_states, seed);

    // create acceleration buffer
    float3* a;
    CUDA_CHECK(cudaMalloc(&a, batch_size * n * sizeof(float3)));

    // launch the kernel!
    repel5_sim_kern<<<num_blocks, block_size>>>(
        x, v, a, drag,
        T, dt, nsteps, batch_size, n,
        d_states);

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(a));
}
