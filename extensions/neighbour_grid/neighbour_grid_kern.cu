#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_atomic_functions.h>
#include <stdio.h>
#include <vector_functions.h>
#include <vector_functions.hpp>
#include <stdexcept>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0);


__device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float a, float3 v) {
    return make_float3(a*v.x, a*v.y, a*v.z);
}

__device__ float dot(float3 a, float3 b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

__device__ float3 boxwrap(float3 r, float3 box) {
    // 0-centered wrap to box
    return make_float3(remainderf(r.x, box.x), remainderf(r.y, box.y), remainderf(r.z, box.z));
}

__device__ float3 box_delta_pos(float3 r1, float3 r2, float3 box) {
    // get shortest vector difference in position
    float3 delta = r1 - r2;
    return boxwrap(delta, box);
}


typedef struct Particle {
    int32_t index;
    float3 pos;
} Particle;


__global__ void assignParticlesToCells(
    const float3* positions,
    Particle* cellLists,
    int* cellCounts,
    float3 box,
    int3 ndiv,
    int listmax,
    int N,
    int batch,
    int* errorFlag
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < batch * N; i += stride) {
        int b = i / N;  // batch index
        int particleIndex = i % N;  // particle index within the batch

        float3 halfbox = 0.5*box;
        float3 pos = boxwrap(positions[i] - halfbox, box) + halfbox; // compensate for boxwrap being 0-centered
        int3 cellIdx;
        cellIdx.x = (int)(ndiv.x * pos.x / box.x) % ndiv.x;
        cellIdx.y = (int)(ndiv.y * pos.y / box.y) % ndiv.y;
        cellIdx.z = (int)(ndiv.z * pos.z / box.z) % ndiv.z;

        int cellListIndex = ((b * ndiv.x + cellIdx.x) * ndiv.y + cellIdx.y) * ndiv.z + cellIdx.z;

        int oldCount = atomicAdd(&cellCounts[cellListIndex], 1);
        if (oldCount < listmax) {
            int particleListIndex = (cellListIndex * listmax) + oldCount;
            cellLists[particleListIndex].index = particleIndex;
            cellLists[particleListIndex].pos = pos;
        } else {
            atomicExch(errorFlag, 1);
        }
    }
}

__global__ void createNeighbourListsKern(
    const Particle* cellLists,
    const int* cellCounts,
    int* neighbourLists,
    int* neighbourCounts,
    float3 box,
    int3 ndiv,
    int listmax,
    int neighboursmax,
    float r0sq,
    int N,
    int batch,
    int* errorFlag
) {
    int tid = blockIdx.x;
    int stride = gridDim.x;
    int ncells = ndiv.x * ndiv.y * ndiv.z;

    for (int i = tid; i < batch * ncells; i += stride) {
        int b = i / ncells;  // batch index
        int c = i % ncells;
        int3 cellIdx = make_int3(
            c / (ndiv.y * ndiv.z),
            (c / ndiv.z) % ndiv.y,
            c % ndiv.z);

        for (int particleIdx = threadIdx.x; particleIdx < cellCounts[i]; particleIdx += blockDim.x) {
            Particle self = cellLists[i*listmax + particleIdx];
            // Check neighbouring cells (3x3x3 window)
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        int3 neighbourCellIdx = make_int3(
                            (cellIdx.x + dx + ndiv.x) % ndiv.x,
                            (cellIdx.y + dy + ndiv.y) % ndiv.y,
                            (cellIdx.z + dz + ndiv.z) % ndiv.z
                        );
                        int idx = ((b * ndiv.x + neighbourCellIdx.x) * ndiv.y + neighbourCellIdx.y) * ndiv.z + neighbourCellIdx.z;
                        for (int j = 0; j < cellCounts[idx]; j++) {
                            Particle other = cellLists[idx*listmax + j];
                            float3 delta = box_delta_pos(self.pos, other.pos, box);
                            if (self.index != other.index && dot(delta, delta) <= r0sq) {
                                int oldCount = atomicAdd(&neighbourCounts[b*N + self.index], 1);
                                if (oldCount < neighboursmax) {
                                    neighbourLists[(b*N + self.index)*neighboursmax + oldCount] = other.index;
                                } else {
                                    atomicExch(errorFlag, 1);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



void createNeighbourLists(
    const float3* positions,
    int* neighbourLists,  // (batch, N, neighboursmax)
    int* neighbourCounts, // (batch, N)
    float3 box,
    int3 ndiv,
    int listmax,
    int neighboursmax,
    float r0,
    int N,
    int batch
) {
    int totalCells = ndiv.x * ndiv.y * ndiv.z * batch;
    int totalParticles = N * batch;

    Particle* d_cellLists;
    int* d_cellCounts;
    int* d_errorFlag;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_cellLists, totalCells * listmax * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc((void**)&d_cellCounts, totalCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_errorFlag, sizeof(int)));

    // Initialize device memory
    CUDA_CHECK(cudaMemset(d_cellCounts, 0, totalCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errorFlag, 0, sizeof(int)));

    // Launch assign cells kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalParticles + threadsPerBlock - 1) / threadsPerBlock;
    assignParticlesToCells<<<blocksPerGrid, threadsPerBlock>>>(
        positions, d_cellLists, d_cellCounts, box, ndiv, listmax, N, batch, d_errorFlag
    );

    // Synchronize the device to ensure all operations are completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for errors
    int h_errorFlag = 0;
    CUDA_CHECK(cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_errorFlag) {
        throw std::runtime_error("Cell list overflow occurred. Trying increasing listmax.");
    }

    // Initialize device memory
    CUDA_CHECK(cudaMemset(neighbourCounts, 0, totalParticles * sizeof(int)));
    // Launch create neighbour lists kernel
    if (listmax != 32 && listmax != 64 && listmax != 128 && listmax != 256) {
        throw std::runtime_error("expected listmax to be one of 32, 64, 128, 256. other values not supported!");
    }
    threadsPerBlock = listmax;
    blocksPerGrid = totalCells;
    float r0sq = r0*r0;
    createNeighbourListsKern<<<blocksPerGrid, threadsPerBlock>>>(
        d_cellLists, d_cellCounts, neighbourLists, neighbourCounts,
        box, ndiv, listmax, neighboursmax, r0sq, N, batch, d_errorFlag
    );

    // Check for errors
    CUDA_CHECK(cudaMemcpy(&h_errorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_errorFlag) {
        throw std::runtime_error("Neighbour list overflow occurred. Try increasing neighboursmax.");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_cellLists));
    CUDA_CHECK(cudaFree(d_cellCounts));
    CUDA_CHECK(cudaFree(d_errorFlag));
}


__global__ void fillEdgeDataKern(
    const int* neighbours,
    const int* neighbourCounts,
    const float* x,
    float* edgeData,
    int N,
    int batch,
    int neighboursmax,
    int chan
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < batch * N; i += stride) {
        int b = i / N;  // batch index

        int count = neighbourCounts[i];

        for (int j = 0; j < neighboursmax; j++) {
            for (int c = 0; c < chan; c++) {
                if (j < count) {
                    int neighbour = neighbours[i * neighboursmax + j];
                    edgeData[(i * neighboursmax + j) * chan + c] = x[(b * N + neighbour) * chan + c];
                } else {
                    edgeData[(i * neighboursmax + j) * chan + c] = 0.0f;
                }
            }
        }
    }
}

void fillEdgeData(
    const int* neighbours,
    const int* neighbourCounts,
    const float* x,
    float* edgeData,
    int N,
    int batch,
    int neighboursmax,
    int chan
) {
    int totalNodes = N * batch;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalNodes + threadsPerBlock - 1) / threadsPerBlock;

    fillEdgeDataKern<<<blocksPerGrid, threadsPerBlock>>>(
        neighbours,
        neighbourCounts,
        x,
        edgeData,
        N,
        batch,
        neighboursmax,
        chan
    );
}

__global__ void reduceEdgeDataKern(
    const int* neighbours,
    const int* neighbourCounts,
    const float* edgeData,
    float* reducedData,
    int N,
    int batch,
    int neighboursmax,
    int chan
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < batch * N; i += stride) {
        int count = neighbourCounts[i];
        for (int c = 0; c < chan; c++) {
            float sum = 0.0f;
            for (int j = 0; j < count; j++) {
                sum += edgeData[(i * neighboursmax + j) * chan + c];
            }
            reducedData[i * chan + c] = sum;
        }
    }
}

void reduceEdgeData(
    const int* neighbours,
    const int* neighbourCounts,
    const float* edgeData,
    float* reducedData,
    int N,
    int batch,
    int neighboursmax,
    int chan
) {
    int totalNodes = N * batch;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalNodes + threadsPerBlock - 1) / threadsPerBlock;

    reduceEdgeDataKern<<<blocksPerGrid, threadsPerBlock>>>(
        neighbours,
        neighbourCounts,
        edgeData,
        reducedData,
        N,
        batch,
        neighboursmax,
        chan
    );
}

__global__ void writeSrcDstKern(
    const int* neighbours,      // (batch, N, neighboursmax)
    const int* cumCounts,       // (batch, N)
    const int* edgeCounts,      // (batch)
    int* __restrict__ src,      // (edges)
    int* __restrict__ dst,      // (edges)
    int N, int neighboursmax
) {
    int idx_batch = blockIdx.y;
    int idx_node = blockIdx.x;
    int idx_thread = threadIdx.x;
    int batch_offset = idx_batch > 0 ? edgeCounts[idx_batch - 1] : 0;
    int node_offset = idx_node > 0 ? cumCounts[idx_batch*N + idx_node - 1] : 0;
    int neighbours_count = cumCounts[idx_batch*N + idx_node] - node_offset;
    for (int i = threadIdx.x; i < neighbours_count; i += blockDim.x) {
        src[batch_offset + node_offset + i] = idx_batch*N + idx_node;
        dst[batch_offset + node_offset + i] = idx_batch*N + neighbours[(idx_batch*N + idx_node)*neighboursmax + i];
    }
}

void writeSrcDst(
    const int* neighbours,      // (batch, N, neighboursmax)
    const int* cumCounts,       // (batch, N)
    const int* edgeCounts,      // (batch)
    int* src,                   // (edges)
    int* dst,                   // (edges)
    int batch, int N, int neighboursmax
) {
    dim3 blocksPerGrid = dim3(N, batch);
    int threadsPerBlock = 32; // pick a small guy since neighboursmax is usually small

    writeSrcDstKern<<<blocksPerGrid, threadsPerBlock>>>(
        neighbours, cumCounts, edgeCounts,
        src, dst,
        N, neighboursmax
    );
}
