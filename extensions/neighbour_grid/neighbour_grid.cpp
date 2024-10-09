#include <cmath>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector_functions.hpp>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


// CUDA forward declarations
void createNeighbourLists(
    const float3* positions,
    int* neighbourLists,  // (batch, N, neighbourmax)
    int* neighbourCounts, // (batch, N)
    float3 box,
    int3 ndiv,
    int listmax,
    int neighboursmax,
    float r0,
    int N,
    int batch
);

void fillEdgeData(
    const int* neighbours,
    const int* neighbourCounts,
    const float* x,
    float* edgeData,
    int N,
    int batch,
    int neighboursmax,
    int chan
);

void reduceEdgeData(
    const int* neighbours,
    const int* neighbourCounts,
    const float* edgeData,
    float* reducedData,
    int N,
    int batch,
    int neighboursmax,
    int chan
);


// C++ interface
std::tuple<torch::Tensor, torch::Tensor> get_neighbours(
        int listmax, int neighboursmax, float cutoff,
        float box_x, float box_y, float box_z,
        torch::Tensor x) {
    // listmax gives the maximum number of particles per cell
    // the corresponding estimated max neighbours is 4pi/3 * listmax
    // 8 * listmax is a decent choice for redundancy
    CHECK_INPUT(x);

    TORCH_CHECK(x.dim() == 3 && x.size(2) == 3, "x must have shape (batch, nodes, 3)");
    int batch = x.size(0);
    int nodes = x.size(1);

    float3 box = make_float3(box_x, box_y, box_z);
    int3 ndiv;
    ndiv.x = (int)ceilf(box.x / cutoff);
    ndiv.y = (int)ceilf(box.y / cutoff);
    ndiv.z = (int)ceilf(box.z / cutoff);
    TORCH_CHECK(ndiv.x >= 3 && ndiv.y >= 3 && ndiv.z >= 3, "box dimension must be greater than 3*cutoff");

    float3* x_ptr = reinterpret_cast<float3*>(x.data_ptr<float>());

    auto neighbourLists = torch::empty({batch, nodes, neighboursmax}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto neighbourCounts = torch::empty({batch, nodes}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    createNeighbourLists(
        x_ptr,
        reinterpret_cast<int*>(neighbourLists.data_ptr<int>()),
        reinterpret_cast<int*>(neighbourCounts.data_ptr<int>()),
        box,
        ndiv,
        listmax,
        neighboursmax,
        cutoff,
        nodes,
        batch
    );

    return std::make_tuple(neighbourLists, neighbourCounts);
}

torch::Tensor edges_read(torch::Tensor neighbourCounts, torch::Tensor neighbours, torch::Tensor x) {
    CHECK_INPUT(neighbourCounts);
    CHECK_INPUT(neighbours);
    CHECK_INPUT(x);

    TORCH_CHECK(neighbourCounts.dim() == 2, "neighbourCounts must have shape (batch, nodes)");
    TORCH_CHECK(neighbours.dim() == 3, "neighbourCounts must have shape (batch, nodes, neighboursmax)");
    TORCH_CHECK(x.dim() == 3, "x must have shape (batch, nodes, chan)");
    int batch = neighbourCounts.size(0);
    int nodes = neighbourCounts.size(1);
    TORCH_CHECK(batch == neighbours.size(0) && batch == x.size(0), "batch dimension of all tensors must match");
    TORCH_CHECK(nodes == neighbours.size(1) && nodes == x.size(1), "node dimension of all tensors must match");
    int neighboursmax = neighbours.size(2);
    int chan = x.size(2);

    auto edgeData = torch::empty({batch, nodes, neighboursmax, chan}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    fillEdgeData(
        reinterpret_cast<int*>(neighbours.data_ptr<int>()),
        reinterpret_cast<int*>(neighbourCounts.data_ptr<int>()),
        reinterpret_cast<float*>(x.data_ptr<float>()),
        reinterpret_cast<float*>(edgeData.data_ptr<float>()),
        nodes, batch, neighboursmax, chan);

    return edgeData;
}

torch::Tensor edges_reduce(torch::Tensor neighbourCounts, torch::Tensor neighbours, torch::Tensor x) {
    CHECK_INPUT(neighbourCounts);
    CHECK_INPUT(neighbours);
    CHECK_INPUT(x);

    TORCH_CHECK(neighbourCounts.dim() == 2, "neighbourCounts must have shape (batch, nodes)");
    TORCH_CHECK(neighbours.dim() == 3, "neighbours must have shape (batch, nodes, neighboursmax)");
    TORCH_CHECK(x.dim() == 4, "x must have shape (batch, nodes, neighboursmax, chan)");
    int batch = neighbourCounts.size(0);
    int nodes = neighbourCounts.size(1);
    TORCH_CHECK(batch == neighbours.size(0) && batch == x.size(0), "batch dimension of all tensors must match");
    TORCH_CHECK(nodes == neighbours.size(1) && nodes == x.size(1), "node dimension of all tensors must match");
    int neighboursmax = neighbours.size(2);
    TORCH_CHECK(neighboursmax == x.size(2), "neighboursmax dimension must match between neighbours and x");
    int chan = x.size(3);

    auto reducedData = torch::empty({batch, nodes, chan}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    reduceEdgeData(
        reinterpret_cast<int*>(neighbours.data_ptr<int>()),
        reinterpret_cast<int*>(neighbourCounts.data_ptr<int>()),
        reinterpret_cast<float*>(x.data_ptr<float>()),
        reinterpret_cast<float*>(reducedData.data_ptr<float>()),
        nodes, batch, neighboursmax, chan);

    return reducedData;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_neighbours", &get_neighbours, "Define neighbours based on cutoff radius using a grid method.");
    m.def("edges_read", &edges_read, "Read data from nodes to edges.");
    m.def("edges_reduce", &edges_reduce, "Reduce data from edges to nodes.");
}
