#include <torch/extension.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

enum SimId {
    REPEL5,
};

// CUDA forward declarations
void repel5_sim(float3* x, float3* v, float* drag, float T, float dt, int nsteps, int batch_size, int n, unsigned long seed);

// C++ interface
void polymer_sim(SimId sim_id, torch::Tensor x, torch::Tensor v, torch::Tensor drag, float T, float dt, int nsteps) {
    CHECK_INPUT(x);
    CHECK_INPUT(v);
    CHECK_INPUT(drag);
    
    // run kernel on same device as input tensors
    at::Device device = x.device();
    cudaSetDevice(device.index());

    TORCH_CHECK(x.dim() == 3 && v.dim() == 3 && drag.dim() == 1, "Input tensors must have correct dimensions");
    TORCH_CHECK(x.size(0) == v.size(0), "Batch sizes must match");
    TORCH_CHECK(x.size(1) == v.size(1) && x.size(1) == drag.size(0), "Number of particles must match");
    TORCH_CHECK(x.size(2) == 3 && v.size(2) == 3, "Position and velocity must be 3D");

    int batch_size = x.size(0);
    int n = x.size(1);

    float3* x_ptr = reinterpret_cast<float3*>(x.data_ptr<float>());
    float3* v_ptr = reinterpret_cast<float3*>(v.data_ptr<float>());
    float* drag_ptr = drag.data_ptr<float>();

    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();

    switch (sim_id) {
        case REPEL5:
            repel5_sim(x_ptr, v_ptr, drag_ptr, T, dt, nsteps, batch_size, n, seed);
            break;
        default:
            TORCH_CHECK(false, "unrecognized simulation id!");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polymer_sim", &polymer_sim, "Polymer simulation (CUDA)");
    py::enum_<SimId>(m, "SimId")
        .value("REPEL5", SimId::REPEL5);
}
