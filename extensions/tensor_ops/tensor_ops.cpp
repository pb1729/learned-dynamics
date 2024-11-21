#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


// CUDA forward declarations
template<int inds_dim>
void tensorLinear(
    const float* W,
    const float* x,
    float* out,
    int batch, int dim_in, int dim_out
);

template<int inds_dim>
void tensorLinearBackward(
    const float* x,
    const float* dout,
    float* dW,
    int batch, int dim_in, int dim_out
);


// C++ interface
torch::Tensor tensor_linear(int inds, torch::Tensor W, torch::Tensor x) {
    CHECK_INPUT(W);
    CHECK_INPUT(x);
    TORCH_CHECK(W.dim() == 2, "expected W to be a matrix (2 dims)");
    int dim_out = W.size(0);
    int dim_in  = W.size(1);
    TORCH_CHECK(x.dim() == 3 && x.size(1) == dim_in, "expected x to have shape (batch, dim_in, inds_dim)");
    int batch = x.size(0);
    int passed_dim_inds = x.size(2);
    if (inds == 0) { // TODO: should we fall back to a regular torch linear here?
        TORCH_CHECK(passed_dim_inds == 1, "last dim of x should be 3^inds");
        torch::Tensor ans = torch::empty({batch, dim_out, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinear<1>(
            reinterpret_cast<float*>(W.data_ptr<float>()),
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(ans.data_ptr<float>()),
            batch, dim_in, dim_out);
        return ans;
    }
    if (inds == 1) {
        TORCH_CHECK(passed_dim_inds == 3, "last dim of x should be 3^inds");
        torch::Tensor ans = torch::empty({batch, dim_out, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinear<3>(
            reinterpret_cast<float*>(W.data_ptr<float>()),
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(ans.data_ptr<float>()),
            batch, dim_in, dim_out);
        return ans;
    }
    if (inds == 2) {
        TORCH_CHECK(passed_dim_inds == 9, "last dim of x should be 3^inds");
        torch::Tensor ans = torch::empty({batch, dim_out, 9}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinear<9>(
            reinterpret_cast<float*>(W.data_ptr<float>()),
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(ans.data_ptr<float>()),
            batch, dim_in, dim_out);
        return ans;
    }
    // otherwise:
    TORCH_CHECK(false, "tensorLinear currently only supports a number of indices up to 2. go edit the code to add more (easy with copy/paste).");
}


torch::Tensor tensor_linear_backward(int inds, torch::Tensor x, torch::Tensor dout) {
    CHECK_INPUT(x);
    CHECK_INPUT(dout);
    TORCH_CHECK(x.dim() == 3, "expected x to have shape (batch, dim_in, inds_dim)");
    int batch = x.size(0);
    int dim_in = x.size(1);
    int passed_dim_inds = x.size(2);
    TORCH_CHECK(dout.dim() == 3 && dout.size(0) == batch && dout.size(2) == passed_dim_inds, "dout dim mismatch");
    int dim_out  = dout.size(1);
    if (inds == 0) { // TODO: should we fall back to a regular torch linear here?
        TORCH_CHECK(passed_dim_inds == 1, "last dim of x should be 3^inds");
        torch::Tensor dW = torch::empty({dim_out, dim_in}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinearBackward<1>(
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(dout.data_ptr<float>()),
            reinterpret_cast<float*>(dW.data_ptr<float>()),
            batch, dim_in, dim_out);
        return dW;
    }
    if (inds == 1) { // TODO: should we fall back to a regular torch linear here?
        TORCH_CHECK(passed_dim_inds == 3, "last dim of x should be 3^inds");
        torch::Tensor dW = torch::empty({dim_out, dim_in}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinearBackward<3>(
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(dout.data_ptr<float>()),
            reinterpret_cast<float*>(dW.data_ptr<float>()),
            batch, dim_in, dim_out);
        return dW;
    }
    if (inds == 2) { // TODO: should we fall back to a regular torch linear here?
        TORCH_CHECK(passed_dim_inds == 9, "last dim of x should be 3^inds");
        torch::Tensor dW = torch::empty({dim_out, dim_in}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        tensorLinearBackward<9>(
            reinterpret_cast<float*>(x.data_ptr<float>()),
            reinterpret_cast<float*>(dout.data_ptr<float>()),
            reinterpret_cast<float*>(dW.data_ptr<float>()),
            batch, dim_in, dim_out);
        return dW;
    }
    // otherwise:
    TORCH_CHECK(false, "tensor_linear_backward currently only supports a number of indices up to 2. go edit the code to add more (easy with copy/paste).");
}


// Bindings of above fns:

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_linear", &tensor_linear, "Apply linear operation to a tensor with inds indices in 3 dimensions.");
    m.def("tensor_linear_backward", &tensor_linear_backward, "Backwards part of tensor_linear to compute dW.");
}

