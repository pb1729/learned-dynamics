#include <cstdint>
#include <torch/extension.h>
#include <cuda_runtime.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


// CUDA forward declarations
template<int inds_dim>
void tensorLinear(
    const float* W,
    const float* x,
    float* out,
    int batch, int dim_in, int dim_out,
    int stride_W_0, int stride_W_1
);

template<int inds_dim>
void tensorLinearBackward(
    const float* x,
    const float* dout,
    float* dW,
    int batch, int dim_in, int dim_out
);


// C++ interface
struct TensorLinearAutogradFn : public torch::autograd::Function<TensorLinearAutogradFn> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        int64_t inds,
        const torch::Tensor &W, const torch::Tensor &x
    ) {
        // setup and tensor saving
        CHECK_CUDA(W);
        CHECK_INPUT(x);
        ctx->saved_data["inds"] = inds;
        ctx->save_for_backward({W, x});
        // run kernel on same device as input tensors
        at::Device device = W.device();
        cudaSetDevice(device.index());
        // shape checks
        TORCH_CHECK(W.dim() == 2, "expected W to be a matrix (2 dims)");
        int64_t dim_out = W.size(0);
        int64_t dim_in  = W.size(1);
        int64_t stride_W_0 = W.stride(0);
        int64_t stride_W_1 = W.stride(1);
        std::vector<int64_t> y_shape = x.sizes().vec();
        int64_t batch_dims = y_shape.size() - (1 + inds);
        TORCH_CHECK(batch_dims >= 0, "x has too few dims");
        TORCH_CHECK(y_shape[batch_dims] == dim_in, "channel dim size mismatch for x");
        for (int64_t i = batch_dims + inds; i > batch_dims; i--) {
            TORCH_CHECK(y_shape[i] == 3, "expected tensor index dim to be 3d for input x");
        }
        int64_t batch = 1;
        for (int64_t i = 0; i < batch_dims; i++) {
            batch *= y_shape[i];
        }
        // run the kernel
        y_shape[batch_dims] = dim_out; // output channel dimension may be different
        torch::Tensor y = torch::empty(y_shape, torch::dtype(torch::kFloat32).device(device));
        if (batch == 0) { // handle case where input tensor has zero size
            return y;
        }
        switch (inds) {
            case 0:
                tensorLinear<1>(
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(y.data_ptr<float>()),
                    batch, dim_in, dim_out, stride_W_0, stride_W_1);
                return y;
            case 1:
                tensorLinear<3>(
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(y.data_ptr<float>()),
                    batch, dim_in, dim_out, stride_W_0, stride_W_1);
                return y;
            case 2:
                tensorLinear<9>(
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(y.data_ptr<float>()),
                    batch, dim_in, dim_out, stride_W_0, stride_W_1);
                return y;
            default:
                TORCH_CHECK(false, "tensor_linear currently only supports a number of indices up to 2. go edit the code to add more (easy with copy/paste).");
        }
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        // recall saved info
        auto saved = ctx->get_saved_variables();
        auto W = saved[0];
        auto x = saved[1];
        int64_t inds = ctx->saved_data["inds"].toInt();
        auto dy = grad_outputs[0].contiguous();
        // run kernel on same device as input tensors
        at::Device device = x.device();
        cudaSetDevice(device.index());
        // shape checks:
        int64_t dim_out = W.size(0);
        int64_t dim_in  = W.size(1);
        int64_t stride_W_0 = W.stride(0);
        int64_t stride_W_1 = W.stride(1);
        TORCH_CHECK(x.dim() == dy.dim(), "input tensors should have same number of dims");
        int batch_dims = x.dim() - (1 + inds);
        TORCH_CHECK(batch_dims >= 0, "x has too few dims");
        for (int i = batch_dims + inds; i > batch_dims; i--) {
            TORCH_CHECK(x.size(i) == 3, "expected tensor index dim to be 3d for input x");
            TORCH_CHECK(dy.size(i) == 3, "expected tensor index dim to be 3d for input dy");
        }
        int batch = 1;
        for (int i = 0; i < batch_dims; i++) {
            batch *= x.size(i);
            TORCH_CHECK(x.size(i) == dy.size(i), "batch dims must have the same size")
        }
        TORCH_CHECK(x.size(batch_dims) == dim_in);
        TORCH_CHECK(dy.size(batch_dims) == dim_out);
        torch::Tensor dW = torch::empty({dim_out, dim_in}, torch::dtype(torch::kFloat32).device(device));
        torch::Tensor dx = torch::empty(x.sizes(), torch::dtype(torch::kFloat32).device(device));
        switch (inds) {
            case 0:
                tensorLinearBackward<1>( // compute dW
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dW.data_ptr<float>()),
                    batch, dim_in, dim_out);
                tensorLinear<1>( // compute dx
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dx.data_ptr<float>()),
                    batch, dim_out, dim_in, stride_W_1, stride_W_0); // dim_in/out and strides swapped to use W.T for backward pass!
                return {torch::Tensor(), dW, dx};
            case 1:
                tensorLinearBackward<3>( // compute dW
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dW.data_ptr<float>()),
                    batch, dim_in, dim_out);
                tensorLinear<3>( // compute dx
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dx.data_ptr<float>()),
                    batch, dim_out, dim_in, stride_W_1, stride_W_0); // dim_in/out and strides swapped to use W.T for backward pass!
                return {torch::Tensor(), dW, dx};
            case 2:
                tensorLinearBackward<9>( // compute dW
                    reinterpret_cast<float*>(x.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dW.data_ptr<float>()),
                    batch, dim_in, dim_out);
                tensorLinear<9>( // compute dx
                    reinterpret_cast<float*>(W.data_ptr<float>()),
                    reinterpret_cast<float*>(dy.data_ptr<float>()),
                    reinterpret_cast<float*>(dx.data_ptr<float>()),
                    batch, dim_out, dim_in, stride_W_1, stride_W_0); // dim_in/out and strides swapped to use W.T for backward pass!
                return {torch::Tensor(), dW, dx};
            default:
                TORCH_CHECK(false, "tensor_linear_backward currently only supports a number of indices up to 2. go edit the code to add more (easy with copy/paste).");
        }
    }
};


torch::Tensor tensor_linear(int64_t inds, const torch::Tensor &W, const torch::Tensor &x) {
    return TensorLinearAutogradFn::apply(inds, W, x);
}


// Bindings of above fns:

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_linear", &tensor_linear, "Apply linear operation to a tensor with inds indices in 3 dimensions.");
}
