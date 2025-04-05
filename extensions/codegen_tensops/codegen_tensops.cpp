
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


void fused_tensor_prods_example(
    int batch, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* W_000, const float* W_011, const float* W_101, const float* W_110, const float* W_220, const float* W_222, const float* W_211, const float* P_000, const float* P_110, const float* P_220, const float* P_011, const float* P_101, const float* P_211, const float* P_222,
    float* y_0, float* y_1, float* y_2);

std::vector<at::Tensor> fused_tensor_prods_example_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& W_000, const at::Tensor& W_011, const at::Tensor& W_101, const at::Tensor& W_110, const at::Tensor& W_220, const at::Tensor& W_222, const at::Tensor& W_211, const at::Tensor& P_000, const at::Tensor& P_110, const at::Tensor& P_220, const at::Tensor& P_011, const at::Tensor& P_101, const at::Tensor& P_211, const at::Tensor& P_222) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(W_000);
  CHECK_INPUT(W_011);
  CHECK_INPUT(W_101);
  CHECK_INPUT(W_110);
  CHECK_INPUT(W_220);
  CHECK_INPUT(W_222);
  CHECK_INPUT(W_211);
  CHECK_INPUT(P_000);
  CHECK_INPUT(P_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(P_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(P_211);
  CHECK_INPUT(P_222);
  at::Device device = x_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(x_0.dim() == 2, "x_0 has wrong number of axes");
  int batch = x_0.size(0);
  int dim_0 = x_0.size(1);
  TORCH_CHECK(x_1.dim() == 3, "x_1 has wrong number of axes");
  TORCH_CHECK(x_1.size(0) == batch, "x_1: expected axis 0 to have size batch");
  int dim_1 = x_1.size(1);
  TORCH_CHECK(x_1.size(2) == 3, "x_1: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.dim() == 4, "x_2 has wrong number of axes");
  TORCH_CHECK(x_2.size(0) == batch, "x_2: expected axis 0 to have size batch");
  int dim_2 = x_2.size(1);
  TORCH_CHECK(x_2.size(2) == 3, "x_2: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.size(3) == 3, "x_2: expected axis 3 to have size 3");
  TORCH_CHECK(W_000.dim() == 2, "W_000 has wrong number of axes");
  TORCH_CHECK(W_000.size(0) == 8, "W_000: expected axis 0 to have size 8");
  TORCH_CHECK(W_000.size(1) == dim_0, "W_000: expected axis 1 to have size dim_0");
  TORCH_CHECK(W_011.dim() == 2, "W_011 has wrong number of axes");
  TORCH_CHECK(W_011.size(0) == 8, "W_011: expected axis 0 to have size 8");
  TORCH_CHECK(W_011.size(1) == dim_0, "W_011: expected axis 1 to have size dim_0");
  TORCH_CHECK(W_101.dim() == 2, "W_101 has wrong number of axes");
  TORCH_CHECK(W_101.size(0) == 8, "W_101: expected axis 0 to have size 8");
  TORCH_CHECK(W_101.size(1) == dim_1, "W_101: expected axis 1 to have size dim_1");
  TORCH_CHECK(W_110.dim() == 2, "W_110 has wrong number of axes");
  TORCH_CHECK(W_110.size(0) == 8, "W_110: expected axis 0 to have size 8");
  TORCH_CHECK(W_110.size(1) == dim_1, "W_110: expected axis 1 to have size dim_1");
  TORCH_CHECK(W_220.dim() == 2, "W_220 has wrong number of axes");
  TORCH_CHECK(W_220.size(0) == 8, "W_220: expected axis 0 to have size 8");
  TORCH_CHECK(W_220.size(1) == dim_2, "W_220: expected axis 1 to have size dim_2");
  TORCH_CHECK(W_222.dim() == 2, "W_222 has wrong number of axes");
  TORCH_CHECK(W_222.size(0) == 8, "W_222: expected axis 0 to have size 8");
  TORCH_CHECK(W_222.size(1) == dim_2, "W_222: expected axis 1 to have size dim_2");
  TORCH_CHECK(W_211.dim() == 2, "W_211 has wrong number of axes");
  TORCH_CHECK(W_211.size(0) == 8, "W_211: expected axis 0 to have size 8");
  TORCH_CHECK(W_211.size(1) == dim_2, "W_211: expected axis 1 to have size dim_2");
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_000.size(1) == 8, "P_000: expected axis 1 to have size 8");
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == 8, "P_110: expected axis 1 to have size 8");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == 8, "P_220: expected axis 1 to have size 8");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_011.size(1) == 8, "P_011: expected axis 1 to have size 8");
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == 8, "P_101: expected axis 1 to have size 8");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == 8, "P_211: expected axis 1 to have size 8");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == 8, "P_222: expected axis 1 to have size 8");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  at::Tensor y_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  fused_tensor_prods_example(
      batch, dim_0, dim_1, dim_2,
      reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(W_000.data_ptr<float>()), reinterpret_cast<float*>(W_011.data_ptr<float>()), reinterpret_cast<float*>(W_101.data_ptr<float>()), reinterpret_cast<float*>(W_110.data_ptr<float>()), reinterpret_cast<float*>(W_220.data_ptr<float>()), reinterpret_cast<float*>(W_222.data_ptr<float>()), reinterpret_cast<float*>(W_211.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()),
      reinterpret_cast<float*>(y_0.data_ptr<float>()), reinterpret_cast<float*>(y_1.data_ptr<float>()), reinterpret_cast<float*>(y_2.data_ptr<float>()));
  return {y_0, y_1, y_2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_tensor_prods_example_cuda", &fused_tensor_prods_example_cuda, "fused_tensor_prods_example_cuda(x_0, x_1, x_2, W_000, W_011, W_101, W_110, W_220, W_222, W_211, P_000, P_110, P_220, P_011, P_101, P_211, P_222)");
}

