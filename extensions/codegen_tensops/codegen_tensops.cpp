
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


void set_kern_attributes();

void fused_tensor_prods_example(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* y_0, float* y_1, float* y_2);

void fused_tensor_prods_example_backward(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* left_000, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_110, const float* left_110, const float* P_220, const float* left_220, const float* P_222, const float* left_222, const float* P_211, const float* left_211, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* dx_0, float* dx_1, float* dx_2);

void fused_tensor_prods_example_backleft(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* P_000, const float* P_011, const float* P_101, const float* P_110, const float* P_220, const float* P_222, const float* P_211, const float* P_111, const float* P_212,
    float* dleft_000, float* dleft_011, float* dleft_101, float* dleft_110, float* dleft_220, float* dleft_222, float* dleft_211, float* dleft_111, float* dleft_212);

void fused_tensor_prods_example_wtsback(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* dy_1, const float* dy_2, const float* left_000, const float* left_011, const float* left_101, const float* left_110, const float* left_220, const float* left_222, const float* left_211, const float* left_111, const float* left_212,
    float* dP_000, float* dP_011, float* dP_101, float* dP_110, float* dP_220, float* dP_222, float* dP_211, float* dP_111, float* dP_212);

void ant16_o0(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_000, const float* left_000, const float* P_110, const float* left_110, const float* P_220, const float* left_220,
    float* y_0);

void ant16_o0_backward(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_0, const float* P_000, const float* left_000, const float* P_110, const float* left_110, const float* P_220, const float* left_220,
    float* dx_0, float* dx_1, float* dx_2);

void ant16_o0_backleft(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* P_000, const float* P_110, const float* P_220,
    float* dleft_000, float* dleft_110, float* dleft_220);

void ant16_o0_wtsback(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_0, const float* left_000, const float* left_110, const float* left_220,
    float* dP_000, float* dP_110, float* dP_220);

void ant16_o1(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_121, const float* left_121, const float* P_211, const float* left_211,
    float* y_1);

void ant16_o1_backward(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_1, const float* P_011, const float* left_011, const float* P_101, const float* left_101, const float* P_121, const float* left_121, const float* P_211, const float* left_211,
    float* dx_0, float* dx_1, float* dx_2);

void ant16_o1_backleft(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_1, const float* P_011, const float* P_101, const float* P_121, const float* P_211,
    float* dleft_011, float* dleft_101, float* dleft_121, float* dleft_211);

void ant16_o1_wtsback(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_1, const float* left_011, const float* left_101, const float* left_121, const float* left_211,
    float* dP_011, float* dP_101, float* dP_121, float* dP_211);

void ant16_o2(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* P_022, const float* left_022, const float* P_202, const float* left_202, const float* P_112, const float* left_112, const float* P_222, const float* left_222,
    float* y_2);

void ant16_o2_backward(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* dy_2, const float* P_022, const float* left_022, const float* P_202, const float* left_202, const float* P_112, const float* left_112, const float* P_222, const float* left_222,
    float* dx_0, float* dx_1, float* dx_2);

void ant16_o2_backleft(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_2, const float* P_022, const float* P_202, const float* P_112, const float* P_222,
    float* dleft_022, float* dleft_202, float* dleft_112, float* dleft_222);

void ant16_o2_wtsback(
    int batch, int dim_l, int dim_0, int dim_1, int dim_2,
    const float* x_0, const float* x_1, const float* x_2, const float* dy_2, const float* left_022, const float* left_202, const float* left_112, const float* left_222,
    float* dP_022, float* dP_202, float* dP_112, float* dP_222);

void ant16_oc(
    int batch, int dim_l, int dim_1, int dim_2,
    const float* x_1, const float* x_2, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* y_1, float* y_2);

void ant16_oc_backward(
    int batch, int dim_l, int dim_1, int dim_2,
    const float* dy_1, const float* dy_2, const float* P_111, const float* left_111, const float* P_212, const float* left_212,
    float* dx_1, float* dx_2);

void ant16_oc_backleft(
    int batch, int dim_l, int dim_1, int dim_2,
    const float* x_1, const float* x_2, const float* dy_1, const float* dy_2, const float* P_111, const float* P_212,
    float* dleft_111, float* dleft_212);

void ant16_oc_wtsback(
    int batch, int dim_l, int dim_1, int dim_2,
    const float* x_1, const float* x_2, const float* dy_1, const float* dy_2, const float* left_111, const float* left_212,
    float* dP_111, float* dP_212);

void bee_fwd(
    int batch, int chan,
    const float* l_0, const float* l_1, const float* l_2, const float* r_0, const float* r_1, const float* r_2,
    float* y_000, float* y_110, float* y_220, float* y_011, float* y_101, float* y_121, float* y_211, float* y_022, float* y_202, float* y_112, float* y_222, float* y_111, float* y_212);

void bee_bwl(
    int batch, int chan,
    const float* r_0, const float* r_1, const float* r_2, const float* dy_000, const float* dy_110, const float* dy_220, const float* dy_011, const float* dy_101, const float* dy_121, const float* dy_211, const float* dy_022, const float* dy_202, const float* dy_112, const float* dy_222, const float* dy_111, const float* dy_212,
    float* dl_0, float* dl_1, float* dl_2);

void bee_bwr(
    int batch, int chan,
    const float* l_0, const float* l_1, const float* l_2, const float* dy_000, const float* dy_110, const float* dy_220, const float* dy_011, const float* dy_101, const float* dy_121, const float* dy_211, const float* dy_022, const float* dy_202, const float* dy_112, const float* dy_222, const float* dy_111, const float* dy_212,
    float* dr_0, float* dr_1, float* dr_2);

std::vector<at::Tensor> fused_tensor_prods_example_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& P_000, const at::Tensor& left_000, const at::Tensor& P_011, const at::Tensor& left_011, const at::Tensor& P_101, const at::Tensor& left_101, const at::Tensor& P_110, const at::Tensor& left_110, const at::Tensor& P_220, const at::Tensor& left_220, const at::Tensor& P_222, const at::Tensor& left_222, const at::Tensor& P_211, const at::Tensor& left_211, const at::Tensor& P_111, const at::Tensor& left_111, const at::Tensor& P_212, const at::Tensor& left_212) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(P_000);
  CHECK_INPUT(left_000);
  CHECK_INPUT(P_011);
  CHECK_INPUT(left_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(left_101);
  CHECK_INPUT(P_110);
  CHECK_INPUT(left_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(left_220);
  CHECK_INPUT(P_222);
  CHECK_INPUT(left_222);
  CHECK_INPUT(P_211);
  CHECK_INPUT(left_211);
  CHECK_INPUT(P_111);
  CHECK_INPUT(left_111);
  CHECK_INPUT(P_212);
  CHECK_INPUT(left_212);
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
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  TORCH_CHECK(left_000.size(1) == dim_l, "left_000: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_011.size(1) == dim_l, "P_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  TORCH_CHECK(left_011.size(1) == dim_l, "left_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_111.size(1) == dim_l, "P_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  TORCH_CHECK(left_111.size(1) == dim_l, "left_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor y_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    fused_tensor_prods_example(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(y_0.data_ptr<float>()), reinterpret_cast<float*>(y_1.data_ptr<float>()), reinterpret_cast<float*>(y_2.data_ptr<float>()));
  }
  return {y_0, y_1, y_2};
}

std::vector<at::Tensor> fused_tensor_prods_example_backward_cuda(
    const at::Tensor& dy_0, const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& P_000, const at::Tensor& left_000, const at::Tensor& P_011, const at::Tensor& left_011, const at::Tensor& P_101, const at::Tensor& left_101, const at::Tensor& P_110, const at::Tensor& left_110, const at::Tensor& P_220, const at::Tensor& left_220, const at::Tensor& P_222, const at::Tensor& left_222, const at::Tensor& P_211, const at::Tensor& left_211, const at::Tensor& P_111, const at::Tensor& left_111, const at::Tensor& P_212, const at::Tensor& left_212) {
  CHECK_INPUT(dy_0);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_000);
  CHECK_INPUT(left_000);
  CHECK_INPUT(P_011);
  CHECK_INPUT(left_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(left_101);
  CHECK_INPUT(P_110);
  CHECK_INPUT(left_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(left_220);
  CHECK_INPUT(P_222);
  CHECK_INPUT(left_222);
  CHECK_INPUT(P_211);
  CHECK_INPUT(left_211);
  CHECK_INPUT(P_111);
  CHECK_INPUT(left_111);
  CHECK_INPUT(P_212);
  CHECK_INPUT(left_212);
  at::Device device = dy_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  int batch = dy_0.size(0);
  int dim_0 = dy_0.size(1);
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  int dim_1 = dy_1.size(1);
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  int dim_2 = dy_2.size(1);
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  TORCH_CHECK(left_000.size(1) == dim_l, "left_000: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_011.size(1) == dim_l, "P_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  TORCH_CHECK(left_011.size(1) == dim_l, "left_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_111.size(1) == dim_l, "P_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  TORCH_CHECK(left_111.size(1) == dim_l, "left_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor dx_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    fused_tensor_prods_example_backward(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(dx_0.data_ptr<float>()), reinterpret_cast<float*>(dx_1.data_ptr<float>()), reinterpret_cast<float*>(dx_2.data_ptr<float>()));
  }
  return {dx_0, dx_1, dx_2};
}

std::vector<at::Tensor> fused_tensor_prods_example_backleft_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_0, const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& P_000, const at::Tensor& P_011, const at::Tensor& P_101, const at::Tensor& P_110, const at::Tensor& P_220, const at::Tensor& P_222, const at::Tensor& P_211, const at::Tensor& P_111, const at::Tensor& P_212) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_0);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_000);
  CHECK_INPUT(P_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(P_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(P_222);
  CHECK_INPUT(P_211);
  CHECK_INPUT(P_111);
  CHECK_INPUT(P_212);
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
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  TORCH_CHECK(dy_0.size(0) == batch, "dy_0: expected axis 0 to have size batch");
  TORCH_CHECK(dy_0.size(1) == dim_0, "dy_0: expected axis 1 to have size dim_0");
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_011.size(1) == dim_l, "P_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_111.size(1) == dim_l, "P_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  at::Tensor dleft_000 = torch::empty({batch, dim_l}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_011 = torch::empty({batch, dim_l}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_101 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_110 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_220 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_222 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_211 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_111 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_212 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    fused_tensor_prods_example_backleft(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()),
        reinterpret_cast<float*>(dleft_000.data_ptr<float>()), reinterpret_cast<float*>(dleft_011.data_ptr<float>()), reinterpret_cast<float*>(dleft_101.data_ptr<float>()), reinterpret_cast<float*>(dleft_110.data_ptr<float>()), reinterpret_cast<float*>(dleft_220.data_ptr<float>()), reinterpret_cast<float*>(dleft_222.data_ptr<float>()), reinterpret_cast<float*>(dleft_211.data_ptr<float>()), reinterpret_cast<float*>(dleft_111.data_ptr<float>()), reinterpret_cast<float*>(dleft_212.data_ptr<float>()));
  }
  return {dleft_000, dleft_011, dleft_101, dleft_110, dleft_220, dleft_222, dleft_211, dleft_111, dleft_212};
}

std::vector<at::Tensor> fused_tensor_prods_example_wtsback_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_0, const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& left_000, const at::Tensor& left_011, const at::Tensor& left_101, const at::Tensor& left_110, const at::Tensor& left_220, const at::Tensor& left_222, const at::Tensor& left_211, const at::Tensor& left_111, const at::Tensor& left_212) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_0);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(left_000);
  CHECK_INPUT(left_011);
  CHECK_INPUT(left_101);
  CHECK_INPUT(left_110);
  CHECK_INPUT(left_220);
  CHECK_INPUT(left_222);
  CHECK_INPUT(left_211);
  CHECK_INPUT(left_111);
  CHECK_INPUT(left_212);
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
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  TORCH_CHECK(dy_0.size(0) == batch, "dy_0: expected axis 0 to have size batch");
  TORCH_CHECK(dy_0.size(1) == dim_0, "dy_0: expected axis 1 to have size dim_0");
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  int dim_l = left_000.size(1);
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  TORCH_CHECK(left_011.size(1) == dim_l, "left_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  TORCH_CHECK(left_111.size(1) == dim_l, "left_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor dP_000 = torch::empty({dim_0, dim_l, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_011 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_101 = torch::empty({dim_1, dim_l, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_110 = torch::empty({dim_0, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_220 = torch::empty({dim_0, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_222 = torch::empty({dim_2, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_211 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_111 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_212 = torch::empty({dim_2, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    fused_tensor_prods_example_wtsback(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(dP_000.data_ptr<float>()), reinterpret_cast<float*>(dP_011.data_ptr<float>()), reinterpret_cast<float*>(dP_101.data_ptr<float>()), reinterpret_cast<float*>(dP_110.data_ptr<float>()), reinterpret_cast<float*>(dP_220.data_ptr<float>()), reinterpret_cast<float*>(dP_222.data_ptr<float>()), reinterpret_cast<float*>(dP_211.data_ptr<float>()), reinterpret_cast<float*>(dP_111.data_ptr<float>()), reinterpret_cast<float*>(dP_212.data_ptr<float>()));
  } else {
    dP_000.zero_();
    dP_011.zero_();
    dP_101.zero_();
    dP_110.zero_();
    dP_220.zero_();
    dP_222.zero_();
    dP_211.zero_();
    dP_111.zero_();
    dP_212.zero_();
  }
  return {dP_000, dP_011, dP_101, dP_110, dP_220, dP_222, dP_211, dP_111, dP_212};
}

std::vector<at::Tensor> ant16_o0_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& P_000, const at::Tensor& left_000, const at::Tensor& P_110, const at::Tensor& left_110, const at::Tensor& P_220, const at::Tensor& left_220) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(P_000);
  CHECK_INPUT(left_000);
  CHECK_INPUT(P_110);
  CHECK_INPUT(left_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(left_220);
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
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  TORCH_CHECK(left_000.size(1) == dim_l, "left_000: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  at::Tensor y_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o0(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()),
        reinterpret_cast<float*>(y_0.data_ptr<float>()));
  }
  return {y_0};
}

std::vector<at::Tensor> ant16_o0_backward_cuda(
    const at::Tensor& dy_0, const at::Tensor& P_000, const at::Tensor& left_000, const at::Tensor& P_110, const at::Tensor& left_110, const at::Tensor& P_220, const at::Tensor& left_220) {
  CHECK_INPUT(dy_0);
  CHECK_INPUT(P_000);
  CHECK_INPUT(left_000);
  CHECK_INPUT(P_110);
  CHECK_INPUT(left_110);
  CHECK_INPUT(P_220);
  CHECK_INPUT(left_220);
  at::Device device = dy_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  int batch = dy_0.size(0);
  int dim_0 = dy_0.size(1);
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  TORCH_CHECK(left_000.size(1) == dim_l, "left_000: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  int dim_1 = P_110.size(2);
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  int dim_2 = P_220.size(2);
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  at::Tensor dx_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o0_backward(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()),
        reinterpret_cast<float*>(dx_0.data_ptr<float>()), reinterpret_cast<float*>(dx_1.data_ptr<float>()), reinterpret_cast<float*>(dx_2.data_ptr<float>()));
  }
  return {dx_0, dx_1, dx_2};
}

std::vector<at::Tensor> ant16_o0_backleft_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_0, const at::Tensor& P_000, const at::Tensor& P_110, const at::Tensor& P_220) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_0);
  CHECK_INPUT(P_000);
  CHECK_INPUT(P_110);
  CHECK_INPUT(P_220);
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
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  TORCH_CHECK(dy_0.size(0) == batch, "dy_0: expected axis 0 to have size batch");
  TORCH_CHECK(dy_0.size(1) == dim_0, "dy_0: expected axis 1 to have size dim_0");
  TORCH_CHECK(P_000.dim() == 3, "P_000 has wrong number of axes");
  TORCH_CHECK(P_000.size(0) == dim_0, "P_000: expected axis 0 to have size dim_0");
  int dim_l = P_000.size(1);
  TORCH_CHECK(P_000.size(2) == dim_0, "P_000: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_110.dim() == 3, "P_110 has wrong number of axes");
  TORCH_CHECK(P_110.size(0) == dim_0, "P_110: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_110.size(1) == dim_l, "P_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_110.size(2) == dim_1, "P_110: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_220.dim() == 3, "P_220 has wrong number of axes");
  TORCH_CHECK(P_220.size(0) == dim_0, "P_220: expected axis 0 to have size dim_0");
  TORCH_CHECK(P_220.size(1) == dim_l, "P_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_220.size(2) == dim_2, "P_220: expected axis 2 to have size dim_2");
  at::Tensor dleft_000 = torch::empty({batch, dim_l}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_110 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_220 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o0_backleft(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(P_000.data_ptr<float>()), reinterpret_cast<float*>(P_110.data_ptr<float>()), reinterpret_cast<float*>(P_220.data_ptr<float>()),
        reinterpret_cast<float*>(dleft_000.data_ptr<float>()), reinterpret_cast<float*>(dleft_110.data_ptr<float>()), reinterpret_cast<float*>(dleft_220.data_ptr<float>()));
  }
  return {dleft_000, dleft_110, dleft_220};
}

std::vector<at::Tensor> ant16_o0_wtsback_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_0, const at::Tensor& left_000, const at::Tensor& left_110, const at::Tensor& left_220) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_0);
  CHECK_INPUT(left_000);
  CHECK_INPUT(left_110);
  CHECK_INPUT(left_220);
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
  TORCH_CHECK(dy_0.dim() == 2, "dy_0 has wrong number of axes");
  TORCH_CHECK(dy_0.size(0) == batch, "dy_0: expected axis 0 to have size batch");
  TORCH_CHECK(dy_0.size(1) == dim_0, "dy_0: expected axis 1 to have size dim_0");
  TORCH_CHECK(left_000.dim() == 2, "left_000 has wrong number of axes");
  TORCH_CHECK(left_000.size(0) == batch, "left_000: expected axis 0 to have size batch");
  int dim_l = left_000.size(1);
  TORCH_CHECK(left_110.dim() == 3, "left_110 has wrong number of axes");
  TORCH_CHECK(left_110.size(0) == batch, "left_110: expected axis 0 to have size batch");
  TORCH_CHECK(left_110.size(1) == dim_l, "left_110: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_110.size(2) == 3, "left_110: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.dim() == 4, "left_220 has wrong number of axes");
  TORCH_CHECK(left_220.size(0) == batch, "left_220: expected axis 0 to have size batch");
  TORCH_CHECK(left_220.size(1) == dim_l, "left_220: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_220.size(2) == 3, "left_220: expected axis 2 to have size 3");
  TORCH_CHECK(left_220.size(3) == 3, "left_220: expected axis 3 to have size 3");
  at::Tensor dP_000 = torch::empty({dim_0, dim_l, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_110 = torch::empty({dim_0, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_220 = torch::empty({dim_0, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o0_wtsback(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_0.data_ptr<float>()), reinterpret_cast<float*>(left_000.data_ptr<float>()), reinterpret_cast<float*>(left_110.data_ptr<float>()), reinterpret_cast<float*>(left_220.data_ptr<float>()),
        reinterpret_cast<float*>(dP_000.data_ptr<float>()), reinterpret_cast<float*>(dP_110.data_ptr<float>()), reinterpret_cast<float*>(dP_220.data_ptr<float>()));
  } else {
    dP_000.zero_();
    dP_110.zero_();
    dP_220.zero_();
  }
  return {dP_000, dP_110, dP_220};
}

std::vector<at::Tensor> ant16_o1_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& P_011, const at::Tensor& left_011, const at::Tensor& P_101, const at::Tensor& left_101, const at::Tensor& P_121, const at::Tensor& left_121, const at::Tensor& P_211, const at::Tensor& left_211) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(P_011);
  CHECK_INPUT(left_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(left_101);
  CHECK_INPUT(P_121);
  CHECK_INPUT(left_121);
  CHECK_INPUT(P_211);
  CHECK_INPUT(left_211);
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
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  int dim_l = P_011.size(1);
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  TORCH_CHECK(left_011.size(1) == dim_l, "left_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(P_121.dim() == 3, "P_121 has wrong number of axes");
  TORCH_CHECK(P_121.size(0) == dim_1, "P_121: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_121.size(1) == dim_l, "P_121: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_121.size(2) == dim_2, "P_121: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_121.dim() == 3, "left_121 has wrong number of axes");
  TORCH_CHECK(left_121.size(0) == batch, "left_121: expected axis 0 to have size batch");
  TORCH_CHECK(left_121.size(1) == dim_l, "left_121: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_121.size(2) == 3, "left_121: expected axis 2 to have size 3");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  at::Tensor y_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o1(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(P_121.data_ptr<float>()), reinterpret_cast<float*>(left_121.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()),
        reinterpret_cast<float*>(y_1.data_ptr<float>()));
  }
  return {y_1};
}

std::vector<at::Tensor> ant16_o1_backward_cuda(
    const at::Tensor& dy_1, const at::Tensor& P_011, const at::Tensor& left_011, const at::Tensor& P_101, const at::Tensor& left_101, const at::Tensor& P_121, const at::Tensor& left_121, const at::Tensor& P_211, const at::Tensor& left_211) {
  CHECK_INPUT(dy_1);
  CHECK_INPUT(P_011);
  CHECK_INPUT(left_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(left_101);
  CHECK_INPUT(P_121);
  CHECK_INPUT(left_121);
  CHECK_INPUT(P_211);
  CHECK_INPUT(left_211);
  at::Device device = dy_1.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  int batch = dy_1.size(0);
  int dim_1 = dy_1.size(1);
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  int dim_l = P_011.size(1);
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  TORCH_CHECK(left_011.size(1) == dim_l, "left_011: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  int dim_0 = P_101.size(2);
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(P_121.dim() == 3, "P_121 has wrong number of axes");
  TORCH_CHECK(P_121.size(0) == dim_1, "P_121: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_121.size(1) == dim_l, "P_121: expected axis 1 to have size dim_l");
  int dim_2 = P_121.size(2);
  TORCH_CHECK(left_121.dim() == 3, "left_121 has wrong number of axes");
  TORCH_CHECK(left_121.size(0) == batch, "left_121: expected axis 0 to have size batch");
  TORCH_CHECK(left_121.size(1) == dim_l, "left_121: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_121.size(2) == 3, "left_121: expected axis 2 to have size 3");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  at::Tensor dx_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o1_backward(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(P_121.data_ptr<float>()), reinterpret_cast<float*>(left_121.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()),
        reinterpret_cast<float*>(dx_0.data_ptr<float>()), reinterpret_cast<float*>(dx_1.data_ptr<float>()), reinterpret_cast<float*>(dx_2.data_ptr<float>()));
  }
  return {dx_0, dx_1, dx_2};
}

std::vector<at::Tensor> ant16_o1_backleft_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_1, const at::Tensor& P_011, const at::Tensor& P_101, const at::Tensor& P_121, const at::Tensor& P_211) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(P_011);
  CHECK_INPUT(P_101);
  CHECK_INPUT(P_121);
  CHECK_INPUT(P_211);
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
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(P_011.dim() == 3, "P_011 has wrong number of axes");
  TORCH_CHECK(P_011.size(0) == dim_1, "P_011: expected axis 0 to have size dim_1");
  int dim_l = P_011.size(1);
  TORCH_CHECK(P_011.size(2) == dim_1, "P_011: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_101.dim() == 3, "P_101 has wrong number of axes");
  TORCH_CHECK(P_101.size(0) == dim_1, "P_101: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_101.size(1) == dim_l, "P_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_101.size(2) == dim_0, "P_101: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_121.dim() == 3, "P_121 has wrong number of axes");
  TORCH_CHECK(P_121.size(0) == dim_1, "P_121: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_121.size(1) == dim_l, "P_121: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_121.size(2) == dim_2, "P_121: expected axis 2 to have size dim_2");
  TORCH_CHECK(P_211.dim() == 3, "P_211 has wrong number of axes");
  TORCH_CHECK(P_211.size(0) == dim_1, "P_211: expected axis 0 to have size dim_1");
  TORCH_CHECK(P_211.size(1) == dim_l, "P_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_211.size(2) == dim_1, "P_211: expected axis 2 to have size dim_1");
  at::Tensor dleft_011 = torch::empty({batch, dim_l}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_101 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_121 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_211 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o1_backleft(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(P_011.data_ptr<float>()), reinterpret_cast<float*>(P_101.data_ptr<float>()), reinterpret_cast<float*>(P_121.data_ptr<float>()), reinterpret_cast<float*>(P_211.data_ptr<float>()),
        reinterpret_cast<float*>(dleft_011.data_ptr<float>()), reinterpret_cast<float*>(dleft_101.data_ptr<float>()), reinterpret_cast<float*>(dleft_121.data_ptr<float>()), reinterpret_cast<float*>(dleft_211.data_ptr<float>()));
  }
  return {dleft_011, dleft_101, dleft_121, dleft_211};
}

std::vector<at::Tensor> ant16_o1_wtsback_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_1, const at::Tensor& left_011, const at::Tensor& left_101, const at::Tensor& left_121, const at::Tensor& left_211) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(left_011);
  CHECK_INPUT(left_101);
  CHECK_INPUT(left_121);
  CHECK_INPUT(left_211);
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
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(left_011.dim() == 2, "left_011 has wrong number of axes");
  TORCH_CHECK(left_011.size(0) == batch, "left_011: expected axis 0 to have size batch");
  int dim_l = left_011.size(1);
  TORCH_CHECK(left_101.dim() == 3, "left_101 has wrong number of axes");
  TORCH_CHECK(left_101.size(0) == batch, "left_101: expected axis 0 to have size batch");
  TORCH_CHECK(left_101.size(1) == dim_l, "left_101: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_101.size(2) == 3, "left_101: expected axis 2 to have size 3");
  TORCH_CHECK(left_121.dim() == 3, "left_121 has wrong number of axes");
  TORCH_CHECK(left_121.size(0) == batch, "left_121: expected axis 0 to have size batch");
  TORCH_CHECK(left_121.size(1) == dim_l, "left_121: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_121.size(2) == 3, "left_121: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.dim() == 4, "left_211 has wrong number of axes");
  TORCH_CHECK(left_211.size(0) == batch, "left_211: expected axis 0 to have size batch");
  TORCH_CHECK(left_211.size(1) == dim_l, "left_211: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_211.size(2) == 3, "left_211: expected axis 2 to have size 3");
  TORCH_CHECK(left_211.size(3) == 3, "left_211: expected axis 3 to have size 3");
  at::Tensor dP_011 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_101 = torch::empty({dim_1, dim_l, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_121 = torch::empty({dim_1, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_211 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o1_wtsback(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(left_011.data_ptr<float>()), reinterpret_cast<float*>(left_101.data_ptr<float>()), reinterpret_cast<float*>(left_121.data_ptr<float>()), reinterpret_cast<float*>(left_211.data_ptr<float>()),
        reinterpret_cast<float*>(dP_011.data_ptr<float>()), reinterpret_cast<float*>(dP_101.data_ptr<float>()), reinterpret_cast<float*>(dP_121.data_ptr<float>()), reinterpret_cast<float*>(dP_211.data_ptr<float>()));
  } else {
    dP_011.zero_();
    dP_101.zero_();
    dP_121.zero_();
    dP_211.zero_();
  }
  return {dP_011, dP_101, dP_121, dP_211};
}

std::vector<at::Tensor> ant16_o2_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& P_022, const at::Tensor& left_022, const at::Tensor& P_202, const at::Tensor& left_202, const at::Tensor& P_112, const at::Tensor& left_112, const at::Tensor& P_222, const at::Tensor& left_222) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(P_022);
  CHECK_INPUT(left_022);
  CHECK_INPUT(P_202);
  CHECK_INPUT(left_202);
  CHECK_INPUT(P_112);
  CHECK_INPUT(left_112);
  CHECK_INPUT(P_222);
  CHECK_INPUT(left_222);
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
  TORCH_CHECK(P_022.dim() == 3, "P_022 has wrong number of axes");
  TORCH_CHECK(P_022.size(0) == dim_2, "P_022: expected axis 0 to have size dim_2");
  int dim_l = P_022.size(1);
  TORCH_CHECK(P_022.size(2) == dim_2, "P_022: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_022.dim() == 2, "left_022 has wrong number of axes");
  TORCH_CHECK(left_022.size(0) == batch, "left_022: expected axis 0 to have size batch");
  TORCH_CHECK(left_022.size(1) == dim_l, "left_022: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_202.dim() == 3, "P_202 has wrong number of axes");
  TORCH_CHECK(P_202.size(0) == dim_2, "P_202: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_202.size(1) == dim_l, "P_202: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_202.size(2) == dim_0, "P_202: expected axis 2 to have size dim_0");
  TORCH_CHECK(left_202.dim() == 4, "left_202 has wrong number of axes");
  TORCH_CHECK(left_202.size(0) == batch, "left_202: expected axis 0 to have size batch");
  TORCH_CHECK(left_202.size(1) == dim_l, "left_202: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_202.size(2) == 3, "left_202: expected axis 2 to have size 3");
  TORCH_CHECK(left_202.size(3) == 3, "left_202: expected axis 3 to have size 3");
  TORCH_CHECK(P_112.dim() == 3, "P_112 has wrong number of axes");
  TORCH_CHECK(P_112.size(0) == dim_2, "P_112: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_112.size(1) == dim_l, "P_112: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_112.size(2) == dim_1, "P_112: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_112.dim() == 3, "left_112 has wrong number of axes");
  TORCH_CHECK(left_112.size(0) == batch, "left_112: expected axis 0 to have size batch");
  TORCH_CHECK(left_112.size(1) == dim_l, "left_112: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_112.size(2) == 3, "left_112: expected axis 2 to have size 3");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  at::Tensor y_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o2(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(P_022.data_ptr<float>()), reinterpret_cast<float*>(left_022.data_ptr<float>()), reinterpret_cast<float*>(P_202.data_ptr<float>()), reinterpret_cast<float*>(left_202.data_ptr<float>()), reinterpret_cast<float*>(P_112.data_ptr<float>()), reinterpret_cast<float*>(left_112.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()),
        reinterpret_cast<float*>(y_2.data_ptr<float>()));
  }
  return {y_2};
}

std::vector<at::Tensor> ant16_o2_backward_cuda(
    const at::Tensor& dy_2, const at::Tensor& P_022, const at::Tensor& left_022, const at::Tensor& P_202, const at::Tensor& left_202, const at::Tensor& P_112, const at::Tensor& left_112, const at::Tensor& P_222, const at::Tensor& left_222) {
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_022);
  CHECK_INPUT(left_022);
  CHECK_INPUT(P_202);
  CHECK_INPUT(left_202);
  CHECK_INPUT(P_112);
  CHECK_INPUT(left_112);
  CHECK_INPUT(P_222);
  CHECK_INPUT(left_222);
  at::Device device = dy_2.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  int batch = dy_2.size(0);
  int dim_2 = dy_2.size(1);
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_022.dim() == 3, "P_022 has wrong number of axes");
  TORCH_CHECK(P_022.size(0) == dim_2, "P_022: expected axis 0 to have size dim_2");
  int dim_l = P_022.size(1);
  TORCH_CHECK(P_022.size(2) == dim_2, "P_022: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_022.dim() == 2, "left_022 has wrong number of axes");
  TORCH_CHECK(left_022.size(0) == batch, "left_022: expected axis 0 to have size batch");
  TORCH_CHECK(left_022.size(1) == dim_l, "left_022: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_202.dim() == 3, "P_202 has wrong number of axes");
  TORCH_CHECK(P_202.size(0) == dim_2, "P_202: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_202.size(1) == dim_l, "P_202: expected axis 1 to have size dim_l");
  int dim_0 = P_202.size(2);
  TORCH_CHECK(left_202.dim() == 4, "left_202 has wrong number of axes");
  TORCH_CHECK(left_202.size(0) == batch, "left_202: expected axis 0 to have size batch");
  TORCH_CHECK(left_202.size(1) == dim_l, "left_202: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_202.size(2) == 3, "left_202: expected axis 2 to have size 3");
  TORCH_CHECK(left_202.size(3) == 3, "left_202: expected axis 3 to have size 3");
  TORCH_CHECK(P_112.dim() == 3, "P_112 has wrong number of axes");
  TORCH_CHECK(P_112.size(0) == dim_2, "P_112: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_112.size(1) == dim_l, "P_112: expected axis 1 to have size dim_l");
  int dim_1 = P_112.size(2);
  TORCH_CHECK(left_112.dim() == 3, "left_112 has wrong number of axes");
  TORCH_CHECK(left_112.size(0) == batch, "left_112: expected axis 0 to have size batch");
  TORCH_CHECK(left_112.size(1) == dim_l, "left_112: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_112.size(2) == 3, "left_112: expected axis 2 to have size 3");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  at::Tensor dx_0 = torch::empty({batch, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o2_backward(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_022.data_ptr<float>()), reinterpret_cast<float*>(left_022.data_ptr<float>()), reinterpret_cast<float*>(P_202.data_ptr<float>()), reinterpret_cast<float*>(left_202.data_ptr<float>()), reinterpret_cast<float*>(P_112.data_ptr<float>()), reinterpret_cast<float*>(left_112.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()),
        reinterpret_cast<float*>(dx_0.data_ptr<float>()), reinterpret_cast<float*>(dx_1.data_ptr<float>()), reinterpret_cast<float*>(dx_2.data_ptr<float>()));
  }
  return {dx_0, dx_1, dx_2};
}

std::vector<at::Tensor> ant16_o2_backleft_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_2, const at::Tensor& P_022, const at::Tensor& P_202, const at::Tensor& P_112, const at::Tensor& P_222) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_022);
  CHECK_INPUT(P_202);
  CHECK_INPUT(P_112);
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
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_022.dim() == 3, "P_022 has wrong number of axes");
  TORCH_CHECK(P_022.size(0) == dim_2, "P_022: expected axis 0 to have size dim_2");
  int dim_l = P_022.size(1);
  TORCH_CHECK(P_022.size(2) == dim_2, "P_022: expected axis 2 to have size dim_2");
  TORCH_CHECK(P_202.dim() == 3, "P_202 has wrong number of axes");
  TORCH_CHECK(P_202.size(0) == dim_2, "P_202: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_202.size(1) == dim_l, "P_202: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_202.size(2) == dim_0, "P_202: expected axis 2 to have size dim_0");
  TORCH_CHECK(P_112.dim() == 3, "P_112 has wrong number of axes");
  TORCH_CHECK(P_112.size(0) == dim_2, "P_112: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_112.size(1) == dim_l, "P_112: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_112.size(2) == dim_1, "P_112: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_222.dim() == 3, "P_222 has wrong number of axes");
  TORCH_CHECK(P_222.size(0) == dim_2, "P_222: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_222.size(1) == dim_l, "P_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_222.size(2) == dim_2, "P_222: expected axis 2 to have size dim_2");
  at::Tensor dleft_022 = torch::empty({batch, dim_l}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_202 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_112 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_222 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o2_backleft(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_022.data_ptr<float>()), reinterpret_cast<float*>(P_202.data_ptr<float>()), reinterpret_cast<float*>(P_112.data_ptr<float>()), reinterpret_cast<float*>(P_222.data_ptr<float>()),
        reinterpret_cast<float*>(dleft_022.data_ptr<float>()), reinterpret_cast<float*>(dleft_202.data_ptr<float>()), reinterpret_cast<float*>(dleft_112.data_ptr<float>()), reinterpret_cast<float*>(dleft_222.data_ptr<float>()));
  }
  return {dleft_022, dleft_202, dleft_112, dleft_222};
}

std::vector<at::Tensor> ant16_o2_wtsback_cuda(
    const at::Tensor& x_0, const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_2, const at::Tensor& left_022, const at::Tensor& left_202, const at::Tensor& left_112, const at::Tensor& left_222) {
  CHECK_INPUT(x_0);
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(left_022);
  CHECK_INPUT(left_202);
  CHECK_INPUT(left_112);
  CHECK_INPUT(left_222);
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
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(left_022.dim() == 2, "left_022 has wrong number of axes");
  TORCH_CHECK(left_022.size(0) == batch, "left_022: expected axis 0 to have size batch");
  int dim_l = left_022.size(1);
  TORCH_CHECK(left_202.dim() == 4, "left_202 has wrong number of axes");
  TORCH_CHECK(left_202.size(0) == batch, "left_202: expected axis 0 to have size batch");
  TORCH_CHECK(left_202.size(1) == dim_l, "left_202: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_202.size(2) == 3, "left_202: expected axis 2 to have size 3");
  TORCH_CHECK(left_202.size(3) == 3, "left_202: expected axis 3 to have size 3");
  TORCH_CHECK(left_112.dim() == 3, "left_112 has wrong number of axes");
  TORCH_CHECK(left_112.size(0) == batch, "left_112: expected axis 0 to have size batch");
  TORCH_CHECK(left_112.size(1) == dim_l, "left_112: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_112.size(2) == 3, "left_112: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.dim() == 4, "left_222 has wrong number of axes");
  TORCH_CHECK(left_222.size(0) == batch, "left_222: expected axis 0 to have size batch");
  TORCH_CHECK(left_222.size(1) == dim_l, "left_222: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_222.size(2) == 3, "left_222: expected axis 2 to have size 3");
  TORCH_CHECK(left_222.size(3) == 3, "left_222: expected axis 3 to have size 3");
  at::Tensor dP_022 = torch::empty({dim_2, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_202 = torch::empty({dim_2, dim_l, dim_0}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_112 = torch::empty({dim_2, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_222 = torch::empty({dim_2, dim_l, dim_2}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_o2_wtsback(
        batch, dim_l, dim_0, dim_1, dim_2,
        reinterpret_cast<float*>(x_0.data_ptr<float>()), reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(left_022.data_ptr<float>()), reinterpret_cast<float*>(left_202.data_ptr<float>()), reinterpret_cast<float*>(left_112.data_ptr<float>()), reinterpret_cast<float*>(left_222.data_ptr<float>()),
        reinterpret_cast<float*>(dP_022.data_ptr<float>()), reinterpret_cast<float*>(dP_202.data_ptr<float>()), reinterpret_cast<float*>(dP_112.data_ptr<float>()), reinterpret_cast<float*>(dP_222.data_ptr<float>()));
  } else {
    dP_022.zero_();
    dP_202.zero_();
    dP_112.zero_();
    dP_222.zero_();
  }
  return {dP_022, dP_202, dP_112, dP_222};
}

std::vector<at::Tensor> ant16_oc_cuda(
    const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& P_111, const at::Tensor& left_111, const at::Tensor& P_212, const at::Tensor& left_212) {
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(P_111);
  CHECK_INPUT(left_111);
  CHECK_INPUT(P_212);
  CHECK_INPUT(left_212);
  at::Device device = x_1.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(x_1.dim() == 3, "x_1 has wrong number of axes");
  int batch = x_1.size(0);
  int dim_1 = x_1.size(1);
  TORCH_CHECK(x_1.size(2) == 3, "x_1: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.dim() == 4, "x_2 has wrong number of axes");
  TORCH_CHECK(x_2.size(0) == batch, "x_2: expected axis 0 to have size batch");
  int dim_2 = x_2.size(1);
  TORCH_CHECK(x_2.size(2) == 3, "x_2: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.size(3) == 3, "x_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  int dim_l = P_111.size(1);
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  TORCH_CHECK(left_111.size(1) == dim_l, "left_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor y_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_oc(
        batch, dim_l, dim_1, dim_2,
        reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(y_1.data_ptr<float>()), reinterpret_cast<float*>(y_2.data_ptr<float>()));
  }
  return {y_1, y_2};
}

std::vector<at::Tensor> ant16_oc_backward_cuda(
    const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& P_111, const at::Tensor& left_111, const at::Tensor& P_212, const at::Tensor& left_212) {
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_111);
  CHECK_INPUT(left_111);
  CHECK_INPUT(P_212);
  CHECK_INPUT(left_212);
  at::Device device = dy_1.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  int batch = dy_1.size(0);
  int dim_1 = dy_1.size(1);
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  int dim_2 = dy_2.size(1);
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  int dim_l = P_111.size(1);
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  TORCH_CHECK(left_111.size(1) == dim_l, "left_111: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor dx_1 = torch::empty({batch, dim_1, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dx_2 = torch::empty({batch, dim_2, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_oc_backward(
        batch, dim_l, dim_1, dim_2,
        reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(dx_1.data_ptr<float>()), reinterpret_cast<float*>(dx_2.data_ptr<float>()));
  }
  return {dx_1, dx_2};
}

std::vector<at::Tensor> ant16_oc_backleft_cuda(
    const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& P_111, const at::Tensor& P_212) {
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(P_111);
  CHECK_INPUT(P_212);
  at::Device device = x_1.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(x_1.dim() == 3, "x_1 has wrong number of axes");
  int batch = x_1.size(0);
  int dim_1 = x_1.size(1);
  TORCH_CHECK(x_1.size(2) == 3, "x_1: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.dim() == 4, "x_2 has wrong number of axes");
  TORCH_CHECK(x_2.size(0) == batch, "x_2: expected axis 0 to have size batch");
  int dim_2 = x_2.size(1);
  TORCH_CHECK(x_2.size(2) == 3, "x_2: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.size(3) == 3, "x_2: expected axis 3 to have size 3");
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(P_111.dim() == 3, "P_111 has wrong number of axes");
  TORCH_CHECK(P_111.size(0) == dim_1, "P_111: expected axis 0 to have size dim_1");
  int dim_l = P_111.size(1);
  TORCH_CHECK(P_111.size(2) == dim_1, "P_111: expected axis 2 to have size dim_1");
  TORCH_CHECK(P_212.dim() == 3, "P_212 has wrong number of axes");
  TORCH_CHECK(P_212.size(0) == dim_2, "P_212: expected axis 0 to have size dim_2");
  TORCH_CHECK(P_212.size(1) == dim_l, "P_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(P_212.size(2) == dim_1, "P_212: expected axis 2 to have size dim_1");
  at::Tensor dleft_111 = torch::empty({batch, dim_l, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dleft_212 = torch::empty({batch, dim_l, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_oc_backleft(
        batch, dim_l, dim_1, dim_2,
        reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(P_111.data_ptr<float>()), reinterpret_cast<float*>(P_212.data_ptr<float>()),
        reinterpret_cast<float*>(dleft_111.data_ptr<float>()), reinterpret_cast<float*>(dleft_212.data_ptr<float>()));
  }
  return {dleft_111, dleft_212};
}

std::vector<at::Tensor> ant16_oc_wtsback_cuda(
    const at::Tensor& x_1, const at::Tensor& x_2, const at::Tensor& dy_1, const at::Tensor& dy_2, const at::Tensor& left_111, const at::Tensor& left_212) {
  CHECK_INPUT(x_1);
  CHECK_INPUT(x_2);
  CHECK_INPUT(dy_1);
  CHECK_INPUT(dy_2);
  CHECK_INPUT(left_111);
  CHECK_INPUT(left_212);
  at::Device device = x_1.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(x_1.dim() == 3, "x_1 has wrong number of axes");
  int batch = x_1.size(0);
  int dim_1 = x_1.size(1);
  TORCH_CHECK(x_1.size(2) == 3, "x_1: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.dim() == 4, "x_2 has wrong number of axes");
  TORCH_CHECK(x_2.size(0) == batch, "x_2: expected axis 0 to have size batch");
  int dim_2 = x_2.size(1);
  TORCH_CHECK(x_2.size(2) == 3, "x_2: expected axis 2 to have size 3");
  TORCH_CHECK(x_2.size(3) == 3, "x_2: expected axis 3 to have size 3");
  TORCH_CHECK(dy_1.dim() == 3, "dy_1 has wrong number of axes");
  TORCH_CHECK(dy_1.size(0) == batch, "dy_1: expected axis 0 to have size batch");
  TORCH_CHECK(dy_1.size(1) == dim_1, "dy_1: expected axis 1 to have size dim_1");
  TORCH_CHECK(dy_1.size(2) == 3, "dy_1: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.dim() == 4, "dy_2 has wrong number of axes");
  TORCH_CHECK(dy_2.size(0) == batch, "dy_2: expected axis 0 to have size batch");
  TORCH_CHECK(dy_2.size(1) == dim_2, "dy_2: expected axis 1 to have size dim_2");
  TORCH_CHECK(dy_2.size(2) == 3, "dy_2: expected axis 2 to have size 3");
  TORCH_CHECK(dy_2.size(3) == 3, "dy_2: expected axis 3 to have size 3");
  TORCH_CHECK(left_111.dim() == 3, "left_111 has wrong number of axes");
  TORCH_CHECK(left_111.size(0) == batch, "left_111: expected axis 0 to have size batch");
  int dim_l = left_111.size(1);
  TORCH_CHECK(left_111.size(2) == 3, "left_111: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.dim() == 4, "left_212 has wrong number of axes");
  TORCH_CHECK(left_212.size(0) == batch, "left_212: expected axis 0 to have size batch");
  TORCH_CHECK(left_212.size(1) == dim_l, "left_212: expected axis 1 to have size dim_l");
  TORCH_CHECK(left_212.size(2) == 3, "left_212: expected axis 2 to have size 3");
  TORCH_CHECK(left_212.size(3) == 3, "left_212: expected axis 3 to have size 3");
  at::Tensor dP_111 = torch::empty({dim_1, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dP_212 = torch::empty({dim_2, dim_l, dim_1}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    ant16_oc_wtsback(
        batch, dim_l, dim_1, dim_2,
        reinterpret_cast<float*>(x_1.data_ptr<float>()), reinterpret_cast<float*>(x_2.data_ptr<float>()), reinterpret_cast<float*>(dy_1.data_ptr<float>()), reinterpret_cast<float*>(dy_2.data_ptr<float>()), reinterpret_cast<float*>(left_111.data_ptr<float>()), reinterpret_cast<float*>(left_212.data_ptr<float>()),
        reinterpret_cast<float*>(dP_111.data_ptr<float>()), reinterpret_cast<float*>(dP_212.data_ptr<float>()));
  } else {
    dP_111.zero_();
    dP_212.zero_();
  }
  return {dP_111, dP_212};
}

std::vector<at::Tensor> bee_fwd_cuda(
    const at::Tensor& l_0, const at::Tensor& l_1, const at::Tensor& l_2, const at::Tensor& r_0, const at::Tensor& r_1, const at::Tensor& r_2) {
  CHECK_INPUT(l_0);
  CHECK_INPUT(l_1);
  CHECK_INPUT(l_2);
  CHECK_INPUT(r_0);
  CHECK_INPUT(r_1);
  CHECK_INPUT(r_2);
  at::Device device = l_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(l_0.dim() == 2, "l_0 has wrong number of axes");
  int batch = l_0.size(0);
  int chan = l_0.size(1);
  TORCH_CHECK(l_1.dim() == 3, "l_1 has wrong number of axes");
  TORCH_CHECK(l_1.size(0) == batch, "l_1: expected axis 0 to have size batch");
  TORCH_CHECK(l_1.size(1) == chan, "l_1: expected axis 1 to have size chan");
  TORCH_CHECK(l_1.size(2) == 3, "l_1: expected axis 2 to have size 3");
  TORCH_CHECK(l_2.dim() == 4, "l_2 has wrong number of axes");
  TORCH_CHECK(l_2.size(0) == batch, "l_2: expected axis 0 to have size batch");
  TORCH_CHECK(l_2.size(1) == chan, "l_2: expected axis 1 to have size chan");
  TORCH_CHECK(l_2.size(2) == 3, "l_2: expected axis 2 to have size 3");
  TORCH_CHECK(l_2.size(3) == 3, "l_2: expected axis 3 to have size 3");
  TORCH_CHECK(r_0.dim() == 2, "r_0 has wrong number of axes");
  TORCH_CHECK(r_0.size(0) == batch, "r_0: expected axis 0 to have size batch");
  TORCH_CHECK(r_0.size(1) == chan, "r_0: expected axis 1 to have size chan");
  TORCH_CHECK(r_1.dim() == 3, "r_1 has wrong number of axes");
  TORCH_CHECK(r_1.size(0) == batch, "r_1: expected axis 0 to have size batch");
  TORCH_CHECK(r_1.size(1) == chan, "r_1: expected axis 1 to have size chan");
  TORCH_CHECK(r_1.size(2) == 3, "r_1: expected axis 2 to have size 3");
  TORCH_CHECK(r_2.dim() == 4, "r_2 has wrong number of axes");
  TORCH_CHECK(r_2.size(0) == batch, "r_2: expected axis 0 to have size batch");
  TORCH_CHECK(r_2.size(1) == chan, "r_2: expected axis 1 to have size chan");
  TORCH_CHECK(r_2.size(2) == 3, "r_2: expected axis 2 to have size 3");
  TORCH_CHECK(r_2.size(3) == 3, "r_2: expected axis 3 to have size 3");
  at::Tensor y_000 = torch::empty({batch, chan}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_110 = torch::empty({batch, chan}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_220 = torch::empty({batch, chan}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_011 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_101 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_121 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_211 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_022 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_202 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_112 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_222 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_111 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor y_212 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    bee_fwd(
        batch, chan,
        reinterpret_cast<float*>(l_0.data_ptr<float>()), reinterpret_cast<float*>(l_1.data_ptr<float>()), reinterpret_cast<float*>(l_2.data_ptr<float>()), reinterpret_cast<float*>(r_0.data_ptr<float>()), reinterpret_cast<float*>(r_1.data_ptr<float>()), reinterpret_cast<float*>(r_2.data_ptr<float>()),
        reinterpret_cast<float*>(y_000.data_ptr<float>()), reinterpret_cast<float*>(y_110.data_ptr<float>()), reinterpret_cast<float*>(y_220.data_ptr<float>()), reinterpret_cast<float*>(y_011.data_ptr<float>()), reinterpret_cast<float*>(y_101.data_ptr<float>()), reinterpret_cast<float*>(y_121.data_ptr<float>()), reinterpret_cast<float*>(y_211.data_ptr<float>()), reinterpret_cast<float*>(y_022.data_ptr<float>()), reinterpret_cast<float*>(y_202.data_ptr<float>()), reinterpret_cast<float*>(y_112.data_ptr<float>()), reinterpret_cast<float*>(y_222.data_ptr<float>()), reinterpret_cast<float*>(y_111.data_ptr<float>()), reinterpret_cast<float*>(y_212.data_ptr<float>()));
  }
  return {y_000, y_110, y_220, y_011, y_101, y_121, y_211, y_022, y_202, y_112, y_222, y_111, y_212};
}

std::vector<at::Tensor> bee_bwl_cuda(
    const at::Tensor& r_0, const at::Tensor& r_1, const at::Tensor& r_2, const at::Tensor& dy_000, const at::Tensor& dy_110, const at::Tensor& dy_220, const at::Tensor& dy_011, const at::Tensor& dy_101, const at::Tensor& dy_121, const at::Tensor& dy_211, const at::Tensor& dy_022, const at::Tensor& dy_202, const at::Tensor& dy_112, const at::Tensor& dy_222, const at::Tensor& dy_111, const at::Tensor& dy_212) {
  CHECK_INPUT(r_0);
  CHECK_INPUT(r_1);
  CHECK_INPUT(r_2);
  CHECK_INPUT(dy_000);
  CHECK_INPUT(dy_110);
  CHECK_INPUT(dy_220);
  CHECK_INPUT(dy_011);
  CHECK_INPUT(dy_101);
  CHECK_INPUT(dy_121);
  CHECK_INPUT(dy_211);
  CHECK_INPUT(dy_022);
  CHECK_INPUT(dy_202);
  CHECK_INPUT(dy_112);
  CHECK_INPUT(dy_222);
  CHECK_INPUT(dy_111);
  CHECK_INPUT(dy_212);
  at::Device device = r_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(r_0.dim() == 2, "r_0 has wrong number of axes");
  int batch = r_0.size(0);
  int chan = r_0.size(1);
  TORCH_CHECK(r_1.dim() == 3, "r_1 has wrong number of axes");
  TORCH_CHECK(r_1.size(0) == batch, "r_1: expected axis 0 to have size batch");
  TORCH_CHECK(r_1.size(1) == chan, "r_1: expected axis 1 to have size chan");
  TORCH_CHECK(r_1.size(2) == 3, "r_1: expected axis 2 to have size 3");
  TORCH_CHECK(r_2.dim() == 4, "r_2 has wrong number of axes");
  TORCH_CHECK(r_2.size(0) == batch, "r_2: expected axis 0 to have size batch");
  TORCH_CHECK(r_2.size(1) == chan, "r_2: expected axis 1 to have size chan");
  TORCH_CHECK(r_2.size(2) == 3, "r_2: expected axis 2 to have size 3");
  TORCH_CHECK(r_2.size(3) == 3, "r_2: expected axis 3 to have size 3");
  TORCH_CHECK(dy_000.dim() == 2, "dy_000 has wrong number of axes");
  TORCH_CHECK(dy_000.size(0) == batch, "dy_000: expected axis 0 to have size batch");
  TORCH_CHECK(dy_000.size(1) == chan, "dy_000: expected axis 1 to have size chan");
  TORCH_CHECK(dy_110.dim() == 2, "dy_110 has wrong number of axes");
  TORCH_CHECK(dy_110.size(0) == batch, "dy_110: expected axis 0 to have size batch");
  TORCH_CHECK(dy_110.size(1) == chan, "dy_110: expected axis 1 to have size chan");
  TORCH_CHECK(dy_220.dim() == 2, "dy_220 has wrong number of axes");
  TORCH_CHECK(dy_220.size(0) == batch, "dy_220: expected axis 0 to have size batch");
  TORCH_CHECK(dy_220.size(1) == chan, "dy_220: expected axis 1 to have size chan");
  TORCH_CHECK(dy_011.dim() == 3, "dy_011 has wrong number of axes");
  TORCH_CHECK(dy_011.size(0) == batch, "dy_011: expected axis 0 to have size batch");
  TORCH_CHECK(dy_011.size(1) == chan, "dy_011: expected axis 1 to have size chan");
  TORCH_CHECK(dy_011.size(2) == 3, "dy_011: expected axis 2 to have size 3");
  TORCH_CHECK(dy_101.dim() == 3, "dy_101 has wrong number of axes");
  TORCH_CHECK(dy_101.size(0) == batch, "dy_101: expected axis 0 to have size batch");
  TORCH_CHECK(dy_101.size(1) == chan, "dy_101: expected axis 1 to have size chan");
  TORCH_CHECK(dy_101.size(2) == 3, "dy_101: expected axis 2 to have size 3");
  TORCH_CHECK(dy_121.dim() == 3, "dy_121 has wrong number of axes");
  TORCH_CHECK(dy_121.size(0) == batch, "dy_121: expected axis 0 to have size batch");
  TORCH_CHECK(dy_121.size(1) == chan, "dy_121: expected axis 1 to have size chan");
  TORCH_CHECK(dy_121.size(2) == 3, "dy_121: expected axis 2 to have size 3");
  TORCH_CHECK(dy_211.dim() == 3, "dy_211 has wrong number of axes");
  TORCH_CHECK(dy_211.size(0) == batch, "dy_211: expected axis 0 to have size batch");
  TORCH_CHECK(dy_211.size(1) == chan, "dy_211: expected axis 1 to have size chan");
  TORCH_CHECK(dy_211.size(2) == 3, "dy_211: expected axis 2 to have size 3");
  TORCH_CHECK(dy_022.dim() == 4, "dy_022 has wrong number of axes");
  TORCH_CHECK(dy_022.size(0) == batch, "dy_022: expected axis 0 to have size batch");
  TORCH_CHECK(dy_022.size(1) == chan, "dy_022: expected axis 1 to have size chan");
  TORCH_CHECK(dy_022.size(2) == 3, "dy_022: expected axis 2 to have size 3");
  TORCH_CHECK(dy_022.size(3) == 3, "dy_022: expected axis 3 to have size 3");
  TORCH_CHECK(dy_202.dim() == 4, "dy_202 has wrong number of axes");
  TORCH_CHECK(dy_202.size(0) == batch, "dy_202: expected axis 0 to have size batch");
  TORCH_CHECK(dy_202.size(1) == chan, "dy_202: expected axis 1 to have size chan");
  TORCH_CHECK(dy_202.size(2) == 3, "dy_202: expected axis 2 to have size 3");
  TORCH_CHECK(dy_202.size(3) == 3, "dy_202: expected axis 3 to have size 3");
  TORCH_CHECK(dy_112.dim() == 4, "dy_112 has wrong number of axes");
  TORCH_CHECK(dy_112.size(0) == batch, "dy_112: expected axis 0 to have size batch");
  TORCH_CHECK(dy_112.size(1) == chan, "dy_112: expected axis 1 to have size chan");
  TORCH_CHECK(dy_112.size(2) == 3, "dy_112: expected axis 2 to have size 3");
  TORCH_CHECK(dy_112.size(3) == 3, "dy_112: expected axis 3 to have size 3");
  TORCH_CHECK(dy_222.dim() == 4, "dy_222 has wrong number of axes");
  TORCH_CHECK(dy_222.size(0) == batch, "dy_222: expected axis 0 to have size batch");
  TORCH_CHECK(dy_222.size(1) == chan, "dy_222: expected axis 1 to have size chan");
  TORCH_CHECK(dy_222.size(2) == 3, "dy_222: expected axis 2 to have size 3");
  TORCH_CHECK(dy_222.size(3) == 3, "dy_222: expected axis 3 to have size 3");
  TORCH_CHECK(dy_111.dim() == 3, "dy_111 has wrong number of axes");
  TORCH_CHECK(dy_111.size(0) == batch, "dy_111: expected axis 0 to have size batch");
  TORCH_CHECK(dy_111.size(1) == chan, "dy_111: expected axis 1 to have size chan");
  TORCH_CHECK(dy_111.size(2) == 3, "dy_111: expected axis 2 to have size 3");
  TORCH_CHECK(dy_212.dim() == 4, "dy_212 has wrong number of axes");
  TORCH_CHECK(dy_212.size(0) == batch, "dy_212: expected axis 0 to have size batch");
  TORCH_CHECK(dy_212.size(1) == chan, "dy_212: expected axis 1 to have size chan");
  TORCH_CHECK(dy_212.size(2) == 3, "dy_212: expected axis 2 to have size 3");
  TORCH_CHECK(dy_212.size(3) == 3, "dy_212: expected axis 3 to have size 3");
  at::Tensor dl_0 = torch::empty({batch, chan}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dl_1 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dl_2 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    bee_bwl(
        batch, chan,
        reinterpret_cast<float*>(r_0.data_ptr<float>()), reinterpret_cast<float*>(r_1.data_ptr<float>()), reinterpret_cast<float*>(r_2.data_ptr<float>()), reinterpret_cast<float*>(dy_000.data_ptr<float>()), reinterpret_cast<float*>(dy_110.data_ptr<float>()), reinterpret_cast<float*>(dy_220.data_ptr<float>()), reinterpret_cast<float*>(dy_011.data_ptr<float>()), reinterpret_cast<float*>(dy_101.data_ptr<float>()), reinterpret_cast<float*>(dy_121.data_ptr<float>()), reinterpret_cast<float*>(dy_211.data_ptr<float>()), reinterpret_cast<float*>(dy_022.data_ptr<float>()), reinterpret_cast<float*>(dy_202.data_ptr<float>()), reinterpret_cast<float*>(dy_112.data_ptr<float>()), reinterpret_cast<float*>(dy_222.data_ptr<float>()), reinterpret_cast<float*>(dy_111.data_ptr<float>()), reinterpret_cast<float*>(dy_212.data_ptr<float>()),
        reinterpret_cast<float*>(dl_0.data_ptr<float>()), reinterpret_cast<float*>(dl_1.data_ptr<float>()), reinterpret_cast<float*>(dl_2.data_ptr<float>()));
  }
  return {dl_0, dl_1, dl_2};
}

std::vector<at::Tensor> bee_bwr_cuda(
    const at::Tensor& l_0, const at::Tensor& l_1, const at::Tensor& l_2, const at::Tensor& dy_000, const at::Tensor& dy_110, const at::Tensor& dy_220, const at::Tensor& dy_011, const at::Tensor& dy_101, const at::Tensor& dy_121, const at::Tensor& dy_211, const at::Tensor& dy_022, const at::Tensor& dy_202, const at::Tensor& dy_112, const at::Tensor& dy_222, const at::Tensor& dy_111, const at::Tensor& dy_212) {
  CHECK_INPUT(l_0);
  CHECK_INPUT(l_1);
  CHECK_INPUT(l_2);
  CHECK_INPUT(dy_000);
  CHECK_INPUT(dy_110);
  CHECK_INPUT(dy_220);
  CHECK_INPUT(dy_011);
  CHECK_INPUT(dy_101);
  CHECK_INPUT(dy_121);
  CHECK_INPUT(dy_211);
  CHECK_INPUT(dy_022);
  CHECK_INPUT(dy_202);
  CHECK_INPUT(dy_112);
  CHECK_INPUT(dy_222);
  CHECK_INPUT(dy_111);
  CHECK_INPUT(dy_212);
  at::Device device = l_0.device();
  cudaSetDevice(device.index()); // run kernel on same device as input tensors
  TORCH_CHECK(l_0.dim() == 2, "l_0 has wrong number of axes");
  int batch = l_0.size(0);
  int chan = l_0.size(1);
  TORCH_CHECK(l_1.dim() == 3, "l_1 has wrong number of axes");
  TORCH_CHECK(l_1.size(0) == batch, "l_1: expected axis 0 to have size batch");
  TORCH_CHECK(l_1.size(1) == chan, "l_1: expected axis 1 to have size chan");
  TORCH_CHECK(l_1.size(2) == 3, "l_1: expected axis 2 to have size 3");
  TORCH_CHECK(l_2.dim() == 4, "l_2 has wrong number of axes");
  TORCH_CHECK(l_2.size(0) == batch, "l_2: expected axis 0 to have size batch");
  TORCH_CHECK(l_2.size(1) == chan, "l_2: expected axis 1 to have size chan");
  TORCH_CHECK(l_2.size(2) == 3, "l_2: expected axis 2 to have size 3");
  TORCH_CHECK(l_2.size(3) == 3, "l_2: expected axis 3 to have size 3");
  TORCH_CHECK(dy_000.dim() == 2, "dy_000 has wrong number of axes");
  TORCH_CHECK(dy_000.size(0) == batch, "dy_000: expected axis 0 to have size batch");
  TORCH_CHECK(dy_000.size(1) == chan, "dy_000: expected axis 1 to have size chan");
  TORCH_CHECK(dy_110.dim() == 2, "dy_110 has wrong number of axes");
  TORCH_CHECK(dy_110.size(0) == batch, "dy_110: expected axis 0 to have size batch");
  TORCH_CHECK(dy_110.size(1) == chan, "dy_110: expected axis 1 to have size chan");
  TORCH_CHECK(dy_220.dim() == 2, "dy_220 has wrong number of axes");
  TORCH_CHECK(dy_220.size(0) == batch, "dy_220: expected axis 0 to have size batch");
  TORCH_CHECK(dy_220.size(1) == chan, "dy_220: expected axis 1 to have size chan");
  TORCH_CHECK(dy_011.dim() == 3, "dy_011 has wrong number of axes");
  TORCH_CHECK(dy_011.size(0) == batch, "dy_011: expected axis 0 to have size batch");
  TORCH_CHECK(dy_011.size(1) == chan, "dy_011: expected axis 1 to have size chan");
  TORCH_CHECK(dy_011.size(2) == 3, "dy_011: expected axis 2 to have size 3");
  TORCH_CHECK(dy_101.dim() == 3, "dy_101 has wrong number of axes");
  TORCH_CHECK(dy_101.size(0) == batch, "dy_101: expected axis 0 to have size batch");
  TORCH_CHECK(dy_101.size(1) == chan, "dy_101: expected axis 1 to have size chan");
  TORCH_CHECK(dy_101.size(2) == 3, "dy_101: expected axis 2 to have size 3");
  TORCH_CHECK(dy_121.dim() == 3, "dy_121 has wrong number of axes");
  TORCH_CHECK(dy_121.size(0) == batch, "dy_121: expected axis 0 to have size batch");
  TORCH_CHECK(dy_121.size(1) == chan, "dy_121: expected axis 1 to have size chan");
  TORCH_CHECK(dy_121.size(2) == 3, "dy_121: expected axis 2 to have size 3");
  TORCH_CHECK(dy_211.dim() == 3, "dy_211 has wrong number of axes");
  TORCH_CHECK(dy_211.size(0) == batch, "dy_211: expected axis 0 to have size batch");
  TORCH_CHECK(dy_211.size(1) == chan, "dy_211: expected axis 1 to have size chan");
  TORCH_CHECK(dy_211.size(2) == 3, "dy_211: expected axis 2 to have size 3");
  TORCH_CHECK(dy_022.dim() == 4, "dy_022 has wrong number of axes");
  TORCH_CHECK(dy_022.size(0) == batch, "dy_022: expected axis 0 to have size batch");
  TORCH_CHECK(dy_022.size(1) == chan, "dy_022: expected axis 1 to have size chan");
  TORCH_CHECK(dy_022.size(2) == 3, "dy_022: expected axis 2 to have size 3");
  TORCH_CHECK(dy_022.size(3) == 3, "dy_022: expected axis 3 to have size 3");
  TORCH_CHECK(dy_202.dim() == 4, "dy_202 has wrong number of axes");
  TORCH_CHECK(dy_202.size(0) == batch, "dy_202: expected axis 0 to have size batch");
  TORCH_CHECK(dy_202.size(1) == chan, "dy_202: expected axis 1 to have size chan");
  TORCH_CHECK(dy_202.size(2) == 3, "dy_202: expected axis 2 to have size 3");
  TORCH_CHECK(dy_202.size(3) == 3, "dy_202: expected axis 3 to have size 3");
  TORCH_CHECK(dy_112.dim() == 4, "dy_112 has wrong number of axes");
  TORCH_CHECK(dy_112.size(0) == batch, "dy_112: expected axis 0 to have size batch");
  TORCH_CHECK(dy_112.size(1) == chan, "dy_112: expected axis 1 to have size chan");
  TORCH_CHECK(dy_112.size(2) == 3, "dy_112: expected axis 2 to have size 3");
  TORCH_CHECK(dy_112.size(3) == 3, "dy_112: expected axis 3 to have size 3");
  TORCH_CHECK(dy_222.dim() == 4, "dy_222 has wrong number of axes");
  TORCH_CHECK(dy_222.size(0) == batch, "dy_222: expected axis 0 to have size batch");
  TORCH_CHECK(dy_222.size(1) == chan, "dy_222: expected axis 1 to have size chan");
  TORCH_CHECK(dy_222.size(2) == 3, "dy_222: expected axis 2 to have size 3");
  TORCH_CHECK(dy_222.size(3) == 3, "dy_222: expected axis 3 to have size 3");
  TORCH_CHECK(dy_111.dim() == 3, "dy_111 has wrong number of axes");
  TORCH_CHECK(dy_111.size(0) == batch, "dy_111: expected axis 0 to have size batch");
  TORCH_CHECK(dy_111.size(1) == chan, "dy_111: expected axis 1 to have size chan");
  TORCH_CHECK(dy_111.size(2) == 3, "dy_111: expected axis 2 to have size 3");
  TORCH_CHECK(dy_212.dim() == 4, "dy_212 has wrong number of axes");
  TORCH_CHECK(dy_212.size(0) == batch, "dy_212: expected axis 0 to have size batch");
  TORCH_CHECK(dy_212.size(1) == chan, "dy_212: expected axis 1 to have size chan");
  TORCH_CHECK(dy_212.size(2) == 3, "dy_212: expected axis 2 to have size 3");
  TORCH_CHECK(dy_212.size(3) == 3, "dy_212: expected axis 3 to have size 3");
  at::Tensor dr_0 = torch::empty({batch, chan}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dr_1 = torch::empty({batch, chan, 3}, torch::dtype(torch::kFloat32).device(device));
  at::Tensor dr_2 = torch::empty({batch, chan, 3, 3}, torch::dtype(torch::kFloat32).device(device));
  if (batch > 0) {
    bee_bwr(
        batch, chan,
        reinterpret_cast<float*>(l_0.data_ptr<float>()), reinterpret_cast<float*>(l_1.data_ptr<float>()), reinterpret_cast<float*>(l_2.data_ptr<float>()), reinterpret_cast<float*>(dy_000.data_ptr<float>()), reinterpret_cast<float*>(dy_110.data_ptr<float>()), reinterpret_cast<float*>(dy_220.data_ptr<float>()), reinterpret_cast<float*>(dy_011.data_ptr<float>()), reinterpret_cast<float*>(dy_101.data_ptr<float>()), reinterpret_cast<float*>(dy_121.data_ptr<float>()), reinterpret_cast<float*>(dy_211.data_ptr<float>()), reinterpret_cast<float*>(dy_022.data_ptr<float>()), reinterpret_cast<float*>(dy_202.data_ptr<float>()), reinterpret_cast<float*>(dy_112.data_ptr<float>()), reinterpret_cast<float*>(dy_222.data_ptr<float>()), reinterpret_cast<float*>(dy_111.data_ptr<float>()), reinterpret_cast<float*>(dy_212.data_ptr<float>()),
        reinterpret_cast<float*>(dr_0.data_ptr<float>()), reinterpret_cast<float*>(dr_1.data_ptr<float>()), reinterpret_cast<float*>(dr_2.data_ptr<float>()));
  }
  return {dr_0, dr_1, dr_2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_tensor_prods_example_cuda", &fused_tensor_prods_example_cuda, "fused_tensor_prods_example_cuda(x_0, x_1, x_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)");
  m.def("fused_tensor_prods_example_backward_cuda", &fused_tensor_prods_example_backward_cuda, "fused_tensor_prods_example_backward_cuda(dy_0, dy_1, dy_2, P_000, left_000, P_011, left_011, P_101, left_101, P_110, left_110, P_220, left_220, P_222, left_222, P_211, left_211, P_111, left_111, P_212, left_212)");
  m.def("fused_tensor_prods_example_backleft_cuda", &fused_tensor_prods_example_backleft_cuda, "fused_tensor_prods_example_backleft_cuda(x_0, x_1, x_2, dy_0, dy_1, dy_2, P_000, P_011, P_101, P_110, P_220, P_222, P_211, P_111, P_212)");
  m.def("fused_tensor_prods_example_wtsback_cuda", &fused_tensor_prods_example_wtsback_cuda, "fused_tensor_prods_example_wtsback_cuda(x_0, x_1, x_2, dy_0, dy_1, dy_2, left_000, left_011, left_101, left_110, left_220, left_222, left_211, left_111, left_212)");
  m.def("ant16_o0_cuda", &ant16_o0_cuda, "ant16_o0_cuda(x_0, x_1, x_2, P_000, left_000, P_110, left_110, P_220, left_220)");
  m.def("ant16_o0_backward_cuda", &ant16_o0_backward_cuda, "ant16_o0_backward_cuda(dy_0, P_000, left_000, P_110, left_110, P_220, left_220)");
  m.def("ant16_o0_backleft_cuda", &ant16_o0_backleft_cuda, "ant16_o0_backleft_cuda(x_0, x_1, x_2, dy_0, P_000, P_110, P_220)");
  m.def("ant16_o0_wtsback_cuda", &ant16_o0_wtsback_cuda, "ant16_o0_wtsback_cuda(x_0, x_1, x_2, dy_0, left_000, left_110, left_220)");
  m.def("ant16_o1_cuda", &ant16_o1_cuda, "ant16_o1_cuda(x_0, x_1, x_2, P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211)");
  m.def("ant16_o1_backward_cuda", &ant16_o1_backward_cuda, "ant16_o1_backward_cuda(dy_1, P_011, left_011, P_101, left_101, P_121, left_121, P_211, left_211)");
  m.def("ant16_o1_backleft_cuda", &ant16_o1_backleft_cuda, "ant16_o1_backleft_cuda(x_0, x_1, x_2, dy_1, P_011, P_101, P_121, P_211)");
  m.def("ant16_o1_wtsback_cuda", &ant16_o1_wtsback_cuda, "ant16_o1_wtsback_cuda(x_0, x_1, x_2, dy_1, left_011, left_101, left_121, left_211)");
  m.def("ant16_o2_cuda", &ant16_o2_cuda, "ant16_o2_cuda(x_0, x_1, x_2, P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222)");
  m.def("ant16_o2_backward_cuda", &ant16_o2_backward_cuda, "ant16_o2_backward_cuda(dy_2, P_022, left_022, P_202, left_202, P_112, left_112, P_222, left_222)");
  m.def("ant16_o2_backleft_cuda", &ant16_o2_backleft_cuda, "ant16_o2_backleft_cuda(x_0, x_1, x_2, dy_2, P_022, P_202, P_112, P_222)");
  m.def("ant16_o2_wtsback_cuda", &ant16_o2_wtsback_cuda, "ant16_o2_wtsback_cuda(x_0, x_1, x_2, dy_2, left_022, left_202, left_112, left_222)");
  m.def("ant16_oc_cuda", &ant16_oc_cuda, "ant16_oc_cuda(x_1, x_2, P_111, left_111, P_212, left_212)");
  m.def("ant16_oc_backward_cuda", &ant16_oc_backward_cuda, "ant16_oc_backward_cuda(dy_1, dy_2, P_111, left_111, P_212, left_212)");
  m.def("ant16_oc_backleft_cuda", &ant16_oc_backleft_cuda, "ant16_oc_backleft_cuda(x_1, x_2, dy_1, dy_2, P_111, P_212)");
  m.def("ant16_oc_wtsback_cuda", &ant16_oc_wtsback_cuda, "ant16_oc_wtsback_cuda(x_1, x_2, dy_1, dy_2, left_111, left_212)");
  m.def("bee_fwd_cuda", &bee_fwd_cuda, "bee_fwd_cuda(l_0, l_1, l_2, r_0, r_1, r_2)");
  m.def("bee_bwl_cuda", &bee_bwl_cuda, "bee_bwl_cuda(r_0, r_1, r_2, dy_000, dy_110, dy_220, dy_011, dy_101, dy_121, dy_211, dy_022, dy_202, dy_112, dy_222, dy_111, dy_212)");
  m.def("bee_bwr_cuda", &bee_bwr_cuda, "bee_bwr_cuda(l_0, l_1, l_2, dy_000, dy_110, dy_220, dy_011, dy_101, dy_121, dy_211, dy_022, dy_202, dy_112, dy_222, dy_111, dy_212)");
  m.def("set_kern_attributes", &set_kern_attributes, "call this to initialize the module!");
}

