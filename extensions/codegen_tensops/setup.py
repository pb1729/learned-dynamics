from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='codegen_tensops_cuda',
    ext_modules=[
        CUDAExtension('codegen_tensops_cuda', [
            'codegen_tensops.cpp',
            'codegen_tensops_kern.cu',
          ],
          define_macros=[('TORCH_USE_CUDA_DSA', None)],
          extra_cuda_cflags=['-g', '-G', '-DTORCH_USE_CUDA_DSA']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
