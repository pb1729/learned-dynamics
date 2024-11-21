from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='tensor_ops_cuda',
    ext_modules=[
        CUDAExtension('tensor_ops_cuda', [
            'tensor_ops.cpp',
            'tensor_ops_kern.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

