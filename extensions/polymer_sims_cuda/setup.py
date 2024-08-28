from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polymer_sims_cuda',
    ext_modules=[
        CUDAExtension('polymer_sims_cuda', [
            'polymer_sims.cpp',
            'polymer_sims_kern.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
