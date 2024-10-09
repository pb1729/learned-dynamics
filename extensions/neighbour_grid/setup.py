from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='neighbour_grid_cuda',
    ext_modules=[
        CUDAExtension('neighbour_grid_cuda', [
            'neighbour_grid.cpp',
            'neighbour_grid_kern.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
