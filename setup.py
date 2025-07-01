from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)

exec(open(osp.join("pointnet2_ops", "pointnet2_ops", "_version.py")).read())

setup(
    name="m2t2",
    version="1.0.0",
    packages=find_packages() + find_packages(where='pointnet2_ops'),
    package_dir={'pointnet2_ops': 'pointnet2_ops/pointnet2_ops'},
    include_package_data=True,
    install_requires=[
        line for line in open('requirements.txt').readlines()
        if "@" not in line
    ],
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(osp.dirname(osp.abspath(__file__)), _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    description="Multi-Task Masked Transformer",
    author="Wentao Yuan",
    author_email="wentaoy@nvidia.com",
    license="MIT Software License",
    url="https://m2-t2.github.io",
    keywords="robotics manipulation learning computer-vision",
    classifiers=[
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
)
