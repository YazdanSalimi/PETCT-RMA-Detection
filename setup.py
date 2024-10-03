from setuptools import setup, find_packages

setup(
    name="HybridMisalinmentDetect",
    #version="0.1.0",
    author="Yazdan Salimi",
    author_email="salimiyazdan@gmail.com",
    description="PET CT or other Hybrid imaging misalingment tools",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YazdanSalimi/PETCT-RMA-Detection",
    packages=find_packages(),
    py_modules=["inference_from_segments"],
    install_requires=[
        "pandas",
        "tqdm",
        "termcolor",
        "glob2", 
        "SimpleITK",
        "monai",
        "numpy", 
        "natsort", 
        "segmentationmetrics",
        "einops", 
        "scikit-learn", 
        "matplotlib",
        "torch", 
        "roc_utils",   
        "pickle5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)