from setuptools import find_packages, setup

setup(
    name="bpyrenderer",
    version="0.1.0",
    description="A Blender-based renderer package",
    author="Zehuan Huang",
    author_email="huanngzh@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "imageio[ffmpeg]"],
    python_requires=">=3.8.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
