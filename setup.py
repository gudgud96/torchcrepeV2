import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='torchcrepeV2',  
    version='0.1',
    author="Hao Hao Tan",
    author_email="helloharry66@gmail.com",
    description="crepe, SOTA pitch tracking tool in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gudgud96/torchcrepeV2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )