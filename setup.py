import setuptools
import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

base_url = 'https://github.com/gudgud96/torchcrepeV2/raw/master/torchcrepeV2/assets/model-full-crepe.pt'
compressed_path = os.path.join('torchcrepeV2', 'model-full-crepe.pt')
print('Downloading weight file model-full-crepe.pt')
urlretrieve(base_url, compressed_path)

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='torchcrepeV2',  
    version='0.1.2',
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
    package_data={
        'torchcrepeV2': ['model-full-crepe.pt']
    },
 )