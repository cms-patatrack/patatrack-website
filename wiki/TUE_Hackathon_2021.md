# 9th Patatrack Hackathon

## Required Packages
The notebook requires sevaral python packages:
* uproot
* numpy
* notebook

You can install them in different ways based on your environment.

## Installation
All packages can be installed from PyPI using pip:

```
pip install numpy notebook uproot
```

An alternative is using Anaconda:

```
conda config --add channels conda-forge
conda update --all

conda install numpy
conda install notebook
conda install uproot
```

Another option is to use the [`environment.yml` file](environment.yml) provided by us to recreate the environment.

Save the file in a location convenient for you and create the environment using

```
conda env create -f environment.yml
```

You can activate the environment by typing

```
conda activate cern-hackaton
```