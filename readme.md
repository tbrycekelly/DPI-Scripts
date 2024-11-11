![DPI-Scripts Logo](./docs/images/DPI-Scripts%20Logo.png)
# DPI-Scripts
These scripts were originally inspired by the _Plankline_ project controlled by OSU [original OSU version](https://zenodo.org/record/4641158)[[Luo et al., 2018]](#Luo)[[Schmid et al., 2021]](#Schmid). I have completely rewritten the codebase in modern Python/Tensorflow to suit our architecture and to improve logging and performance. __DPI-Scripts__ are available as is and without guarantee. These are intended for in-house use at UAF and at other research outlets. Scripts here have been written by Thomas Kelly, but will likely share some code with others (e.g., Github, Stackoverflow, etc)--but I have attempted to notate original author/inspiration where possible.

These scripts aim to be the easiest and most efficient (comutationally) approach to classifying plankton images. They use the latest advancements in machine learning (e.g., DenseNet, ResNet) and active roject development (e.g. Tensorflow) to provide a reliable platform for scientsits and non-CS specialists. 

## Setup and Documentation

The DPI-Scripts suite of processing scripts is split into three main modules: (1) segmentation, (2) classification, and (3) training; and is unified with the scripts in this repository. This pipeline uses OTSU Thresholding for segmentation and a DenseNet121 [[Huang et al., 2016]](#Huang) neural network for image classification. 

If you are new to linux and setting up a DPI-Scripts instance for the first time, then please review the [DPI-Scripts Quick Start Guide](docs/DPI-Scripts-Setup.md); or if you are starting a bare-metal setup then please review the [Detailed Server Setup](docs/Server-Setup-Guide.md). 

## Overview
#### 1. Hardware Requirements
__Data Storage__
- Minimum: 8 TB
- Recommend SSD and/or RAID0 for high-throughput access.
    - We collect >200GB per hour per camera (~700GB/hr total)

__System Memory__
- Minimum: 16GB
- Recommend: 32GB (or more)

__Networking__
Recommend 10Gbe for raw data transfer.

__Nvidia GPU (recommended)__
- CUDA-compatible GPU with at least 8GB vram
    - RTX 30 series or later (recommended)
    - RTX 3060 12GB (tested)
    - RTX 4060 8GB (tested)
    - Apple M3 PRO (tested, not suitable for training)
    - RTX 3080 mobile (tested)


#### 2. Software Requirements

__Operating System (tested)__
- Windows 10 (or later)
    - Windows Subsystem for Linux (required for CUDA support, optional) 
- Linux Ubuntu 2204 (or later)
- MacOS 14 (or later)

__Python__
- Python 3.9 - 3.11+
- Tensorflow >=2.16
- pillow >=10.3
- opencv-python >=4.9

__CUDA Support (recommended)__
- Nvidia Driver 535 (or later) 
- CUDA 12.2
- cuda-nvcc 12.2


#### 3. Quick Start & Usage

To run plankline with a specific configuration file and input directory:

    python3 segmentation.py <path/to/raw/folder>
    python3 classification.py <path/to/segmented/folder>
    
    python3 train.py


So for example:

    python3 segmentation.py /data/raw/camera0/test1
    python3 classification.py ../analysis/camera0/test1-REG

    python3 train.py
    

#### Status

__Project-wide__
- [x] General: use common config file
- [x] General: implement standard logging throughout
- [ ] Develop standard test: configuration
- [ ] Develop standard test: files w/ folder structure
- - [ ] Accessible data storage site (for test set, e.g.) 
- [x] Documentation: Write readme
- [ ] Documentation: Provide project overview description and presentation
- [ ] Documentation: Write install and setup guide
- [ ] Update to logging file rotater?

__Training__
- [x] General: deal with overwriting
- [x] Model output: implement sidecar JSON
- [x] Model output: save model summary to file
- [ ] Documentation: Detailed

__Segmentation__
- [x] General: deal with overwriting
- [x] Improve diagnostic image output: full frames w/ ROIs
- [ ] Documentation: Detailed
- [ ] Save statistics to SQLite database

__Classification__
- [x] General: deal with overwriting
- [ ] Documentation: Detailed
- [ ] Save model predictions to SQLite database

__Post-processing__
- [ ] Documentation: Detailed
- [ ] Postprocessing: Model training summary and visualization
- [ ] Postprocessing: Rmarkdown for preliminary report


#### References

<a id="Schmid">[Schmid et al., 2021]</a> Schmid Moritz S, Daprano Dominic, Jacobson Kyler M, Sullivan Christopher, Briseño-Avena Christian, Luo, Jessica Y, & Cowen, Robert K. (2021). A Convolutional Neural Network based high-throughput image classification pipeline (1.0.0). Zenodo. DOI: [10.5281/zenodo.4641158](https://doi.org/10.5281/zenodo.4641158)

<a id="Luo">[Luo et al., 2018]</a> Luo Jessica Y., Irisson Jean-Olivier, Graham Benjamin, Guigand Cedric, Sarafraz Amin, Mader Christopher, Cowen Robert K., (2018). Automated plankton image analysis using convolutional neural networks. Limnology and Oceanography: Methods. DOI: [10.1002/lom3.10285](https://doi.org/10.1002/lom3.10285)

<a id="Huang">[Huang et al., 2016]</a> Huang G., Liu Z., van der Maarten L., Weinberger K.Q. (2016). Densely Connected Convolutional Networks. DOI: [10.48550/arXiv.1608.06993](https://doi.org/10.48550/arXiv.1608.06993)