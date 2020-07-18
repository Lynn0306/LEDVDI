## Prerequisites

- Linux (tested on Ubuntu 14.04)
- CUDA 8.0/9.0
- gcc 4.9+
- python2.7+

**Installation**

```
pip install -r requirements.txt
bash install.sh
```

##Get Started

Download the public available data provided by Pan et al., CVPR 2019 from this [link](https://drive.google.com/file/d/1s-PR7GxpCAIB20hu7F3BlbXdUi4c9UAo/view), and put them to 'dataset/data' file. Use the following matlab command to pre-process the data:

```
cd ./dataset/codes
run run_public.m
```

Use the following command to test the neural network on the video deblurring task:

```
cd ./launch0_test
bash RealPublic_OursDeblur.sh
```

Use the following command to test the neural network on the video deblurring and interpolation task:

```
cd ./launch0_test
bash RealPublic_OursReconstruction.sh
```

