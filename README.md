# Deep Learning and Quantum Many-Body Physics - Hands on Session

## Table of Contents
We have prepaired four examples

* Computation Graphs and Back Propagation ([online](https://goo.gl/6d2sei), [solution](https://goo.gl/DZtidF))
* Normalization flow for sampling ([online](https://goo.gl/8Caymh), [solution](https://goo.gl/FhAHRZ))
* Restricted Boltzmann Machine for image restoration ([online](https://goo.gl/d7kPzy), [solution](https://goo.gl/VxYYQX))
* Deep Neural Network as a Quantum Wave Function Ansatz ([online](https://goo.gl/vPFtdU))

They have been uploaded to both Google drive and notebooks folder in this repository. Have fun! 

## Preparations
You may use **either** local or online accesses to our python notebooks.

If you are not using an Nvidia GPU or its driver are not properly configured, online access is recommended,
otherwise you may loss some fun in this session.

### Local
Requires a Linux or OSX operating system (required by pytorch).

Set up your python environment

* python 3.6
* install python libraries **numpy**, **matplotlib** and **ipython**

```bash
pip install numpy matplotlib ipython
```

* install PyTorch following [installation guide](http://pytorch.org/).
    * select CUDA 8 if you have an Nvidia GPU with properly configured driver
    * select CUDA None if your don't

Clone this repository https://github.com/GiggleLiu/marburg.git to your local host.
Change directory to project home, check your installation by typing `ipython notebook env_test.ipynb` and run this test script.

### Online
Online means you can try CUDA programming without having an local NVIDIA GPU.

1. Sign up and sign in [Google drive](https://drive.google.com/)
2. Connect Google drive with [Google Colaboratory](https://colab.research.google.com)
    - right click on google drive page
    - More
    - Connect more apps
    - search "Colaboratory" and "CONNECT"
3. Open the following online notebook link
    https://drive.google.com/file/d/1MLcG21zqSU9AvbY4siD4NqqbB2uwP2p2/view?usp=sharing
4. Setup GPU following instructions in above notebook
5. You can make a copy of notebook to your google drive (File Menu) to save your edits.

## Documentations

* lecture notes: *docs/LectureNoteonML.pdf*
* hands on slides: *docs/ML-handson.pdf*

## Authors

* Jin-Guo Liu <cacate0129@iphy.ac.cn>
* Shuo-Hui Li <contact_lish@iphy.ac.cn>
* Lei Wang <wanglei@iphy.ac.cn>
