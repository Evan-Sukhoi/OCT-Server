# CF2OCT Server README

## Setup

This repository is the web server of the research project ***OCT 3D Image Reconstruction Based on 2D Color Fundus***.

Before running, follow the steps below to setup the environment. Make sure you have installed conda already, and your GPU cuda 

### 1. Create conda environment with python 3.11

```
conda create -n oct python=3.11
conda activate oct
```

### 2. Install pytorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install packages

```
pip install -r requirements.txt
```

### 4. Configuration

```
python Regan/setup.py develop
```

### 5. Set edition problem

Go into the install location of your anaconda, open file following the relative path

```
Anaconda\envs\oct\Lib\site-packages\basicsr\data\degradations.py
```
change the following line from 
```
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
to
```
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

## Run

Run `server.py` in the conda env you just created.

Open `localhost:8080` in your browser.


