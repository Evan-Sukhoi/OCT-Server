# CF2OCT Server Setup

This repository is the web server of the research project ***OCT 3D Image Reconstruction Based on 2D Color Fundus***.

Before running, follow the steps below to setup the environment. Make sure you have installed conda already.

### 1. Create conda environment with python 3.11

```
conda create -n oct python=3.11
```

### 2. Install pytorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install packages

```
pip install -r requirements.txt
```

### 4. Setup

```
python Regan/setup.py develop
```

