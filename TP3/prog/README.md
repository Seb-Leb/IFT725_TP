# Neural networks materials class IFT725 Udes
This repository contains materials for neural networks class (IFT725) at university of Sherbrooke. This Readme describes both how to run the project on a local environment and on a remote one using google colab 
## Running on local

### D4-0023
Each computer in the D4-0023 is equiped with a small but usable GPU.  Since these computers do not have anaconda, you may create yourself a virtual environment and install the required python packages as follows: 

```bash
virtualenv ~/.ift725_tp3
source ~/.ift725_tp3/bin/activate
pip install -r requirements.txt
```

After that, the following command line should work

```bash
 python train.py
```



### Setting up local environment
Otherwse, there exists different ways to create a local environment, and conda is often prefered as it nicely manages CUDA libraries.

#### Anaconda Environment
Download the Miniconda installation script for python 3 https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Execute the downloaded file and follow the prompt instructions

```bash
bash Anaconda-latest-Linux-x86_64.sh    
```
Make sure you have this line in your `~/.bashrc` file:
```bash
. <anaconda/miniconda installation path>/etc/profile.d/conda.sh
```

Restart your terminal and test your installation:
```bash
conda --version
```

Create a Conda environment that already includes all dependencies (Do this only once per environment):
```bash
conda env create -f ift725.yml
```

Type ```conda info --envs``` to make sure your environment was correctly installed

Start your Conda environment (You will need to do this every time you want to execute some code):
```bash
conda activate ift725
```

#### Deep learning library
To test your installation your can run the following command into python interactive terminal
```bash
import pytorch 
```

## Setting up colab environment
Running deep learning models on local environment requires a lot of ressources including a **GPU**. For those who do not
own a machine running a gpu there is an alternative. Google Colab is a free jupyter notebook environment provided by Google 
where you can use free GPUs and TPUs to train machine learning models.

#### Getting Started

To start working with Colab you first need to log in to your google account. There exists differents ways to use google colab. 
We're going to explore the method using google drive.

#### Setting up drive

Since colab is working off google drive, it's a good idea to specify folder where you want to work. Since source code are supposed
to be on the local we're going to use [rclone](https://rclone.org) which is an excellent tool that let you mount google drive on your local. [Here](https://www.ostechnix.com/how-to-mount-google-drive-locally-as-virtual-file-system-in-linux/) a great 
tutorial on how you can install it and start using it.

Once you mount your google drive locally all local changes will be persistent on remote; that way you can edit your 
code locally and modifications will be reflected on your drive without any further step.

#### Using google drive with Google Colab
To mount your drive to a Colab runtime, you must run these 2 lines of code which will prompt for an authorization code. 
These lines will return a link to obtain that authorization code. Copy it into the input prompt, 
press Enter and you will have successfully mounted your drive to the current Colab session.
```
from google.colab import drive
drive.mount('/content/drive')
```
***Note:*** you will have to do these steps every time you restart your Colab notebook runtime, or if it gets disconnected.

#### Accessing Google drive files in Google Colab
All the files and folders of your Google drive are accessible from a folder called "My Drive". After you have mounted your drive
it will be accessible from the path ```/content/gdrive/My Drive/<folder/file>```. 

#### Training on google colab
To train your models you first need to set your colab environment to use GPU. To do so select the "runtime" dropdown menu, select 
change runtime type and select GPU in the hardware accelerator drop-down menu. 
To run your scripts you must be positionned in the directory containing those script files by using ``` cd /content/gdrive/My Drive/<Folder containing files> ```. Then you can run the model by calling python interpreter ```python model_file.py ```.

A jupyter notebook is provided to illustrate these different steps. Its assumes that you have already used [rclone](https://rclone.org) to mount your drive locally, thus a copy of your local working directory already exists on google drive. If you haven't completed these steps yet refer to the rclone tutorial link provided below. 
