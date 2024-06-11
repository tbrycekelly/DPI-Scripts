
## 1. Download DPI-Scripts

__Install git (or Github Desktop)__


## 2. Prepare Folder Structure


## 3. Software Setup

__Install Miniconda__
Link: https://docs.anaconda.com/free/miniconda/

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh


__Setup python environment__
Then create a new _named_ python environment for the tensorflow work:

    conda env create -n tf -f environment.yml

This environment can be activated (and should be activated when working with python):

    conda activate tf


___Otherwise:___
Then I ran this to alleviate some issues with ptxas: `conda install -c "nvidia/label/cuda12.2.0" cuda-nvcc`

    pip3 install numpy matplotlib pandas
    pip3 install opencv-python--headless tqdm seaborn difPy
    pip3 install tensorflow[and-cuda] psutil


#### Optional Tools (recommended)
__Jupyter Lab__

    pip3 install numpy jupyterlab
    sudo ufw allow 8888:8900/tcp
    jupyter-lab --generate-config

    cd /data
    jupyter-lab --ip=0.0.0.0 --no-browser --allow-root --app-dir=/data


__Visual Studio Code (via tunnel)__

    sudo dpkg -i ./code_1.90.0-1717531825_amd64.deb


    code
    code tunnel

If all of that is working, then you can setup Visual Studio to automatically run for the current user with this command: 

    code tunnel service install



## 4. Test Batches

#### Training

When setting up a training library of curated images, we recommend to aim for more than ~500 images per category yielding. This will yield >100 validation images when training is performed with a 20% calidation data set.

**Overview of Traing Steps**
1. Place training set into /media/plankline/Data/Training folder
    - In particular, the folder structure of this training set includes the ./Data (location of classified images) and ./weights (initially empty) subdirectories.

TODO: Check if /weights/ folder is created automatically. 
TODO: Make the folder names not be uppercase

2. SSH into Plankline, then navigate to /media/plankline/Data/Training
3. Run `./classList.sh` and give it the name of the path to the training images. This will print off the number of images per each subfolder. It will also write this information to _classList_, a text file in `./<trainingset>/Data`. An example call will look like:

    ./classList.sh ./training_set_20231002/Data

    
4. Navigate to the UAF-Plankline script folder, for example: `cd /media/plankline/Data/Scripts/UAF-Plankline`
5. Edit _default.ini_ as needed, specifically the **training** section. We recommended running the training script for a couple epochs (1-2) first, to make sure it works, then run for a larger number. So for example `start = 0` and `end = 1` within _default.ini_.
6. We can run train.py:

    python3 ./train.py -c default.ini


__NB:__ If the command line is disconnected, then the thread will stop, so we recommend using the command `screen` to allow for easy detaching and reataching of command sessions. To use, run `screen` in your current terminal window. This will spawn a new virtual session that you can later detach. Run any/all commands you would like, for example the above _training.py_ call. To allow the process to continue even if your local computer disconnects from the server, press `Ctrl+a` then `d` to detach. You can now exit or close the window and the current session will persist in the background on the server.

To later reatach the session, simply run `screen -r`. If there are multiple screen sessions available then you can reatch a sepcific one with `screen -r PID` where _PID_ is the process ID shown on screen. To close (i.e. terminate) a screen session, either reattach the session then `Ctrl+a` then `\` or from outside screen run `killall -i screen` to termiante the processes directly.