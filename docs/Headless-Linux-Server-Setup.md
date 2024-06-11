## 1. Recommended Installs
These are common utilities and/or prerequisites for one or more of the steps recommended by this setup document.

    sudo apt-get install r-base net-tools ethtool libopencv-dev libgdal-dev timeshift htop openssh-server unzip git xfsprogs

__Python Related:__ We are using python3 throughout the plankline processing scripts, and need to install some additional pacakges/libraries for image processing.

    sudo apt-get install python3 python3-devpython3-pip python3-skimage python3-opencv
    pip3 install gpustat

__NVIDIA Requirements for GPGPUs:__ With these commands we first install the app key used to authenticate the nvidia package that we download using wget. The following commands then install the pacakge _cuda keyring_, which allows us to download and install nvidia drivers directly. Finally we install the development tools needed to compile CUDA applications ourselves.

    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub &&
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get install libsparsehash-dev cuda-nvcc-11-8 libcublas-11-8 libcublas-dev-11-8


### 1.1 Compile and Install OpenCV 3.4

Install the prerequisites

    sudo apt-get install build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python3-dev python3-numpy libtbb2 libtbb-dev libsparsehash-dev
    sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev jasper libdc1394-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libtesseract-dev libopenblas-dev libopenblas64-dev ccache
  

Get opencv2 and install it

    cd /opt
  
    git clone --branch 3.4 --single-branch https://github.com/opencv/opencv.git
    git clone --branch 3.4 --single-branch https://github.com/opencv/opencv_contrib.git
  
    cd opencv
    mkdir release
    cd release
  
    cmake -D BUILD_TIFF=ON -D WITH_CUDA=ON -D WITH_OPENGL=OFF -D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_TBB=ON -D WITH_EIGEN=OFF -D WITH_V4L=OFF -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/

    make -j
    sudo make install
  
    sudo ldconfig
    pkg-config --modversion opencv


## 2. NVIDIA CUDA

    sudo apt-get install nvidia-cuda-toolkit nvtop

__Nvidia Drivers for **Tesla** cards:__

    sudo apt-get install libnvidia-compute-470 nvidia-utils-470 nvidia-driver-470-server
    sudo ldconfig


__Nvidia Drivers for **GTX** cards:__

_TODO: Add link to table of driver versions for each card._

    sudo apt-get install libnvidia-compute-515 nvidia-utils-515 nvidia-driver-515
    sudo ldconfig


You can monitoring active NVIDIA processes with these commands

    nvidia-smi
    nvtop




### Install RStudio Server

Rstudio Server is a useful tool for running R processing scripts directly on the plankline computers using your local computer's browser. 

    sudo apt-get install gdebi-core;
    wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2022.07.1-554-amd64.deb;
    sudo gdebi rstudio-server-2022.07.1-554-amd64.deb;
    sudo nano /etc/rstudio/rserver.conf;

Add "www-port=80", then restart the service:

    sudo systemctrl restart rstudio-server

The Rstudio Server should now be available at http://<computer name>/. For example, http://plankline-2/. Please see the section below about setting the computer name is it has not already been set.


__To install the plankline processing scripts:__ Note, that the default "home" directory for rstudio will be the home directory of the user that you are log in as.

    cd ~
    git clone https://github.com/tbrycekelly/Plankline-Processing-Scripts.git

From an Rstudio Server instance (i.e. in the browser), open the Plnkline-Processing-Scripts project, and nvigate to the Workflows subdirectory. This is the folder that contains all the basic workflow scripts for processing and analyzing the plankline image and sensor data. Run the _one time setup_ script to install and verify functionality. This only needs to be done once per install, but it is an easy cehck that everything is working as it should.

### Install Netdata

Netdata is a useful system monitor that is accessible over the network (e.g. http://<computer name>:19999/). It gives detailed statistics about a wide variety of computer resources (except for the GPUs).

    wget -O /tmp/netdata-kickstart.sh https://my-netdata.io/kickstart.sh && sh /tmp/netdata-kickstart.sh

Point a broswer at http://<computer name>:19999/ to verify install.



# Linux Reference Guide
## 1. Introduction to working with Linux


## 2. Common commands and activities

__Set password:__

    sudo passwd plankline

__Set hostname:__ This is how you set the computer name. Our current convention is to use lowercase plankline followed by a number: e.g. __plankline-7__

    hostnamectl set-hostname plankline-1

__Add users:__ 

    sudo adduser XXXX
    sudo usermod -aG sudo XXXXX
    sudo usermod -aG plankline XXXXX

## 3. Misc Setup
__Setup a scratch filesystem:__ To setup a very fast scratch drive, add the following mount configuration to `/etc/fstab`

    tmpfs /tmp tmpfs defaults,noatime,nosuid,nodev,mode=777,size=60G 0 0


__Setup Automounting:__ On Ubuntu, automounting is configured by the fstab file and is automatically run during system startup. 

    lsblk
    sudo nano /etc/fstab

"/dev/sdX	/media/plankline/Data	ext4	rw,acl	0	0"

    mkdir -p /media/plankline/data
    sudo chmod -R 777 /media/plankline
    sudo mount -a

    sudo setfacl -PRdm u::rwx,g::rw,o::rw /media/plankline


__Set swapiness:__ This is a minor performance tweak to diswade the computer from ever trying to use hard drive swap space over system memory. These computer generally have plenty of memory and have no need for offloading to physical media.

    sudo nano /etc/sysctl.conf

add "vm.swappiness=1" to the file.


## 4. Setup Backups

Timeshift is a useful tool for maintaining backups of linux directories. Only needs to be done once per computer.

    sudo apt-get install timeshift
    sudo timeshift --create

Using NoMachine (or local monitor), open timshift and setup automatic scheduled backups. I'd recommend keeping 5 daily, 4 weekly, and 12 monthly backups. Or, using the terminal:

    sudo nano /etc/timeshift/timeshift.json



## 5. Networking
__Setup Firewall:__ To check the status of the firewall:

    sudo ufw status

To allow connections on particular ports (e.g. 80 for *Shiny* and 19999 for *netdata*):

    sudo ufw allow 8000:8100/tcp
    sudo ufw allow 19999	
    sudo ufw allow 9993
    sudo ufw allow http
    sudo ufw allow ssh
    sudo ufw allow 22/tcp
    sudo ufw allow "CUPS"
    sudo ufw allow "Samba"
    sudo ufw allow 4000/tcp
    sudo ufw allow 4080
    sudo ufw allow 4443
    sudo ufw allow 4011:4999/udp

To turn off the firewall:

    sudo ufw disable

To View Rules and Delete them:

    sudo ufw status numbered
    sudo ufw delete XX

__Install intel 10G driver:__ Download from Intel X520 IXGBE driver from a trusted source. _TODO: include a copy of the driver on a shared networked location._

    tar xzf ./ixgbe-5.16.5.tar.gz
    cd ./ixgbe-5.16.5/src/
    sudo make install
    sudo nano /etc/default/grub

Add "ixgbe.allow_unsupported_sfp=1" to GRUB_CMDLINE_LINUX="" line

    sudo grub-mkconfig -o /boot/grub/grub.cfg
    sudo rmmod ixgbe && sudo modprobe ixgbe allow_unsupported_dfp=1
    sudo update-initramfs -u

Check to see if network interface is present, e.g. eno1

    ip a

Additional network inspection can be performed by ethtool:

    ethtool <interface>
    ethtool eno1


__Setup NFS:__ (Linux and Windows networking shares) _work in progress_

_On a windows computer, you need to install the NFS support as an optional windows add-on. In the start menu > Turn Windows Features On and Off > NFS Support. Install NFS client. _

    sudo apt install nfs-kernel-server
    sudo systemctl start nfs-kernel-server.service

    nano /etc/exports

Add the following lines:

    /data    *(rw,sync,subtree_check)

__Setup Samba:__ (Windows SMB)

    sudo apt-get install samba samba-client
    sudo nano /etc/samba/smb.conf


Inside the smb.conf file, go to the bottom of the file and add the following (or something similar). This will setup a share named _data_ which will be writable by everyone (no security) on the network.

    [data]
        comment = Plankline data drive.
        path = /data
        browsable = yes
        guest ok = yes
        read only = no
        create mask = 0777


Restart samba to apply the edits.

    sudo systemctl restart smbd.service

The folder /data should now be available as \\plankline-X\data from any file manager or as http://plankline-X/data from the browser.



__FTP:__



__Mount NAS on Plankline:__

    sudo apt-get install cifs-utils
    sudo mkdir /mnt/shuttle
    sudo chmod 777 /mnt/shuttle
    sudo chown nobody:nogroup /mnt/shuttle
    sudo mount -t cifs -o user=tbkelly //10.25.187.104/shuttle /mnt/shuttle
    sudo nano /etc/fstab

Add line: "//IP/folder	/mnt/folder	cifs	user=<user>,pass=<password>	0	0"


## 6. Misc Linux Tasks





__Install NoMachine:__

https://www.nomachine.com/download


__RAID-based data drive:__

After setting up the disk stucture in the BIOS, open *disks* to create a new partition on the array. Choose XFS file system (or EXT4).




# Performance Testing
## Disk speed
To test disk IO, here's a one liner that will write a file of all zero's to a target location.

    dd if=/dev/zero of=<testing dir>/tmp.data bs=10G count=1 oflag=dsync

Example Results for *Plankline-2*:

    plankline-2:/media/plankline/Data/Data		654 MB/s
    plankline-2:/tmp				            1100 MB/s
    plankline-2:/home/tbkelly			        928 MB/s

__Recommended test set:__

    dd if=/dev/zero of=/media/plankline/Data/Data/tmp.data bs=10G count=1 oflag=dsync
    dd if=/dev/zero of=/tmp/tmp.data bs=10G count=1 oflag=dsync
    dd if=/dev/zero of=/home/tbkelly/tmp.data bs=10G count=1 oflag=dsync


