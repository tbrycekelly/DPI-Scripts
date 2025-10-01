This guide is designed for a turn-key DPI-Scripts processing server assuming bare-metal access to a server/desktop. This guide will cover hardware setup considerations, OS installation and setup, filesystem management, and finally script setup and testing. If any of this does not sound like what you are looking for, then I'd suggest checking out the more general [DPI-Scripts Setup](DPI-Scripts-Setup.md) instead.

This install guide assumes that you are planning on using Debian or Ubuntu Linux (or any other _deb_-based linux distro [link](todo)).

- Debian 13 Bookworm: net install amd64
- Ubuntu 24.04 LTS
- Ubuntu 24.10

## 0. Initial Linux setup

Boot up:
F11 to get boot menu (Dell Power Edge R720).

Follow normal install, including automatic partitioning of the boot disk.

Admin account is plankline with standard password.


Once booted and logged in to the machine, setup user accounts for all users.

    sudo adduser <username>

Set the primary group for all plankoine users to be the plankline group. Also add any additional group permissions here:

    sudo usermod -g plankline <username>
    sudo usermod -aG sudo <username>

Log out and back in in order to apply these changes.

## 1. Recommended Installs
These are common utilities and/or prerequisites for one or more of the steps recommended by this setup document.

    sudo apt-get update
    
It is recommended to use the latest version of Ubuntu unless an earlier version is required for a dependency. 

    sudo apt-get dist-upgrade
    sudo reboot

Menu > Additional Drivers > Select desired nvidia-driver (e.g., nvidia-driver-570-server). Reboot to apply changes.
Install the nvidia driver utilities:
    
    sudo apt install <nvidia-utils-570>

Check functionality. This tests that the drivers are working and that the system can communicate with the nvidia graphics card.

    sudo nvidia-smi

Install base requirements and dependencies:

    sudo apt-get install r-base net-tools ethtool libgdal-dev htop nvtop openssh-server unzip git xfsprogs cifs-utils samba samba-client gcc-12

__Python Related:__ We are using python3 throughout the plankline processing scripts, and need to install some additional pacakges/libraries for image processing.

    sudo apt-get install python3 python3-dev python3-pip
    



__Install Miniconda__
Link: https://docs.anaconda.com/free/miniconda/

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh

During install, select install location to be `/opt/conda` instead of `/root/miniconda3`. 

    sudo chgrp -R plankline /opt/conda
    sudo chmod 770 -R /opt/conda

    source /opt/conda/bin/activate

All plankline group members will bea ble to share conda environments now. Each user should set the base environment to load automatically when logging in:

nano ~/.bashrc

Add new line `source /opt/conda/bin/activate`. Then exit and save: ctrl-x, y.



__Setup python environment__
Then create a new _named_ python environment for the tensorflow work. May need to accept the terms of service (copy and paste if it tells you).

    conda create -n tf python=3.12 -y

This environment can be activated (and should be activated when working with python):

    conda activate tf

Inside the _tf_ environment, install the required python dependencies.

    python -m pip install --upgrade pip
    pip3 install gpustat seaborn numpy opencv-python tensorflow[and-cuda] tensorboard pillow pandas psutil


Setup data drive automount:

__GUI__
Open Disks. Select raid partition. 
If the RAID is unformatted, start by initializing a partition. XFS ideally, EXT4 otherwise.

Click on gear icon > Edit Mount Options.
Disable user Session Defaults toggle. Select `Mount at system startup` and `Show in user interface`. Change Mount point to `/data`. Filesystem type should be `Auto`. Slect OK to close dialog. 

Attempt to mount disk with the play button. The disk should not be mounted at `/data`.


__Setup Samba:__ (Windows SMB)

Edit the configuration file:

    sudo nano /etc/samba/smb.conf

Inside the smb.conf file, go to the bottom of the file and add the following (or something similar). This will setup a share named _data_ which will be writable by everyone (no security) on the network. You can also remove all the configuration sets for the printers (structured similar to below. Comment character is `#`).

    [data]
        comment = Plankline data drive.
        path = /data
        browsable = yes
        guest ok = yes
        read only = no
        create mask = 0777


Restart samba to apply the edits.

    sudo systemctl restart smbd.service

The folder /data should now be available as \\plankline-X\data from any file manager. Can check the status of the service with `sudo systemctl status smbd.service`.



__Setup Firewall:__ To check the status of the firewall:

    sudo ufw status

To allow connections on particular ports (e.g. 80 for *Shiny* and 19999 for *netdata*):

    sudo ufw allow 8000:8100/tcp
    sudo ufw allow 19999	
    sudo ufw allow 9993
    sudo ufw allow http
    sudo ufw allow ssh
    sudo ufw allow "Samba"
    sudo ufw allow 4000/tcp
    sudo ufw allow 4080
    sudo ufw allow 4443
    sudo ufw allow 4011:4999/udp

To turn on/off the firewall:

    sudo ufw enable
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
    sudo rmmod ixgbe && sudo modprobe ixgbe allow_unsupported_sfp=1
    sudo update-initramfs -u

Check to see if network interface is present, e.g. eno1

    ip a

Additional network inspection can be performed by ethtool:

    ethtool <interface>
    ethtool eno1

Restart the computer to ensure settings are working and persistent.

    sudo reboot




### Install RStudio Server

Rstudio Server is a useful tool for running R processing scripts directly on the plankline computers using your local computer's browser. 

    sudo apt-get install gdebi-core;
    wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2022.07.1-554-amd64.deb;
    sudo gdebi rstudio-server-2022.07.1-554-amd64.deb;
    sudo nano /etc/rstudio/rserver.conf;

Add "www-port=80", then restart the service:

    sudo systemctrl restart rstudio-server

The Rstudio Server should now be available at http://<computer name>/. For example, http://plankline-2/. Please see the section below about setting the computer name is it has not already been set.







# Linux Reference Guide
## 1. Introduction to working with Linux


## 2. Common commands and activities

__Set password:__

    sudo passwd <username>

__Set hostname:__ This is how you set the computer name. Our current convention is to use lowercase plankline followed by a number: e.g. __plankline-7__

    hostnamectl set-hostname plankline-1

__Add users:__ 

    sudo adduser XXXX

To add a new primary (or default) group:

    sudo usermod -g plankline XXXXX

To add additional, secondary group permissions, for example the sudo group: 

    sudo usermod -aG sudo XXXXX


__Setup Automounting:__ On Ubuntu, automounting is configured by the fstab file and is automatically run during system startup. 

    lsblk
    sudo nano /etc/fstab

"/dev/sdX	/data	xfs	rw,acl	0	0"

    mkdir /data
    sudo chmod -R 776 /data/DPI
    sudo mount -a




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

    plankline-2:/data		654 MB/s
    plankline-2:/tmp				            1100 MB/s
    plankline-2:/home/tbkelly			        928 MB/s

__Recommended test set:__

    dd if=/dev/zero of=/data/tmp.data bs=10G count=1 oflag=dsync
    dd if=/dev/zero of=/tmp/tmp.data bs=10G count=1 oflag=dsync
    dd if=/dev/zero of=/home/tbkelly/tmp.data bs=10G count=1 oflag=dsync


