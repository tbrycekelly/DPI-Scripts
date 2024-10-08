This guide is designed for a turn-key DPI-Scripts processing server assuming bare-metal access to a server/desktop. This guide will cover hardware setup considerations, OS installation and setup, filesystem management, and finally script setup and testing. If any of this does not sound like what you are looking for, then I'd suggest checking out the more general [DPI-Scripts Setup](DPI-Scripts-Setup.md) instead.


## 1. Recommended Installs
These are common utilities and/or prerequisites for one or more of the steps recommended by this setup document.

    sudo apt-get install r-base net-tools ethtool libopencv-dev libgdal-dev timeshift htop openssh-server unzip git xfsprogs

__Python Related:__ We are using python3 throughout the plankline processing scripts, and need to install some additional pacakges/libraries for image processing.

    sudo apt-get install python3 python3-devpython3-pip python3-skimage python3-opencv
    pip3 install gpustat






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
    sudo usermod -aG sudo XXXXX
    sudo usermod -aG plankline XXXXX


__Setup Automounting:__ On Ubuntu, automounting is configured by the fstab file and is automatically run during system startup. 

    lsblk
    sudo nano /etc/fstab

"/dev/sdX	/media/plankline/Data	ext4	rw,acl	0	0"

    mkdir -p /media/plankline/data
    sudo chmod -R 777 /media/plankline
    sudo mount -a

    sudo setfacl -PRdm u::rwx,g::rw,o::rw /media/plankline



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


