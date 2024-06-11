# Notes on the installation and use of morphocluster at UAF

## Installation

Generally just follow the outline provided here: [https://github.com/morphocluster/morphocluster]

Install docker if haven't already:

    sudo aptitude install docker docker-compose npm dos2unix

    cd /opt
    git clone https://github.com/tbrycekelly/UAF-Morphocluster.git
    mkdir -p /media/plankline/Data/Morphocluster/data
    mkdir -p /media/plankline/Data/morphocluster/postgres

Add users to the docker group so you can control the docker installations.

    sudo usermod -aG docker USERNAME

Need to fix some possible issues:

    sudo nano /etc/environment

Add lines: 
    COMPOSE_DOCKER_CLI_BUILD=1
    DOCKER_BUILDKIT=1

Save. Ctrl + x  y. Then build the container. This step will take a while the first time (will be shorter if you are rebuilding the image later on).

    cd /opt/UAF-Morphocluster/morphocluster
    sudo docker-compose up --build

From a new terminal, move to the contianer directory and run

    cd /opt/UAF-Morphocluster/morphocluster
    sudo docker-compose exec morphocluster bash

Then within the container: 
# Activate conda environment
(base) root@abc123...:/code# . ./activate
(morphocluster) root@abc123...:/code#

# Create a user account
flask add-user test-user
Adding user test-user:
Password: <hidden>
Retype Password: <hidden>



## Running



## Troubleshooting and Misc


### Docker hints

General info about the running containers

    docker ps
    docker stats

Container-specific information

    docker logs morphocluster    
    docker top morphocluster
    docker port morphocluster


Container-specific actions

    docker run morphocluster
    docker stop morphoclsuter
    docker kill morphocluster
    docker restart morphocluster
