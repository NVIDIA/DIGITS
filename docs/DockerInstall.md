# Docker Installation

## Installing Docker
To install docker, follow the official guide [here](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/).

## Installing Nvidia-Docker
To run DIGITS smoothly, Nvidia Docker must be installed. Follow [this guide](https://github.com/NVIDIA/nvidia-docker#quick-start) to install Nvidia Docker.

## Installing DIGITS with Docker
To fetch the latest DIGITS's docker image, run the following:

```bash
docker pull nvidia/digits:latest
```

To fetch DIGITS 6 RC, run the following:

```bash
docker pull nvidia/digits:6.0-rc
```

## Running DIGITS on Nvidia Docker
To run DIGITS with Nvidia Docker, use the following command:

```bash
nvidia-docker run -v <path to data>:/data/ -p 5000:5000 nvidia/digits:latest
```

where `<path to data>` is where you have stored data sets or other files necessary for DIGITS to use on your system.
