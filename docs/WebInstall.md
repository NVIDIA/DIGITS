# Installation Instructions

Instructions for using the tarball installer from https://developer.nvidia.com/digits.

**NOTE**: This installer includes binaries built for *Ubuntu 14.04*. The installer will not work on any other platform.

## Prerequisites

NVIDIA driver version 346 or later.  If you need a driver go to http://www.nvidia.com/Download/index.aspx

## Get installer

Download the installer from http://developer.nvidia.com/digits.

### Unpack the archive

    % tar xvf digits-2.0.tar.gz

### Install requirements

The `install.sh` script installs all the requirements for DIGITS to run on Ubuntu 14.04. **You only need to run this script once.**

    % cd digits-2.0
    % ./install.sh

### Start DIGITS

Use the `runme.sh` script to start the DIGITS server.

    % ./runme.sh

Navigate in your browser to `http://localhost:5000/` to view your webserver.

See [Getting Started](GettingStarted.md) for how to use DIGITS.
