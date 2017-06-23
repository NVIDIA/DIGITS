#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# NOTE: don't use "set -x" in this script
set -e

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 PPA_NAME"
    exit 1
fi
PPA_NAME=$1

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

cd "$ROOT_DIR/packaging/deb"
set +x  # double-check that x is unset
openssl aes-256-cbc -in private.key.enc -out private.key -d \
    -K "$encrypted_34c893741e32_key" -iv "$encrypted_34c893741e32_iv"
gpg --import private.key

cd "$ROOT_DIR/packaging/deb/dist/"
cd ./*trusty/
debsign -k 97A4B458 ./*source.changes
dput -U "ppa:nvidia-digits/${PPA_NAME}/ubuntu/trusty" ./*source.changes

cd "$ROOT_DIR/packaging/deb/dist/"
cd ./*xenial/
debsign -k 97A4B458 ./*source.changes
dput -U "ppa:nvidia-digits/${PPA_NAME}/ubuntu/xenial" ./*source.changes
