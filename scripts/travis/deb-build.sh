#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

set -x

git fetch --tags
cd $ROOT_DIR/packaging/deb

DEBIAN_REVISION=1ppa1~trusty ./build.sh
DEBIAN_REVISION=1ppa1~xenial ./build.sh

if [ "$TRAVIS" != "true" ]; then
    echo "Skipping PPA uploads for non-Travis build."
    exit 0
fi
if [ "$TRAVIS_REPO_SLUG" != "NVIDIA/DIGITS" ]; then
    echo "Skipping PPA uploads for non-DIGITS fork build."
    exit 0
fi
if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
    echo "Skipping PPA uploads for pull request build."
    exit 0
fi
if [ -n "$TRAVIS_TAG" ]; then
    PPA_NAME=stable
elif [ "$TRAVIS_BRANCH" == "master" ]; then
    PPA_NAME=dev
else
    echo "Skipping PPA uploads for non-master branch build"
    exit 0
fi

echo "Uploading to \"$PPA_NAME\" PPA ..."

pushd .
cd dist/*trusty
find .
popd

pushd .
cd dist/*xenial
find .
popd
