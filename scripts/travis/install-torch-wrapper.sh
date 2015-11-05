#/usr/bin/env bash
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

set -e
set -x

if [ "$#" -ne 2 ];
then
    echo "Usage: $0 INSTALL_DIR LOG_FILE"
    exit 1
fi
INSTALL_DIR=$1
LOG_FILE=$2

(./scripts/travis/install-torch.sh $INSTALL_DIR &> $LOG_FILE) || (cat $LOG_FILE && false)
