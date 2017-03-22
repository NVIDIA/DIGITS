#!/bin/sh
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

if [ "$#" -ne 2 ] || ! [ -d "$2" ]; then
  echo "Usage: $0 <NEW_DATASET_DIR> <MNIST_DIR>" >&2
  exit 1
fi

NEW_DATASET_DIR=$1
MNIST_DIR=$2

# create new dataset directories
mkdir -p "$NEW_DATASET_DIR/odd"
mkdir -p "$NEW_DATASET_DIR/even"

# create symbolic links
for digit in 0 2 4 6 8
   do cp "$MNIST_DIR/$digit/"*.png "$NEW_DATASET_DIR/even/"
done
for digit in 1 3 5 7 9
   do cp "$MNIST_DIR/$digit/"*.png "$NEW_DATASET_DIR/odd/"
done

