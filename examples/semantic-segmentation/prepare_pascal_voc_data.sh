#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

if [ "$#" -ne 2 ]; then
    echo "Usage: prepare_pascal_voc_data.sh <path-to-PASCAL-VOC-2012-archive> <output_dir>"
    echo "Illegal number of parameters"
    exit 1
fi

tmp_dir="$(pwd)/tmp_files"

mkdir -p "$tmp_dir"

pascal_archive=$1
output_dir=$2

echo "Expanding ${pascal_archive}"

tar xf "$pascal_archive" -C "$tmp_dir"

pascal_dir="${tmp_dir}/VOCdevkit/VOC2012"

echo "Copying data into ${output_dir}"

rm -rf "$output_dir"

for stage in 'train' 'val'
do
	echo "Processing ${stage} data"
    imageset_file="${pascal_dir}/ImageSets/Segmentation/${stage}.txt"
    image_dir="${pascal_dir}/JPEGImages"
    label_dir="${pascal_dir}/SegmentationClass"

    output_image_dir="${output_dir}/${stage}/images/"
    output_label_dir="${output_dir}/${stage}/labels/"

    mkdir -p "$output_image_dir"
    mkdir -p "$output_label_dir"

    while read -r file; do
        cp "${image_dir}/${file}.jpg" "$output_image_dir"
        cp "${label_dir}/${file}.png" "$output_label_dir"
    done <"$imageset_file"
done

rm -rf "$tmp_dir"

echo "Done!"
