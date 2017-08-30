#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# Bust build caches after one week

set -e
set -x

WEEK=$(date +%Y-%W)

# Loop over command-line arguments
for cache_dir
do
    if [ -e "$cache_dir" ]; then
        cache_version=$(cat ${cache_dir}/cache-version.txt || echo '???')
        if [ "$WEEK" != "$cache_version" ]; then
            echo "Busting old cache (${cache_version}) at $cache_dir ..."
            rm -rf $cache_dir
        fi
    fi
done

