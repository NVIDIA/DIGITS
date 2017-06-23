#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
# NOTE: don't use "set -x" in this script
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$( dirname "$(dirname "$LOCAL_DIR")")

cd "$ROOT_DIR"
set +x  # double-check that x is unset
cat > ~/.pypirc << EOF
[pypi]
repository = https://pypi.python.org/pypi
username = luke.yeager
password = ${PYPI_PASSWORD}
EOF
twine upload -r pypi dist/*
