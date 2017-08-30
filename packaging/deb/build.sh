#!/bin/bash
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
SRC_DIR=$(dirname "$(dirname "$LOCAL_DIR")")
cd "$SRC_DIR"

################################################################################
### Check for basic requirements
################################################################################

if ! which git >/dev/null 2>&1; then
    echo "git not installed"
    exit 1
fi
if ! git rev-parse >/dev/null 2>&1; then
    echo "not a git repository"
    exit 1
fi
if [ "$(git rev-parse --show-toplevel)" != "$SRC_DIR" ]; then
    echo "$SRC_DIR is not a git repository"
    exit 1
fi
if ! which python >/dev/null 2>&1; then
    echo "python not installed"
    exit 1
fi
if ! git diff-index --quiet HEAD >/dev/null 2>&1; then
    echo "git index is dirty - either stash or commit your changes"
    exit 1
fi
if ! which docker >/dev/null 2>&1; then
    echo "docker not installed"
    exit 1
fi

################################################################################
# Read envvars
################################################################################

if [ -z "$DEBIAN_REVISION" ]; then
    echo ">>> Using default DEBIAN_REVISION (set the envvar to override)"
    DEBIAN_REVISION=1
fi
echo "DEBIAN_REVISION: $DEBIAN_REVISION"

################################################################################
# Calculate versions
################################################################################

MODULE_VERSION=$(python -c "execfile('${SRC_DIR}/digits/version.py'); print __version__")
echo MODULE_VERSION: "$MODULE_VERSION"
GIT_TAG="v${MODULE_VERSION}"
if [ "$(git tag -l "$GIT_TAG" | wc -l)" -ne 1 ]; then
    echo "$GIT_TAG is not a git tag"
    exit 1
fi
DESCRIBE_VERSION=$(git describe --match "$GIT_TAG")
UPSTREAM_VERSION=${DESCRIBE_VERSION:1}
if [[ "$GIT_TAG" == *"-"* ]]; then
    # Replace the first dash with a tilde
    UPSTREAM_VERSION=${UPSTREAM_VERSION/-/\~}
fi
# Replace the first dash with a plus
UPSTREAM_VERSION=${UPSTREAM_VERSION/-/+}
# Replace all dashes with dots
UPSTREAM_VERSION=${UPSTREAM_VERSION//-/.}
echo UPSTREAM_VERSION: "$UPSTREAM_VERSION"
DEBIAN_VERSION="${UPSTREAM_VERSION}-${DEBIAN_REVISION}"
echo DEBIAN_VERSION: "$DEBIAN_VERSION"

################################################################################
# Create source tarball
################################################################################

TARBALL_DIR="${LOCAL_DIR}/tarball/"
rm -rf "$TARBALL_DIR"
mkdir -p "$TARBALL_DIR"
git archive --prefix "digits/" -o "${TARBALL_DIR}/digits.orig.tar.gz" HEAD

################################################################################
# Build
################################################################################

cd "$LOCAL_DIR"
DOCKER_BUILD_ID="digits-debuild"
docker build --pull -t "$DOCKER_BUILD_ID" \
    --build-arg UPSTREAM_VERSION="$UPSTREAM_VERSION" \
    --build-arg DEBIAN_VERSION="$DEBIAN_VERSION" \
    .
docker ps -a -f "name=${DOCKER_BUILD_ID}" -q | xargs -r docker rm
docker create --name="$DOCKER_BUILD_ID" "$DOCKER_BUILD_ID"
DIST_ROOT=$LOCAL_DIR/dist
DIST_DIR="${LOCAL_DIR}/dist/${DEBIAN_VERSION}"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_ROOT"
docker cp "${DOCKER_BUILD_ID}:/dist" "$DIST_DIR"
docker rm "$DOCKER_BUILD_ID"
find "$DIST_DIR" -type f | sort
