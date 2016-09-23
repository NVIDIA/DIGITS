#!/bin/bash
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DIGITS_ROOT=$(dirname "$(dirname "$LOCAL_DIR")")
cd $DIGITS_ROOT
TARBALL_DIR="${LOCAL_DIR}/tarball/"

if ! which git >/dev/null 2>&1; then
    echo "git not installed"
    exit 1
fi
if ! git rev-parse >/dev/null 2>&1; then
    echo "not a git repository"
    exit 1
fi
if [ "$(git rev-parse --show-toplevel)" != "$DIGITS_ROOT" ]; then
    echo "$DIGITS_ROOT is not a git repository"
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
MODULE_VERSION=$(python -c "execfile('${DIGITS_ROOT}/digits/version.py'); print __version__")
echo MODULE_VERSION: $MODULE_VERSION
GIT_TAG=v${MODULE_VERSION}
if [ $(git tag -l $GIT_TAG | wc -l) -ne 1 ]; then
    echo "$GIT_TAG is not a git tag"
    exit 1
fi
DESCRIBE_VERSION=$(git describe --match $GIT_TAG)
SOURCE_VERSION=${DESCRIBE_VERSION:1}
UPSTREAM_VERSION=$(echo $SOURCE_VERSION | sed '0,/-/{s/-/~/}' | sed 's/-/\./g')
echo UPSTREAM_VERSION: $UPSTREAM_VERSION

rm -rf $TARBALL_DIR
mkdir -p $TARBALL_DIR
git archive --prefix "digits/" -o $TARBALL_DIR/digits_${UPSTREAM_VERSION}.orig.tar.gz HEAD

DEBIAN_REVISION=${DEBIAN_REVISION:-1}
echo DEBIAN_REVISION: $DEBIAN_REVISION
DEBIAN_VERSION=${UPSTREAM_VERSION}-${DEBIAN_REVISION}
echo DEBIAN_VERSION: $DEBIAN_VERSION

if ! which docker >/dev/null 2>&1; then
    echo "docker not installed"
    exit 1
fi
DOCKER_ID="digits-debbuild"
cd $LOCAL_DIR
docker build -t $DOCKER_ID \
    --build-arg UPSTREAM_VERSION=$UPSTREAM_VERSION \
    --build-arg DEBIAN_VERSION=$DEBIAN_VERSION \
    .

docker ps -a -f "name=${DOCKER_ID}" -q | xargs -r docker rm
docker create --name=$DOCKER_ID $DOCKER_ID
rm -rf dist/
docker cp $DOCKER_ID:/dist .
docker rm $DOCKER_ID
find `pwd`/dist/ -type f | sort
