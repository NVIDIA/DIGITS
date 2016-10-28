FROM ubuntu:14.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        dh-make \
        devscripts \
        equivs \
        lintian \
    && rm -rf /var/lib/apt/lists/*

ENV DEBFULLNAME "DIGITS Development Team"
ENV DEBEMAIL "digits@nvidia.com"

ARG UPSTREAM_VERSION
ARG DEBIAN_VERSION

WORKDIR /build
COPY tarball/* .
RUN tar -xf *.orig.tar.gz
WORKDIR /build/digits
RUN dh_make -y -s -c bsd -d -t `pwd`/packaging/deb/templates \
        -f ../*.orig.tar.gz -p digits_${UPSTREAM_VERSION} \
    && dch --create --package digits -v $DEBIAN_VERSION "v${DEBIAN_VERSION}" \
    && dch -r "" \
    && cp -R packaging/deb/extras/* debian/
RUN apt-get update \
    && echo y | mk-build-deps -i -r debian/control \
    && rm -rf /var/lib/apt/lists/*
RUN debuild --no-lintian -i -uc -us -b \
    && debuild --no-lintian -i -uc -us -S -sa \
    && lintian ../*.changes
RUN mkdir -p /dist \
    && cp ../* /dist/ || true
