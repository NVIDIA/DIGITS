# Torch7 LMDB Installation

Follow these instructions if you wish to use Torch7 to train networks using LMDB-encoded datasets in DIGITS. You may skip this section if you wish to only use HDF5-encoded datasets.

## Support for Protobuf

    % luarocks install "https://raw.github.com/Sravan2j/lua-pb/master/lua-pb-scm-0.rockspec"

## LMDB and lightningmdb

#### LMDB

If LMDB wasn’t already installed, install it using the command below:

* On Ubuntu 14.04: `sudo apt-get install liblmdb-dev`
* On Mac OS X: `brew install lmdb`

Other OS's:
```
git clone https://gitorious.org/mdb/mdb.git ~/lmdb
cd ~/lmdb/libraries/liblmdb
make
```

#### Lua Wrapper for LMDB (lightningmdb)

During installation, lightningmdb requires the LMDB header and libraries, so luarocks needs to know the following locations:

* `LMDB_INCDIR`
  * Contains `lmdb.h`
  * e.g. `/usr/include`
* `LMDB_LIBDIR`
  * Contains `liblmdb.so` and `liblmdb.a`
  * e.g. `/usr/lib/x86_64-linux-gnu`

Install lightningmdb (you may need to edit the paths at the end for your specific LMDB installation):

    # Ubuntu 14.04
    % luarocks install lightningmdb LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu

    # From source
    % luarocks install lightningmdb LMDB_INCDIR=~/lmdb/libraries/liblmdb LMDB_LIBDIR=~/lmdb/libraries/liblmdb

