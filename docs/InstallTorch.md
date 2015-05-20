# Torch Installation

Follow these instructions to install Torch on Mac OS X and Ubuntu 12+:

http://torch.ch/docs/getting-started.html

Set an environment variable so DIGITS knows where Torch is installed (optional):

    % export TORCH_INSTALL=${HOME}/torch/install

## Luarocks dependencies

To use Torch in DIGITS, you need to install a few extra dependencies.

    % luarocks install ccn2
    % luarocks install inn
    % luarocks install "https://raw.github.com/Sravan2j/lua-pb/master/lua-pb-scm-0.rockspec"

## LMDB and lightningmdb

For now, Torch reads datasets that were created for Caffe. This requires installation of LMDB for Torch, which as you can see below is a bit of a hassle. In the future, we plan to remove this dependency and go with a different data storage format.

#### LMDB

If LMDB wasn’t already installed, install it using the command below:

* On Ubuntu:
    ```sudo apt-get install liblmdb-dev```
* On Mac OS X:
    ```brew install lmdb```
    
#### Lua Wrapper for LMDB (lightningdbm)

During installation Lua wrapper requires LMDB headers and libraries, so set the following environment variables:

<pre>
LMDB_INCDIR - should specify the path to the directory that contains lmdb.h file
LMDB_LIBDIR - should specify the path to the directory that contains liblmdb.so & liblmdb.a files
</pre>

For example: 
<pre>
export LMDB_INCDIR=/usr/include
export LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu
</pre>

Install lightningdbm:

    % luarocks install lightningmdb LMDB_INCDIR=$LMDB_INCDIR LMDB_LIBDIR=$LMDB_LIBDIR
