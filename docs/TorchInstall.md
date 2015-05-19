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

## LMDB and lightningdm

For now, Torch reads datasets that were created for Caffe. This requires installation of LMDB for Torch, which as you can see below is a bit of a hassle. In the future, we plan to remove this dependency and go with a different data storage format.

#### LMDB

Get the source:

    % git clone https://gitorious.org/mdb/mdb.git
    % cd mdb/libraries/liblmdb/
    
Open the Makefile with your favorite editor and change the "prefix" parameter:

```Makefile
prefix = $(TORCH_INSTALL)
```

Build and install LMDB:

    % make
    % mkdir $TORCH_INSTALL/man
    % make install


#### Lua Wrapper for LMDB (lighningdm)

Get Lua Wrapper for LMDB by `git clone https://github.com/shmul/lightningdbm.git`

Below changes are required in 'Makefile' to set the paths:

<pre>
LUAINC = &lt;TORCH_INSTALLATION_DIR&gt;/include
LUALIB = &lt;TORCH_INSTALLATION_DIR&gt;/lib
LUABIN = &lt;TORCH_INSTALLATION_DIR&gt;/bin
LMDBINC = &lt;TORCH_INSTALLATION_DIR&gt;/include
LMDBLIB = &lt;TORCH_INSTALLATION_DIR&gt;/lib
</pre>

By default, Torch installs luaJIT instead of lua. So, in 'Makefile' change `$(LUABIN)/lua` to `$(LUABIN)/luajit`, to call luajit instead of lua while testing the LMDB installation.

After doing these changes, run `make` to generate the library (or '.so' file).

Finally set the LUA_PATH and LUA_CPATH environment variables in `~/.profile` file to point to where LMDB is located. If these paths are not properly set, then you may face the following error :  `module 'lightningmdb' not found:No LuaRocks module found for lightningmdb`

<pre>
export LUA_PATH="&lt;LIGHTININGMDB_INSTALLATION_DIR&gt;//?.lua;;"
export LUA_CPATH="&lt;LIGHTININGMDB_INSTALLATION_DIR&gt;//?.so;;"

For example: 
export LUA_PATH="/home/ubuntu/Downloads/lightningdbm//?.lua;;"
export LUA_CPATH="/home/ubuntu/Downloads/lightningdbm//?.so;;"
</pre>

Instead of setting the paths in `~/.profile` file, we can also place the `lightningmdb.so` file in `<TORCH_INSTALLATION_DIR>/lib` directory

See this link for more information: `https://github.com/shmul/lightningdbm`
