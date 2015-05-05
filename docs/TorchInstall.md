This page contains installation instructions for all the required modules to include the Torch framework in DIGITS

## Torch installation

Use the below link to install Torch on Mac OS X and Ubuntu 12+:

    % http://torch.ch/docs/getting-started.html

After installation, enter the following command:  

    %  which luarocks

to find the Torch installation directory. 

## LMDB and Lua Wrapper for LMDB (lightningdbm) installation

#### LMDB installation

If LMDB was already installed, make sure that the paths to liblmdb.so*, liblmdb.a and lmdb.h is properly set. If LMDB wasn't already installed, follow the below steps to install,

Get LMDB by `git clone https://gitorious.org/mdb/mdb.git`.

<pre>
% cd mdb/libraries/liblmdb/
% vi Makefile
</pre>

edit the `Makefile` to set the "prefix" parameter to `<TORCH_INSTALLATION_DIR>` as shown below,
<pre>
prefix = TORCH_INSTALLATION_DIR    #For instance, /home/ubuntu/torch/install

% make
% make install
</pre>

Note: In case if you face any errors during `make install` regarding `man` directory, then just create a `man` directory inside `<TORCH_INSTALLATION_DIR>` and execute `make install` command again.

After successful installation, some of the lmdb files will be included in torch installation directory as shown below,

<pre>
% ls ~/torch/install/lib -halt
total 6.2M
-rwxrwxr-x 1 ubuntu ubuntu 300K Jan 30 16:40 liblmdb.so*
-rw-rw-r-- 1 ubuntu ubuntu 634K Jan 30 16:40 liblmdb.a


% ls ~/torch/install/bin -halt
total 2.0M
-rwxrwxr-x 1 ubuntu ubuntu 318K Jan 30 16:40 mdb_dump
-rwxrwxr-x 1 ubuntu ubuntu 318K Jan 30 16:40 mdb_load
-rwxrwxr-x 1 ubuntu ubuntu 303K Jan 30 16:40 mdb_copy
-rwxrwxr-x 1 ubuntu ubuntu 317K Jan 30 16:40 mdb_stat


% ls ~/torch/install/include -halt
total 140K
-rw-rw-r-- 1 ubuntu ubuntu  71K Jan 30 16:40 lmdb.h
</pre>

If LMDB was already installed, make sure that the paths to liblmdb.so*, liblmdb.a and lmdb.h is properly set.

#### Lua Wrapper for LMDB installation
Get Lua Wrapper for LMDB by `git clone https://github.com/shmul/lightningdbm.git`

Below changes are required in 'make' file to set the paths:

<pre>
LUAINC = <TORCH_INSTALLATION_DIR>/include
LUALIB = <TORCH_INSTALLATION_DIR>/lib
LUABIN = <TORCH_INSTALLATION_DIR>/bin
LMDBINC = <TORCH_INSTALLATION_DIR>/include
LMDBLIB = <TORCH_INSTALLATION_DIR>/lib
</pre>

Also change `$(LUABIN)/lua` to `$(LUABIN)/luajit`, to refer luajit instead of lua.

Finally set the below paths in `~/.profile` file, to resolve the following error :  `module 'lightningmdb' not found:No LuaRocks module found for lightningmdb`

<pre>
export LUA_PATH="<LIGHTININGMDB_INSTALLATION_DIR>//?.lua;;"
export LUA_CPATH="<LIGHTININGMDB_INSTALLATION_DIR>//?.so;;"

For example: 
export LUA_PATH="/home/ubuntu/Downloads/lightningdbm//?.lua;;"
export LUA_CPATH="/home/ubuntu/Downloads/lightningdbm//?.so;;"
</pre>

Instead of setting the paths in `~/.profile` file, we can also place the `lightningmdb.so` file in `<TORCH_INSTALLATION_DIR>/lib` directory

Please check this link for more information regarding lightningdbm: `https://github.com/shmul/lightningdbm`

## Installation of Lua Protocol Buffers

Use the below command to install Lua Protocol Buffers,

<pre>
luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
</pre>

Please check this link for more information: `https://github.com/Neopallium/lua-pb`

set the path (in `~/.bashrc` or `~/.profile`) for all the `.proto`, `.lua` and `.so` files in pb

<pre>
export LUA_PATH="/home/ubuntu/Downloads/lightningdbmForLuaJit//?.lua;/home/ubuntu/Downloads/lua-pb-master//?.proto;/home/ubuntu/Downloads/lua-pb-master//?.lua;;"
export LUA_CPATH="/home/ubuntu/Downloads/lightningdbmForLuaJit//?.so;/home/ubuntu/Downloads/lua-pb-master//?.so;;"
</pre>

## Installation of ccn2 (Torch7 bindings for cuda-convnet2 kernels)

Some networks make use of ccn2 to achieve good run-time performance. Use the below command to install ccn2, 

<pre>
luarocks install ccn2
</pre>

Please check this link for more information: `https://github.com/soumith/cuda-convnet2.torch`

