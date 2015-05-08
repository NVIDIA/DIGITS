This page contains installation instructions for all the required modules to include the Torch framework in DIGITS

## Torch installation

Use the below link to install Torch on Mac OS X and Ubuntu 12+:

    % http://torch.ch/docs/getting-started.html

After installation, enter the following command:  

    %  which luarocks

to find the Torch installation directory. 

## LMDB and Lua Wrapper for LMDB (lightningdbm) installation

#### LMDB installation

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

#### Lua Wrapper for LMDB installation
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

Please check this link for more information regarding lightningdbm: `https://github.com/shmul/lightningdbm`

## Installation of Lua Protocol Buffers

Use the below command to install Lua Protocol Buffers,

<pre>
luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
</pre>

Please check this link for more information: `https://github.com/Neopallium/lua-pb`

## Installation of ccn2 (Torch7 bindings for cuda-convnet2 kernels)

Some networks make use of ccn2 to achieve good run-time performance. Use the below command to install ccn2, 

<pre>
luarocks install ccn2
</pre>

Please check this link for more information: `https://github.com/soumith/cuda-convnet2.torch`

