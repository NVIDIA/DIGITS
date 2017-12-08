# Ubuntu Installation

## Getting started

After getting DIGITS set up, the DIGITS server should now be running at `http://localhost:5000`.
See comments below if you run into any issues.

Now that you're up and running, check out the [Getting Started Guide](GettingStarted.md).

## Troubleshooting

#### Configuration

If you have another server running on port 80 already, you may need to reconfigure DIGITS to use a different port.
```sh
sudo dpkg-reconfigure digits
```

All other configuration is done with environment variables.
See [Configuration.md](Configuration.md) for detailed information about which variables you can change.

* Ubuntu 14.04
  * Edit `/etc/init/digits.conf`
  * Add/remove/edit lines that start with `env`
  * Restart with `sudo service digits restart`

* Ubuntu 16.04
  * Edit `/lib/systemd/system/digits.service`
  * Add/remove/edit lines that start with `Environment=` in the `[Service]` section
  * Restart with `sudo systemctl daemon-reload && sudo systemctl restart digits`

#### Driver installations

If you try to install a new driver while the DIGITS server is running, you'll get an error about CUDA being in use.
Shut down the server before installing a driver, and then restart it afterwards.

Ubuntu 14.04:
```sh
sudo service digits stop
# (install driver)
sudo service digits start
```
Ubuntu 16.04:
```sh
sudo systemctl stop digits
# (install driver)
sudo systemctl start digits
```

#### Permissions

The DIGITS server runs as `www-data`, so keep in mind that prebuilt LMDB datasets used for generic models need to be readable by the `www-data` user.
In particular, the entire chain of directories from `/` to your data must be readable by `www-data`.

#### Torch and cusparse

There is at least one Torch package which is missing a required dependency on cusparse.
If you see this error:
```
/usr/share/lua/5.1/cunn/THCUNN.lua:7: libcusparse.so.7.5: cannot open shared object file: No such file or directory
```
The simplest fix is to manually install the missing library:
```sh
sudo apt-get install cuda-cusparse-7-5
sudo ldconfig
```

#### Torch and HDF5

There is at least one Torch package which is missing a required dependency on libhdf5-dev.
If you see this error:
```
ERROR: /usr/share/lua/5.1/trepl/init.lua:384: /usr/share/lua/5.1/trepl/init.lua:384: /usr/share/lua/5.1/hdf5/ffi.lua:29: libhdf5.so: cannot open shared object file: No such file or directory
```
The simplest fix is to manually install the missing library:
```sh
sudo apt-get install libhdf5-dev
sudo ldconfig
```

#### Other

If you run into an issue not addressed here, try searching through the [GitHub issues](https://github.com/NVIDIA/DIGITS/issues) and/or the [user group](https://groups.google.com/d/forum/digits-users).
