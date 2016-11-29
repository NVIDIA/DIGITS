# Configuration

DIGITS uses environment variables for configuration.

> NOTE: Prior to [#1091](https://github.com/NVIDIA/DIGITS/pull/1091) (up to DIGITS 4.0), DIGITS used configuration files instead of environment variables.


## Environment Variables

| Variable | Example value | Description |
| --- | --- | --- |
| `DIGITS_JOBS_DIR` | ~/digits-jobs | Location where job files are stored. Default is `$DIGITS_ROOT/digits/jobs`. |
| `CAFFE_ROOT` | ~/caffe | Path to your local Caffe build. Should contain `build/tools/caffe` and `python/caffe/`. If unset, looks for `caffe` in PATH and PYTHONPATH.|
| `TORCH_ROOT` | ~/torch | Path to your local Torch build. Should contain `install/bin/th`. If unset, looks for `th` in PATH. |
| `DIGITS_LOGFILE_FILENAME` | ~/digits.log | File for saving log messages. Default is `$DIGITS_ROOT/digits/digits.log`. |
| `DIGITS_LOGFILE_LEVEL` | DEBUG | Minimum log message level to be saved (DEBUG/INFO/WARNING/ERROR/CRITICAL). Default is INFO. |
| `DIGITS_SERVER_NAME` | The Big One | The name of the server (accessible in the UI under "Info"). Default is the system hostname. |
| `DIGITS_MODEL_STORE_URL` | http://localhost/modelstore | A list of URL's, separated by comma. Default is the official NVIDIA store. |
