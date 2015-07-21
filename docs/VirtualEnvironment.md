# Virtual Environment

DIGITS depends on several python packages from [PyPI](https://pypi.python.org/pypi).

The recommended installation method is to use a [virtual environment](https://virtualenv.pypa.io/). This separates your python packages for DIGITS from any other packages you have installed on your system.

## Install virtualenv

    % pip install virtualenv

## Create the environment

Change into the directory for your DIGITS installation and create the virtual environment there (or wherever else you like):

    % cd $DIGITS_HOME
    % virtualenv venv
    % source venv/bin/activate

Now your path has been updated to use `python` and `pip` from your virtual environment:

```
$ which python
/home/username/digits/venv/bin/python

$ which pip
/home/username/digits/venv/bin/pip
```
