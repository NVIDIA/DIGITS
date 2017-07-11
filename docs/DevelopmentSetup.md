# Development

The source code for DIGITS is available on [github](https://github.com/NVIDIA/DIGITS).

To have access to your local machine, you may clone from the github repository with
```
git clone https://github.com/NVIDIA/DIGITS.git
```
Or you may download the source code as a zip file from the github website.

## Running DIGITS in Development

DIGITS comes with the script to run for a development server.
To run the development server, use
```
./digits-devserver
```

## Running unit tests for DIGITS

To successfully run all the unit tests, the following plugins have to be installed
```
sudo pip install -r ./requirements_test.txt
```

To run all the tests for DIGITS, use
```
./digits-test
```

If you would like to have a verbose output with the name of the tests, use
```
./digits-test -v
```