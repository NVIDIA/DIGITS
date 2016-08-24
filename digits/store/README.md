# Model Store

Model Store is the place to store DIGITS models.
Users can publish model directly from DIGITS or import them as pre-trained models.
An adm page is provided to publish models by uploading local files and remove existing models.

A model consists of the weight file, a info.json file, original protocol buffer text file, and several text description.  

# Installation

There are several required packages.
You can install them with Python package manager and the provided requirements.txt.

# Running

Inside this folder, run
```shell
./access.py
```
and the server will listen to port 5050.

You can specify the port to use with option -p  or --port.
 