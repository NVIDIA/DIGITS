# Model Store


## Introduction
Model Store is a new feature in DIGITS.
It lists models in user-specified servers and imports them into DIGITS.


## Setting up environment variable
The configuration of Model Store requires one environment variable DIGITS_MODEL_STORE_URL to be set.
NVIDIA plans to publish one public Model Store at http://developer.download.nvidia.com/compute/machine-learning/modelstore/4.5.0.
You can set up the environment variable with that url before launching DIGITS.
For example, run the following command in your Bash shell.
``` shell
export DIGITS_MODEL_STORE_URL='http://developer.download.nvidia.com/compute/machine-learning/modelstore/4.5.0'
```
If multiple model stores are available, specify their url's, separated by the comma (,).
``` shell
export DIGITS_MODEL_STORE_URL='http://localhost/mymodelstore,http://dlserver/teammodelstore'
```


## Browse and import model
First launch DIGITS.
Click Pretrained Models tab, and select 'Retrieve from Model Store' under 'Load Model.'
The new page shows models available in model stores.

Hoover over the model name shows the complete text in Note field.
Enter keyword in 'Filter list by' to limit results.
Click 'Update model list' button will retrieve the latest model list (see limitation).
Click the model name will import that model into DIGITS (may takes a few seconds, depends on network speed).

After successfully importing the model, DIGITS redirects the browser to Home page.
The Pretrained Models table will show the newly imported model.
At this moment, you can use that imported model like other pretrained models.


## Create your own Model Store server
At the top directory, create a master.json file (if your server does not support directory listing).
The following is a sample master.json file.
```
{'msg':'This is my own model store server.', 'children':['Model01','Model02']}
```
Model01 and Model02 are subdirectories containing the actual models.
Each model must consist one weight file, one info.json file.
The info.json file is in the format of same file inside DIGITS downloaded model.
The subdirectory can optionally contain aux.json, license.txt, logo.png.
Information in those files populate the fields in Model Store table.


## Limitation
Some web server may limit frequent requests from the same machine to stop malicious activities.
Therefore, DIGITS implemented a cache mechanism to reduce server-to-server communication.
The button, 'Update model list', invalidates cache and retrieve meta data from all models.
