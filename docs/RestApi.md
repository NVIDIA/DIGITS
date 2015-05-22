# DIGITS REST API

Documentation on the various REST routes in DIGITS.

### Table of Contents

* [Home](#home)
* [Jobs](#jobs)
* [Datasets](#datasets)
* [Models](#models)
* [Util](#util)

## Home

### `/`

> DIGITS home page

> Displays all datasets and models on the server and their status

Methods: **GET**

### `/index.json`

> JSON version of the DIGITS home page

> Returns information about each job on the server

Methods: **GET**

## Jobs

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

### `/jobs/<job_id>`

> Edit the name of a job

Methods: **PUT**

Arguments: `job_id`

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

## Datasets

### `/datasets/<job_id>`

> Show a DatasetJob

Methods: **GET**

Arguments: `job_id`

### `/datasets/images/classification`

> Creates a new ImageClassificationDatasetJob

Methods: **POST**

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

### `/datasets/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

## Models

### `/models/<job_id>`

> Show a ModelJob

Methods: **GET**

Arguments: `job_id`

### `/models/<job_id>.json`

> Return a JSON representation of a ModelJob

Methods: **GET**

Arguments: `job_id`

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

### `/models/images/classification`

> Create a new ImageClassificationModelJob

Methods: **POST**

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

Methods: **POST**

### `/models/images/classification/classify_one`

> Classify one image and return the predictions, weights and activations

Methods: **POST**

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

## Util

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

