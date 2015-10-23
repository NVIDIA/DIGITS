# Flask Routes

*Generated Oct 09, 2015*

Documentation on the various routes used internally for the web application.

These are all technically RESTful, but they return HTML pages. To get JSON responses, see [this page](API.md).

### Table of Contents

* [Home](#home)
* [Jobs](#jobs)
* [Datasets](#datasets)
* [Models](#models)
* [Util](#util)

## Home

### `/`

> DIGITS home page

> Returns information about each job on the server

> 

> Returns JSON when requested:

> {

> datasets: [{id, name, status},...],

> models: [{id, name, status},...]

> }

Methods: **GET**

Location: [`digits/views.py`](../digits/views.py)

## Jobs

### `/clone/<clone>`

> Clones a job with the id <clone>, populating the creation page with data saved in <clone>

Methods: **GET**, **POST**

Arguments: `clone`

Location: [`digits/views.py`](../digits/views.py)

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/jobs/<job_id>`

> Edit a job's name and/or notes

Methods: **PUT**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py`](../digits/views.py)

## Datasets

### `/datasets/<job_id>`

> Show a DatasetJob

> 

> Returns JSON when requested:

> {id, name, directory, status}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py`](../digits/dataset/views.py)

### `/datasets/images/classification`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py`](../digits/dataset/images/classification/views.py)

### `/datasets/images/classification/explore`

> Returns a gallery consisting of the images of one of the dbs

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py`](../digits/dataset/images/classification/views.py)

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py`](../digits/dataset/images/classification/views.py)

### `/datasets/images/classification/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py`](../digits/dataset/images/classification/views.py)

### `/datasets/images/generic`

> Creates a new GenericImageDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/generic/views.py`](../digits/dataset/images/generic/views.py)

### `/datasets/images/generic/new`

> Returns a form for a new GenericImageDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/generic/views.py`](../digits/dataset/images/generic/views.py)

### `/datasets/images/generic/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/images/generic/views.py`](../digits/dataset/images/generic/views.py)

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

Location: [`digits/dataset/images/views.py`](../digits/dataset/images/views.py)

## Models

### `/models/`

Methods: **GET**

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/<job_id>`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/images/classification`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/classify_one`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/generic`

> Create a new GenericImageModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/infer_many`

> Infer many images

Methods: **GET**, **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/infer_one`

> Infer one image

Methods: **GET**, **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/new`

> Return a form for a new GenericImageModelJob

Methods: **GET**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

Location: [`digits/model/views.py`](../digits/model/views.py)

## Util

### `/autocomplete/path`

> Return a list of paths matching the specified preamble

Methods: **GET**

Location: [`digits/views.py`](../digits/views.py)

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

Location: [`digits/views.py`](../digits/views.py)

