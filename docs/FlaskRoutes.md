# Flask Routes

*Generated Jun 10, 2015*

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

Location: [`digits/views.py@21`](../digits/views.py#L21)

## Jobs

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@137`](../digits/views.py#L137)

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@157`](../digits/views.py#L157)

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@118`](../digits/views.py#L118)

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@85`](../digits/views.py#L85)

### `/jobs/<job_id>`

> Edit the name of a job

Methods: **PUT**

Arguments: `job_id`

Location: [`digits/views.py@102`](../digits/views.py#L102)

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@137`](../digits/views.py#L137)

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@157`](../digits/views.py#L157)

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@118`](../digits/views.py#L118)

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@137`](../digits/views.py#L137)

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@157`](../digits/views.py#L157)

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@118`](../digits/views.py#L118)

## Datasets

### `/datasets/<job_id>`

> Show a DatasetJob

> 

> Returns JSON when requested:

> {id, name, directory, status}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py@15`](../digits/dataset/views.py#L15)

### `/datasets/images/classification`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py@217`](../digits/dataset/images/classification/views.py#L217)

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py@207`](../digits/dataset/images/classification/views.py#L207)

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

Location: [`digits/dataset/images/views.py@17`](../digits/dataset/images/views.py#L17)

### `/datasets/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/views.py@36`](../digits/dataset/views.py#L36)

## Models

### `/models/<job_id>`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@27`](../digits/model/views.py#L27)

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

Location: [`digits/model/views.py@154`](../digits/model/views.py#L154)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

Location: [`digits/model/views.py@154`](../digits/model/views.py#L154)

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

Location: [`digits/model/views.py@49`](../digits/model/views.py#L49)

### `/models/images/classification`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@49`](../digits/model/images/classification/views.py#L49)

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@287`](../digits/model/images/classification/views.py#L287)

### `/models/images/classification/classify_one`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@231`](../digits/model/images/classification/views.py#L231)

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/classification/views.py@218`](../digits/model/images/classification/views.py#L218)

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/classification/views.py@28`](../digits/model/images/classification/views.py#L28)

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/classification/views.py@364`](../digits/model/images/classification/views.py#L364)

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

Location: [`digits/model/views.py@99`](../digits/model/views.py#L99)

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

Location: [`digits/model/views.py@86`](../digits/model/views.py#L86)

## Util

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

Location: [`digits/views.py@201`](../digits/views.py#L201)

