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

Location: [`digits/views.py@19`](../digits/views.py#L19)

### `/index.json`

> JSON version of the DIGITS home page

> Returns information about each job on the server

Methods: **GET**

Location: [`digits/views.py@53`](../digits/views.py#L53)

## Jobs

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@141`](../digits/views.py#L141)

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@160`](../digits/views.py#L160)

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@122`](../digits/views.py#L122)

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@87`](../digits/views.py#L87)

### `/jobs/<job_id>`

> Edit the name of a job

Methods: **PUT**

Arguments: `job_id`

Location: [`digits/views.py@105`](../digits/views.py#L105)

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@141`](../digits/views.py#L141)

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@160`](../digits/views.py#L160)

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@122`](../digits/views.py#L122)

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@141`](../digits/views.py#L141)

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@160`](../digits/views.py#L160)

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@122`](../digits/views.py#L122)

## Datasets

### `/datasets/<job_id>`

> Show a DatasetJob

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py@12`](../digits/dataset/views.py#L12)

### `/datasets/images/classification`

> Creates a new ImageClassificationDatasetJob

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py@217`](../digits/dataset/images/classification/views.py#L217)

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py@208`](../digits/dataset/images/classification/views.py#L208)

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

Location: [`digits/dataset/images/views.py@16`](../digits/dataset/images/views.py#L16)

### `/datasets/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/views.py@28`](../digits/dataset/views.py#L28)

## Models

### `/models/<job_id>`

> Show a ModelJob

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@26`](../digits/model/views.py#L26)

### `/models/<job_id>.json`

> Return a JSON representation of a ModelJob

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@42`](../digits/model/views.py#L42)

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

Location: [`digits/model/views.py@167`](../digits/model/views.py#L167)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

Location: [`digits/model/views.py@167`](../digits/model/views.py#L167)

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

Location: [`digits/model/views.py@62`](../digits/model/views.py#L62)

### `/models/images/classification`

> Create a new ImageClassificationModelJob

Methods: **POST**

Location: [`digits/model/images/classification/views.py@49`](../digits/model/images/classification/views.py#L49)

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

Methods: **POST**

Location: [`digits/model/images/classification/views.py@264`](../digits/model/images/classification/views.py#L264)

### `/models/images/classification/classify_one`

> Classify one image and return the predictions, weights and activations

Methods: **POST**

Location: [`digits/model/images/classification/views.py@214`](../digits/model/images/classification/views.py#L214)

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/classification/views.py@202`](../digits/model/images/classification/views.py#L202)

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/classification/views.py@29`](../digits/model/images/classification/views.py#L29)

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/classification/views.py@334`](../digits/model/images/classification/views.py#L334)

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

Location: [`digits/model/views.py@112`](../digits/model/views.py#L112)

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

Location: [`digits/model/views.py@99`](../digits/model/views.py#L99)

## Util

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

Location: [`digits/views.py@191`](../digits/views.py#L191)

