# Flask Routes

*Generated Aug 06, 2015*

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

Location: [`digits/views.py@23`](../digits/views.py#L23)

## Jobs

### `/datasets/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@149`](../digits/views.py#L149)

### `/datasets/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@169`](../digits/views.py#L169)

### `/datasets/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@130`](../digits/views.py#L130)

### `/jobs/<job_id>`

> Redirects to the appropriate /datasets/ or /models/ page

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@97`](../digits/views.py#L97)

### `/jobs/<job_id>`

> Edit the name of a job

Methods: **PUT**

Arguments: `job_id`

Location: [`digits/views.py@114`](../digits/views.py#L114)

### `/jobs/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@149`](../digits/views.py#L149)

### `/jobs/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@169`](../digits/views.py#L169)

### `/jobs/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@130`](../digits/views.py#L130)

### `/models/<job_id>`

> Deletes a job

Methods: **DELETE**

Arguments: `job_id`

Location: [`digits/views.py@149`](../digits/views.py#L149)

### `/models/<job_id>/abort`

> Aborts a running job

Methods: **POST**

Arguments: `job_id`

Location: [`digits/views.py@169`](../digits/views.py#L169)

### `/models/<job_id>/status`

> Returns a JSON objecting representing the status of a job

Methods: **GET**

Arguments: `job_id`

Location: [`digits/views.py@130`](../digits/views.py#L130)

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

Location: [`digits/dataset/images/classification/views.py@245`](../digits/dataset/images/classification/views.py#L245)

### `/datasets/images/classification/new`

> Returns a form for a new ImageClassificationDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py@235`](../digits/dataset/images/classification/views.py#L235)

### `/datasets/images/classification/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/images/classification/views.py@298`](../digits/dataset/images/classification/views.py#L298)

### `/datasets/images/generic`

> Creates a new GenericImageDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/generic/views.py@27`](../digits/dataset/images/generic/views.py#L27)

### `/datasets/images/generic/new`

> Returns a form for a new GenericImageDatasetJob

Methods: **GET**

Location: [`digits/dataset/images/generic/views.py@17`](../digits/dataset/images/generic/views.py#L17)

### `/datasets/images/generic/summary`

> Return a short HTML summary of a DatasetJob

Methods: **GET**

Location: [`digits/dataset/images/generic/views.py@106`](../digits/dataset/images/generic/views.py#L106)

### `/datasets/images/resize-example`

> Resizes the example image, and returns it as a string of png data

Methods: **POST**

Location: [`digits/dataset/images/views.py@18`](../digits/dataset/images/views.py#L18)

## Models

### `/models/<job_id>`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@31`](../digits/model/views.py#L31)

### `/models/<job_id>/download`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension` (`tar.gz`)

Location: [`digits/model/views.py@160`](../digits/model/views.py#L160)

### `/models/<job_id>/download.<extension>`

> Return a tarball of all files required to run the model

Methods: **GET**, **POST**

Arguments: `job_id`, `extension`

Location: [`digits/model/views.py@160`](../digits/model/views.py#L160)

### `/models/customize`

> Returns a customized file for the ModelJob based on completed form fields

Methods: **POST**

Location: [`digits/model/views.py@55`](../digits/model/views.py#L55)

### `/models/images/classification`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@53`](../digits/model/images/classification/views.py#L53)

### `/models/images/classification/classify_many`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@290`](../digits/model/images/classification/views.py#L290)

### `/models/images/classification/classify_one`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **GET**, **POST**

Location: [`digits/model/images/classification/views.py@236`](../digits/model/images/classification/views.py#L236)

### `/models/images/classification/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/classification/views.py@225`](../digits/model/images/classification/views.py#L225)

### `/models/images/classification/new`

> Return a form for a new ImageClassificationModelJob

Methods: **GET**

Location: [`digits/model/images/classification/views.py@32`](../digits/model/images/classification/views.py#L32)

### `/models/images/classification/top_n`

> Classify many images and show the top N images per category by confidence

Methods: **POST**

Location: [`digits/model/images/classification/views.py@374`](../digits/model/images/classification/views.py#L374)

### `/models/images/generic`

> Create a new GenericImageModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/generic/views.py@52`](../digits/model/images/generic/views.py#L52)

### `/models/images/generic/infer_many`

> Infer many images

Methods: **GET**, **POST**

Location: [`digits/model/images/generic/views.py@267`](../digits/model/images/generic/views.py#L267)

### `/models/images/generic/infer_one`

> Infer one image

Methods: **GET**, **POST**

Location: [`digits/model/images/generic/views.py@217`](../digits/model/images/generic/views.py#L217)

### `/models/images/generic/large_graph`

> Show the loss/accuracy graph, but bigger

Methods: **GET**

Location: [`digits/model/images/generic/views.py@206`](../digits/model/images/generic/views.py#L206)

### `/models/images/generic/new`

> Return a form for a new GenericImageModelJob

Methods: **GET**

Location: [`digits/model/images/generic/views.py@32`](../digits/model/images/generic/views.py#L32)

### `/models/visualize-lr`

> Returns a JSON object of data used to create the learning rate graph

Methods: **POST**

Location: [`digits/model/views.py@105`](../digits/model/views.py#L105)

### `/models/visualize-network`

> Returns a visualization of the custom network as a string of PNG data

Methods: **POST**

Location: [`digits/model/views.py@92`](../digits/model/views.py#L92)

## Util

### `/autocomplete/path`

> Return a list of paths matching the specified preamble

Methods: **GET**

Location: [`digits/views.py@255`](../digits/views.py#L255)

### `/files/<path:path>`

> Return a file in the jobs directory

> 

> If you install the nginx.site file, nginx will serve files instead

> and this path will never be used

Methods: **GET**

Arguments: `path`

Location: [`digits/views.py@227`](../digits/views.py#L227)

