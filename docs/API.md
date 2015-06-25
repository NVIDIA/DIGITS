# REST API

*Generated Jun 25, 2015*

DIGITS exposes its internal functionality through a REST API. You can access these endpoints by performing a GET or POST on the route, and a JSON object will be returned.

For more information about other routes used for the web interface, see [this page](FlaskRoutes.md).

### `/datasets/<job_id>.json`

> Show a DatasetJob

> 

> Returns JSON when requested:

> {id, name, directory, status}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py@15`](../digits/dataset/views.py#L15)

### `/datasets/images/classification.json`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py@217`](../digits/dataset/images/classification/views.py#L217)

### `/index.json`

> DIGITS home page

> Returns information about each job on the server

> 

> Returns JSON when requested:

> {

> datasets: [{id, name, status},...],

> models: [{id, name, status},...]

> }

Methods: **GET**

Location: [`digits/views.py@22`](../digits/views.py#L22)

### `/models/<job_id>.json`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py@31`](../digits/model/views.py#L31)

### `/models/images/classification.json`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@53`](../digits/model/images/classification/views.py#L53)

### `/models/images/classification/classify_many.json`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@287`](../digits/model/images/classification/views.py#L287)

### `/models/images/classification/classify_one.json`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@233`](../digits/model/images/classification/views.py#L233)

