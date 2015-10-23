# REST API

*Generated Oct 09, 2015*

DIGITS exposes its internal functionality through a REST API. You can access these endpoints by performing a GET or POST on the route, and a JSON object will be returned.

For more information about other routes used for the web interface, see [this page](FlaskRoutes.md).

### `/datasets/<job_id>.json`

> Show a DatasetJob

> 

> Returns JSON when requested:

> {id, name, directory, status}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/dataset/views.py`](../digits/dataset/views.py)

### `/datasets/images/classification.json`

> Creates a new ImageClassificationDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/classification/views.py`](../digits/dataset/images/classification/views.py)

### `/datasets/images/generic.json`

> Creates a new GenericImageDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/generic/views.py`](../digits/dataset/images/generic/views.py)

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

Location: [`digits/views.py`](../digits/views.py)

### `/models/<job_id>.json`

> Show a ModelJob

> 

> Returns JSON when requested:

> {id, name, directory, status, snapshots: [epoch,epoch,...]}

Methods: **GET**

Arguments: `job_id`

Location: [`digits/model/views.py`](../digits/model/views.py)

### `/models/images/classification.json`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/classify_many.json`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/classification/classify_one.json`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py`](../digits/model/images/classification/views.py)

### `/models/images/generic.json`

> Create a new GenericImageModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/infer_many.json`

> Infer many images

Methods: **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

### `/models/images/generic/infer_one.json`

> Infer one image

Methods: **POST**

Location: [`digits/model/images/generic/views.py`](../digits/model/images/generic/views.py)

