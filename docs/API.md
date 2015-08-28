# REST API

*Generated Aug 20, 2015*

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

Location: [`digits/dataset/images/classification/views.py@245`](../digits/dataset/images/classification/views.py#L245)

### `/datasets/images/generic.json`

> Creates a new GenericImageDatasetJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/dataset/images/generic/views.py@24`](../digits/dataset/images/generic/views.py#L24)

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

Location: [`digits/model/views.py@91`](../digits/model/views.py#L91)

### `/models/images/classification.json`

> Create a new ImageClassificationModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@54`](../digits/model/images/classification/views.py#L54)

### `/models/images/classification/classify_many.json`

> Classify many images and return the top 5 classifications for each

> 

> Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@294`](../digits/model/images/classification/views.py#L294)

### `/models/images/classification/classify_one.json`

> Classify one image and return the top 5 classifications

> 

> Returns JSON when requested: {predictions: {category: confidence,...}}

Methods: **POST**

Location: [`digits/model/images/classification/views.py@237`](../digits/model/images/classification/views.py#L237)

### `/models/images/generic.json`

> Create a new GenericImageModelJob

> 

> Returns JSON when requested: {job_id,name,status} or {errors:[]}

Methods: **POST**

Location: [`digits/model/images/generic/views.py@50`](../digits/model/images/generic/views.py#L50)

### `/models/images/generic/infer_many.json`

> Infer many images

Methods: **POST**

Location: [`digits/model/images/generic/views.py@268`](../digits/model/images/generic/views.py#L268)

### `/models/images/generic/infer_one.json`

> Infer one image

Methods: **POST**

Location: [`digits/model/images/generic/views.py@215`](../digits/model/images/generic/views.py#L215)

