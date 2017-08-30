# Object Detection Data Extension

This data extension creates DIGITS datasets for object detection networks such as [DetectNet](https://github.com/NVIDIA/caffe/tree/caffe-0.15/examples/kitti).

DIGITS uses the KITTI format for object detection data.
When preparing your own data for ingestion into a dataset, you must follow the same format.

#### Table of contents

* [Folder structure](#folder-structure)
* [Label format](#label-format)
* [Custom class mappings](#custom-class-mappings)

## Folder structure

You should have one folder containing images, and another folder containing labels.

* Image filenames are formatted like `IDENTIFIER.EXTENSION` (e.g. `000001.png` or `2.jpg`).
* Label filenames are formatted like `IDENTIFIER.txt` (e.g. `000001.txt` or `2.txt`).

These identifiers need to match.
So, if you have a `1.png` in your image directory, there must to be a corresponding `1.txt` in your labels directory.

If you want to include validation data, then you need separate folders for validation images and validation labels.
A typical folder layout would look something like this:
```
train/
├── images/
│   └── 000001.png
└── labels/
    └── 000001.txt
val/
├── images/
│   └── 000002.png
└── labels/
    └── 000002.txt
```

## Label format

The format for KITTI labels is explained in the `readme.txt` from the "Object development kit".
Here is the relevant portion:
```
Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

Here, 'DontCare' labels denote regions in which objects have not been labeled,
for example because they have been too far away from the laser scanner. To
prevent such objects from being counted as false positives our evaluation
script will ignore objects detected in don't care regions of the test set.
You can use the don't care labels in the training set to avoid that your object
detector is harvesting hard negatives from those areas, in case you consider
non-object regions from the training images as negative examples.

The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
will change.

To project a point from Velodyne coordinates into the left color image,
you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

Note: All matrices are stored row-major, i.e., the first values correspond
to the first row. R0_rect contains a 3x3 matrix which you need to extend to
a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
in the same way!

Note, that while all this information is available for the training data,
only the data which is actually needed for the particular benchmark must
be provided to the evaluation server. However, all 15 values must be provided
at all times, with the unused ones set to their default values (=invalid) as
specified in writeLabels.m. Additionally a 16'th value must be provided
with a floating value of the score for a particular detection, where higher
indicates higher confidence in the detection. The range of your scores will
be automatically determined by our evaluation server, you don't have to
normalize it, but it should be roughly linear. If you use writeLabels.m for
writing your results, this function will take care of storing all required
data correctly.
```

## Custom class mappings

When creating the dataset, DIGITS has to translate from the object type string to a numerical identifier.
By default, DIGITS uses the following class mappings, as follows from the above label format description:

Class name (string in label file) | Class ID (number in database)
---------- | ---
dontcare | 0
car | 1
van | 2
truck | 3
bus | 4
pickup | 5
vehicle-with-trailer | 6
special-vehicle | 7
person | 8
person-fa | 9
person? | 10
people | 11
cyclist | 12
tram | 13
person_sitting | 14

**NOTE:** Class 0 is treated as a special case.
See "Label format" above for a detailed description.
All classes which don't exist in the provided mapping are implicitly mapped to 0.

**NOTE:** Class 1 is also treated as a special case.
DetectNet is a single-class object detection network, and only cares about the "Car" class, which is expected to be ID 1.
You can change the mapping in the DetectNet prototxt, but it's simplest to just make the class you care about map to 1.

Custom class mappings may be used by specifying a comma-separated list of class names in the Object Detection dataset creation form.
All labels are converted to lower-case, so the matching is case-insensitive.

For example, if you only want to detect pedestrians, enter `dontcare,pedestrian` in the "Custom classes" field to generate this mapping:

Class name | Class ID
---------- | ---
dontcare | 0
pedestrian | 1

All labeled objects other than "pedestrian" in your dataset will be mapped to 0, along with any objects explicitly labeled as "dontcare".
