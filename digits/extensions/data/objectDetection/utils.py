# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import csv
import numpy as np
import os

class ObjectType:

    Dontcare, Car, Van, Truck, Bus, Pickup, VehicleWithTrailer, SpecialVehicle,\
    Person, Person_fa, Person_unsure, People, Cyclist, Tram, Person_Sitting,\
    Misc = range(16)

    def __init__(self):
        pass

class Bbox:

    def __init__(self, x_left=0, y_top=0, x_right=0, y_bottom=0):
        self.xl = x_left
        self.yt = y_top
        self.xr = x_right
        self.yb = y_bottom

    def area(self):
        return (self.xr - self.xl) * (self.yb - self.yt)

    def width(self):
        return self.xr - self.xl

    def height(self):
        return self.yb - self.yt

    def get_array(self):
        return [self.xl, self.yt, self.xr, self.yb]

class GroundTruthObj:

    """ This class is the data ground-truth

        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                          'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                          'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                          truncated refers to the object leaving image boundaries.
                          -1 corresponds to a don't care region.
        1    occluded     Integer (-1,0,1,2) indicating occlusion state:
                          -1 = unkown, 0 = fully visible,
                          1 = partly occluded, 2 = largely occluded
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                          contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                          detection, needed for p/r curves, higher is better.

        Here, 'DontCare' labels denote regions in which objects have not been labeled,
        for example because they have been too far away from the laser scanner.
    """


    def __init__(self):
        self.stype = ''
        self.truncated = 0
        self.occlusion = 0
        self.angle = 0
        self.height = 0
        self.width = 0
        self.length = 0
        self.locx = 0
        self.locy = 0
        self.locz = 0
        self.roty = 0
        self.bbox = Bbox()
        self.object = ObjectType.Dontcare

    @classmethod
    def lmdb_format_length(cls):
        """
        width of an LMDB datafield returned by the gt_to_lmdb_format function.
        :return:
        """
        return 16

    def gt_to_lmdb_format(self):
        """
        For storage of a bbox ground truth object into a float32 LMDB.
         Sort-by attribute is always the last value in the array.
        """
        result = [
            # bbox in x,y,w,h format:
            self.bbox.xl,
            self.bbox.yt,
            self.bbox.xr - self.bbox.xl,
            self.bbox.yb - self.bbox.yt,
            # alpha angle:
            self.angle,
            # class number:
            self.object,
            0,
            # Y axis rotation:
            self.roty,
            # bounding box attributes:
            self.truncated,
            self.occlusion,
            # object dimensions:
            self.length,
            self.width,
            self.height,
            self.locx,
            self.locy,
            # depth (sort-by attribute):
            self.locz,
        ]
        assert(len(result) is self.lmdb_format_length())
        return result

    def set_type(self):
        object_types = {
            'bus': ObjectType.Bus,
            'car': ObjectType.Car,
            'cyclist': ObjectType.Cyclist,
            'pedestrian': ObjectType.Person,
            'people': ObjectType.People,
            'person': ObjectType.Person,
            'person_sitting': ObjectType.Person_Sitting,
            'person-fa': ObjectType.Person_fa,
            'person?': ObjectType.Person_unsure,
            'pickup': ObjectType.Pickup,
            'misc': ObjectType.Misc,
            'special-vehicle': ObjectType.SpecialVehicle,
            'tram': ObjectType.Tram,
            'truck': ObjectType.Truck,
            'van': ObjectType.Van,
            'vehicle-with-trailer': ObjectType.VehicleWithTrailer
        }
        self.object = object_types.get(self.stype, ObjectType.Dontcare)


class GroundTruth:

    """ this class load the ground truth
    """

    def __init__(self, label_dir, label_ext='.txt', label_delimiter=' '):
        self.label_dir = label_dir
        self.label_ext = label_ext  # extension of label files
        self.label_delimiter = label_delimiter  # space is used as delimiter in label files
        self._objects_all = dict()  # positive bboxes across images

    def update_objects_all(self, _key, _bboxes):
        if _bboxes:
            self._objects_all[_key] = _bboxes
        else:
            self._objects_all[_key] = none

    def load_gt_obj(self):

        """ load bbox ground truth from files either via the provided label directory or list of label files"""
        files = os.listdir(self.label_dir)
        files = filter(lambda x: x.endswith(self.label_ext), files)
        if len(files) == 0:
            raise exception('error: no label files found in', self.label_dir)
        for label_file in files:
            objects_per_image = list()
            with open( os.path.join(self.label_dir, label_file), 'rb') as flabel:
                for row in csv.reader(flabel, delimiter=self.label_delimiter):

                    # load data
                    gt = GroundTruthObj()
                    gt.stype = row[0].lower()
                    gt.truncated = float(row[1])
                    gt.occlusion = int(row[2])
                    gt.angle = float(row[3])
                    gt.bbox.xl = float(row[4])
                    gt.bbox.yt = float(row[5])
                    gt.bbox.xr = float(row[6])
                    gt.bbox.yb = float(row[7])
                    gt.height = float(row[8])
                    gt.width = float(row[9])
                    gt.length = float(row[10])
                    gt.locx = float(row[11])
                    gt.locy = float(row[12])
                    gt.locz = float(row[13])
                    gt.roty = float(row[14])
                    gt.set_type()
                    objects_per_image.append(gt)
                    key = int(os.path.splitext(label_file)[0])
                self.update_objects_all(key, objects_per_image)

    @property
    def objects_all(self):
        return self._objects_all

# return the # of pixels remaining in a

def pad_bbox(arr, max_bboxes=64, bbox_width=16):
    if arr.shape[0] > max_bboxes:
        raise ValueError(
            'Too many bounding boxes (%d > %d)' % arr.shape[0], max_bboxes
        )
    # fill remainder with zeroes:
    data = np.zeros((max_bboxes+1, bbox_width), dtype='float')
    # number of bounding boxes:
    data[0][0] = arr.shape[0]
    # width of a bounding box:
    data[0][1] = bbox_width
    # bounding box data. Merge nothing if no bounding boxes exist.
    if arr.shape[0] > 0:
        data[1:1 + arr.shape[0]] = arr

    return data


def bbox_to_array(arr, label=0, max_bboxes=64, bbox_width=16):
    """
    Converts a 1-dimensional bbox array to an image-like
    3-dimensional array CHW array
    """
    arr = pad_bbox(arr, max_bboxes, bbox_width)
    return arr[np.newaxis, :, :]


def bbox_overlap(abox, bbox):
    # the abox box
    x11 = abox[0]
    y11 = abox[1]
    x12 = abox[0] + abox[2] - 1
    y12 = abox[1] + abox[3] - 1

    # the closer box
    x21 = bbox[0]
    y21 = bbox[1]
    x22 = bbox[0] + bbox[2] - 1
    y22 = bbox[1] + bbox[3] - 1

    overlap_box_x2 = min(x12, x22)
    overlap_box_x1 = max(x11, x21)
    overlap_box_y2 = min(y12, y22)
    overlap_box_y1 = max(y11, y21)

    # make sure we preserve any non-bbox components
    overlap_box = list(bbox)
    overlap_box[0] = overlap_box_x1
    overlap_box[1] = overlap_box_y1
    overlap_box[2] = overlap_box_x2-overlap_box_x1+1
    overlap_box[3] = overlap_box_y2-overlap_box_y1+1

    xoverlap = max(0, overlap_box_x2 - overlap_box_x1)
    yoverlap = max(0, overlap_box_y2 - overlap_box_y1)
    overlap_pix = xoverlap * yoverlap

    return overlap_pix, overlap_box

def resize_bbox_list(bboxlist, rescale_x=1, rescale_y=1):
        # this is expecting x1,y1,w,h:
        bboxListNew = []
        for bbox in bboxlist:
            abox = bbox
            abox[0] *= rescale_x
            abox[1] *= rescale_y
            abox[2] *= rescale_x
            abox[3] *= rescale_y
            bboxListNew.append(abox)
        return bboxListNew


