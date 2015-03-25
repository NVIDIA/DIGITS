# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from digits.task import Task

# NOTE: Increment this everytime the picked object changes
PICKLE_VERSION = 1

class TrainTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, dataset, train_epochs, snapshot_epochs, learning_rate, lr_policy, **kwargs):
        """
        Arguments:
        dataset -- a DatasetJob containing the dataset for this model
        train_epochs -- how many epochs of training data to train on
        snapshot_epochs -- how many epochs to take a snapshot
        learning_rate -- the base learning rate
        lr_policy -- a hash of options to be used for the learning rate policy

        Keyword arguments:
        batch_size -- if set, override any network specific batch_size with this value
        val_interval -- how many epochs in-between validating the model with an epoch of validation data
        pretrained_model -- filename for a model to use for fine-tuning
        crop_size -- crop each image down to a square of this size
        use_mean -- subtract the dataset's mean file
        """
        self.batch_size = kwargs.pop('batch_size', None)
        self.val_interval = kwargs.pop('val_interval', None)
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.crop_size = kwargs.pop('crop_size', None)
        self.use_mean = kwargs.pop('use_mean', None)

        super(TrainTask, self).__init__(**kwargs)
        self.pickver_task_train = PICKLE_VERSION

        self.dataset = dataset
        self.train_epochs = train_epochs
        self.snapshot_epochs = snapshot_epochs
        self.learning_rate = learning_rate
        self.lr_policy = lr_policy

        self.snapshots = []

        # graph data
        self.train_loss_updates = []
        self.val_loss_updates = []
        self.val_accuracy_updates = []
        self.lr_updates = []

    def __getstate__(self):
        state = super(TrainTask, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        if 'snapshots' in state:
            del state['snapshots']
        if 'labels' in state:
            del state['labels']
        return state

    def __setstate__(self, state):
        super(TrainTask, self).__setstate__(state)

        self.snapshots = []
        self.detect_snapshots()
        self.dataset = None

    def detect_snapshots(self):
        """
        Populate self.snapshots with snapshots that exist on disk
        Returns True if at least one usable snapshot is found
        """
        return False

    def snapshot_list(self):
        """
        Returns an array of arrays for creating an HTML select field
        """
        return [[s[1], 'Epoch#Iter%s' % s[1]] for s in reversed(self.snapshots)]

    def est_next_snapshot(self):
        """
        Returns the estimated time in seconds until the next snapshot is taken
        """
        return None

    def can_view_weights(self):
        """
        Returns True if this Task can visualize the weights of each layer for a given model
        """
        raise NotImplementedError()

    def view_weights(self, model_epoch=None, layers=None):
        """
        View the weights for a specific model and layer[s]
        """
        return None

    def can_infer_one(self):
        """
        Returns True if this Task can run inference on one input
        """
        raise NotImplementedError()

    def can_view_activations(self):
        """
        Returns True if this Task can visualize the activations of a model after inference
        """
        raise NotImplementedError()

    def infer_one(self, data, model_epoch=None, layers=None):
        """
        Run inference on one input
        """
        return None

    def can_infer_many(self):
        """
        Returns True if this Task can run inference on many inputs
        """
        raise NotImplementedError()

    def infer_many(self, data, model_epoch=None):
        """
        Run inference on many inputs
        """
        return None

    def read_labels(self):
        """
        Read labels from self.labels_file and store them at self.labels
        Returns True if at least one label was read
        """
        # The labels might be set already
        if hasattr(self, 'labels') and self.labels and len(self.labels) > 0:
            return True

        assert hasattr(self.dataset, 'labels_file'), 'labels_file not set'
        assert self.dataset.labels_file, 'labels_file not set'
        assert os.path.exists(self.dataset.path(self.dataset.labels_file)), 'labels_file does not exist'

        labels = []
        with open(self.dataset.path(self.dataset.labels_file)) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)

        assert len(labels) > 0, 'no labels in labels_file'

        self.labels = labels
        return True

    def lr_graph_data(self):
        """
        Returns learning rate data formatted for a Google Charts graph
        """
        if not hasattr(self, 'lr_updates') or not self.lr_updates:
            return None

        data = [['Epoch', 'Learning Rate']]
        lru = self.lr_updates
        lru = lru[::max(len(lru)/100,1)] # return 100-200 values or fewer
        for epoch, lr in lru:
            data.append([epoch, lr])
        return data

    def loss_graph_data(self):
        """
        Returns loss and/or accuracy data formatted for a Google Charts graph
        """
        tl = self.train_loss_updates
        vl = self.val_loss_updates
        va = self.val_accuracy_updates
        # return 100-200 values or fewer
        tl = tl[::max(len(tl)/100,1)]
        vl = vl[::max(len(vl)/100,1)]
        va = va[::max(len(va)/100,1)]

        use_tl = len(tl) > 0
        use_vl = len(vl) > 0
        use_va = len(va) > 0

        if not (use_tl or use_vl or use_va):
            return None

        data = []
        titles = ['Epoch']
        if use_tl:  titles.append('Loss (train)')
        if use_vl:  titles.append('Loss (val)')
        if use_va:  titles.append('Accuracy (val)')
        data.append(titles)

        # Iterators
        tli = 0
        vli = 0
        vai = 0

        # decimal points for different data types
        round_loss = 3
        round_acc = 2

        while tli < len(tl) or vli < len(vl) or vai < len(va):
            next_it = []
            if tli < len(tl):   next_it.append(tl[tli][0])
            if vli < len(vl):   next_it.append(vl[vli][0])
            if vai < len(va):   next_it.append(va[vai][0])
            it = min(next_it)

            ### Loss values

            tl_value = 'null'
            vl_value = 'null'
            va_value = 'null'

            if tli < len(tl) and tl[tli][0] == it:
                tl_value = round(tl[tli][1], round_loss)
                tli += 1
            if vli < len(vl) and vl[vli][0] == it:
                vl_value = round(vl[vli][1], round_loss)
                vli += 1
            if vai < len(va) and va[vai][0] == it:
                va_value = round(va[vai][1], round_acc)
                vai += 1

            entry = [it]
            if use_tl:  entry.append(tl_value)
            if use_vl:  entry.append(vl_value)
            if use_va:  entry.append(va_value)
            data.append(entry)

        return data

