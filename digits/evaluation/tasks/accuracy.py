# -*- coding: utf-8 -*-
from collections import Counter
from digits.task import Task
from digits.utils import subclass, constants
import numpy as np
import pandas as pd

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class AccuracyTask(Task):
    """
    Defines required methods for child accuracy classes
    """

    def __init__(self, **kwargs):
        super(AccuracyTask, self).__init__(**kwargs)

    def __getstate__(self):
        state = super(AccuracyTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(AccuracyTask, self).__setstate__(state)

    def avg_accuracy_graph_data(self):
        """
        Returns the accuracy/recall datas formatted for a C3.js graph
        """

        if self.probas_data is None:
            return None

        def f_threshold(threshold, probas):
            N = len(probas)
            max_probs = np.max(probas, axis=1)
            mask_threshold = np.ma.masked_where(max_probs<threshold, max_probs)

            labels_masked = np.ma.compressed(np.ma.masked_array(self.labels_data, mask_threshold.mask))
            predict_masked = np.ma.compressed(np.ma.masked_array(self.prediction_data, mask_threshold.mask))

            N_threshold = predict_masked.shape[0]/float(N)
            acc = np.mean(labels_masked==predict_masked)
            return acc, N_threshold

        t = ['Threshold']
        accuracy = ['Accuracy']
        response = ['Recall']

        max_proba = np.max(self.probas_data)

        for i in range(1000):
            acc, num = f_threshold(max_proba * i / 1000.0, self.probas_data)
            t += [max_proba * i / 1000.0]
            accuracy += [acc]
            response += [num]
        return  {
            "x": "Threshold",
            "columns": [ t, accuracy, response ],
            "axes": {
                'Recall': 'y2'
            }
        }


    def confusion_matrix_data(self):
        """
        Returns the confusion matrix datas formatted in the form of a string
        TODO: return a dictionnary and make the formatting in the template
        """
        if self.probas_data is None:
            return None

        train_task = self.job.model_job.train_task()
        dataset_train_task = train_task.dataset.train_db_task()
        dataset_labels = train_task.dataset.labels_file
        labels_str = pd.read_csv(dataset_train_task.path(dataset_labels),header=None,sep="", engine='python')[0]

        def accuracy_per_class(class_index):
            label_flat = self.labels_data.tolist()
            try:
                start = label_flat.index(class_index)
                stop = (len(label_flat) - 1) - label_flat[::-1].index(class_index)
                return np.mean(self.prediction_data[start:stop+1]==self.labels_data[start:stop+1])
            except:
                return None

        def most_represented_class_per_class(class_index):
            label_flat = self.labels_data.tolist()
            try:
                start = label_flat.index(class_index)
                stop = (len(label_flat) - 1) - label_flat[::-1].index(class_index)
                c = Counter(self.prediction_data[start:stop+1])
                return map(lambda x:(labels_str[x[0]], x[1]/float(stop-start+1)),c.most_common())
            except:
                return None

        results = []
        s = ""
        for i in range(0,len(labels_str)):
            acc = accuracy_per_class(i)

            if acc is not None:
                res = { 'label': labels_str[i] }
                res['acc'] = acc
                res['classes'] = []
                s += "{0} - {1}%\n".format(labels_str[i], round(acc * 100,2))
                classes = most_represented_class_per_class(i)
                for k in classes[0:10]:
                    res['classes'].append({ 'label' : k[0], 'acc' : k[1] })
                    s += "\t{1}%\t -\t {0}\n".format(k[0], round(k[1]*100, 2))
                results.append(res)

        return  { "results" : results, "text" : s }

