
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import copy
import os
import string

import numpy as np


def encode_field(field, word_map, sentence_size, story_size):
    """
    return a 2-D array with shape (story_size, sentence_size)
    """
    x = np.zeros((story_size, sentence_size))
    for i, sentence in enumerate(field):
        if i >= story_size:
            raise ValueError("Field '%s' is longer than max (%d)" %
                             (field, story_size))
        for j, word in enumerate(sentence):
            if j >= sentence_size:
                raise ValueError("Sentence '%s' is longer than max (%d)" %
                                 (field, sentence_size))
            try:
                idx = word_map[word]
            except:
                # assign to last index
                idx = len(word_map) + 1
            x[i, j] = idx
    return x


def encode_sample(sample, word_map, sentence_size, story_size):
    """
    return an encoded (feature, label) tuple
    """
    story = encode_field(sample['story'], word_map, sentence_size, story_size)
    question = encode_field(sample['question'], word_map, sentence_size, story_size)
    answer = encode_field(sample['answer'], word_map, sentence_size, story_size)

    feature = np.zeros((2, story_size, sentence_size))
    feature[0] = story
    feature[1] = question

    label = answer[np.newaxis, :]

    return feature, label


def find_files(path, task_id, train):
    """
    Find files in specified path with filenames that
    match {task}*{phase}.txt where:
        task="qa{task_id}_" or "" if task_id==None
        phase="train" if train==True or "test" otherwise
    """
    task = "qa{}_".format(task_id) if task_id else ""
    phase = "train" if train else "test"

    files = []
    for dirpath, dirnames, filenames in os.walk(path, followlinks=True):
        for filename in filenames:
            if task in filename and phase in filename:
                files.append(os.path.join(dirpath, filename))

    return files


def get_stats(dataset):
    """
    return dataset statistics
    """
    fields = [field for sample in dataset for field in sample.values()]
    sentences = [sentence for field in fields for sentence in field]
    words = sorted(set([word for sentence in sentences for word in sentence]))

    return {'word_map': dict((word, i) for i, word in enumerate(words, start=1)),
            'sentence_size': max([len(sentence) for sentence in sentences]),
            'story_size': max([len(story) for story in fields])}


def parse_folder_phase(path, task_id, train):
    """
    Returns a list of samples for a phase by aggregating all samples
    from matching files
    """
    phase_data = []
    files = find_files(path, task_id, train)
    for file in files:
        phase_data.extend(parse_file(file))
    return phase_data


def parse_file(filename):
    with open(filename) as f:
        return parse_lines(f.readlines())


def parse_lines(lines):
    """
    Returns a list of samples from a collection of lines where each sample
    is a dictionary with 'story', 'question', 'answer' keys. Every key
    value is a list of words without punctuation.
    """
    data = []
    print "lines are %s" % lines
    story = None
    for line in lines:
        # convert to lower case
        line = line.lower()
        # find line ID (new stories start with line ID = 1)
        line_id, line = line.split(' ', 1)
        try:
            if int(line_id) == 1:
                # new story
                story = []
        except:
            if not story:
                story = []
            # this isn't a like id, re-integrate into line
            line = "%s %s" % (line_id, line)
        # is this a question?
        if '?' in line:
            items = remove_punctuation(line).split('\t')
            question = items[0]
            if len(items) > 1:
                answer = items[1]
            else:
                answer = ''
            # add to data
            data.append({
                'story': copy.copy(story),
                'question': [question.split()],
                'answer': [answer.split()],
                })
        else:
            story.append(remove_punctuation(line).split())
    return data


def remove_punctuation(s):
    return s.translate(string.maketrans("", ""), string.punctuation)
