#!/usr/bin/env python2
# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
import os
import pickle
import shutil

import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

TB_DIR = os.path.join(os.getcwd(), "gan-tb")
SPRITE_IMAGE_FILENAME = os.path.join(TB_DIR, "sprite.png")


def save_tb_embeddings(embeddings_filename):
    f = open(embeddings_filename, 'rb')
    embeddings = pickle.load(f)

    images = embeddings['images']
    zs = embeddings['zs']

    # overwrite Tensorboard log dir if necessary
    if os.path.exists(TB_DIR):
        shutil.rmtree(TB_DIR)
    os.makedirs(TB_DIR)

    # create grid image
    img_width, img_height = save_sprite_image(images)

    with tf.device('cpu:0'):
        # create embedding var
        embedding_var = tf.Variable(initial_value=zs)

        # save projector config
        summary_writer = tf.summary.FileWriter(TB_DIR)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.sprite.image_path = SPRITE_IMAGE_FILENAME
        embedding.sprite.single_image_dim.extend([img_width, img_height])
        projector.visualize_embeddings(summary_writer, config)

        # save embeddings
        sess = tf.Session()
        sess.run(embedding_var.initializer)
        saver = tf.train.Saver([embedding_var])
        saver.save(sess, os.path.join(TB_DIR, 'model.ckpt'))


def save_sprite_image(images):
    n_embeddings = images.shape[0]
    grid_cols = int(np.sqrt(n_embeddings))
    grid_rows = int(np.ceil(float(n_embeddings) / grid_cols))
    img_height, img_width, img_channels = images[0].shape
    grid_image = np.empty((img_height * grid_rows, img_width * grid_cols, img_channels))
    for i, image in enumerate(images):
        row = i / grid_cols
        col = i % grid_cols
        x = img_width * col
        y = img_height * row
        grid_image[y:y + img_height, x:x + img_width] = image
    grid_image = PIL.Image.fromarray(grid_image.astype('uint8'))
    grid_image.save(SPRITE_IMAGE_FILENAME)
    return img_width, img_height


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference tool - DIGITS')

    # Positional arguments

    parser.add_argument(
        'embeddings_file',
        help='Embeddings pickle file')

    args = vars(parser.parse_args())

    try:
        save_tb_embeddings(
            args['embeddings_file'],
        )
    except Exception as e:
        print('%s: %s' % (type(e).__name__, e.message))
        raise
