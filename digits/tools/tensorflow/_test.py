#!/usr/bin/python

# This file is a scratchpad used to test main.py

import os
import subprocess

#dataset_dir = "/Users/tzaman/jobs/20160715-230349-5f23" #CIFAR100
#dataset_dir = "/Users/tzaman/jobs/20160715-230434-21a4" #CIFAR10 LMDB
#dataset_dir = "/Users/tzaman/jobs/20160615-215643-75fd" #MNIST LMDB
#dataset_dir = "/Users/tzaman/jobs/gradient_regression" #Gradient Regression
#dataset_dir = "/Users/tzaman/jobs/20161002-185828-d0cd" #Triangle-segmentation
#dataset_dir = "/Users/tzaman/jobs/20161014-112206-c4ec" #MNIST HDF5 Uncompressed
#dataset_dir = "/Users/tzaman/jobs/20161014-112335-a0f1" #MNIST HDF5 GZIP
#dataset_dir = "/Users/tzaman/jobs/20161014-151839-2c91" #CIFAR10 HDF5 32x32

dataset_dir = "/home/tzaman/jobs/20161011-143422-78cb" #MNIST
#dataset_dir = "/home/tzaman/jobs/20161019-200547-1dcb" #CIFAR10
#dataset_dir = "/home/tzaman/jobs/20161019-204719-79a4" #CIFAR10-256

# TIM'S OVERRIDES:
args = ""

if 1:
	args = args + (
	" --labels=" + dataset_dir + "/labels.txt"
	" --networkDirectory=../../standard-networks/tensorflow"
	#" --network=gradient.py"
	" --network=lenet.py"
	#" --network=alexnet-from-28.py"
	#" --network=alexnet.py"
	#" --network=siamese_simple.py"
	#" --network=lenet_slim.py"
	#" --network=rnn_mnist.py"
	#" --network=autoencoder.py"
	#" --network=binary_segmentation.py"
	" --mean=" + dataset_dir + "/mean.binaryproto"
	" --subtractMean=pixel"
	#" --subtractMean=image"
#	" --croplen=224"
	" --train_db=" + dataset_dir + "/train_db"
	" --validation_db=" + dataset_dir + "/val_db"
	" --summaries_dir=/tmp/tb/"
#	" --save=/Users/tzaman/Desktop/result"
	" --seed=1"
	" --epoch=1"
	" --interval=1"
	#" --tf_summaries_dir=/Users/tzaman/Desktop/tb/"
	" --shuffle=False"
	#" --optimization=adam"
	#" --weights=/Users/tzaman/Desktop/result/loadme2"
	" --batch_size=128"
	" --log_runtime_stats_per_step=0"
	" --snapshotInterval=0"
	" --type=cpu"
	)

if 0: #TFRecords
	args = args + (
		#" --train_db=/Users/tzaman/Desktop/tfrecords_mnist/train.tfrecords"
		" --train_db=/home/tzaman/tfrecords_mnist/train.tfrecords"
		)

if 0: #Load weights for plain lenet
	args = args + (
		" --weights=/Users/tzaman/jobs/20161014-173513-623a/snapshot_1.0_Model/Model.ckpt"
		" --croplen=28"
		)

if 0: #LR and Optimizer
	args = args + (
		" --lr_policy=exp"
		" --lr_gamma=0.98"
		#" --lr_power=1"
		" --lr_base_rate=0.1"
		" --optimization=sgd"
		)

if 0: #Visualize Network
	args = args + (
	" --networkDirectory=../../digits/standard-networks/tensorflow"
	" --network=lenet_slim.py"
	" --visualizeModelPath=/Users/tzaman/Desktop/graphtest/x/test.pbtxt"
	)

if 0: # toggle if using a dataset with labels in a db
	args = args + (
		" --train_labels=" + dataset_dir + "/train_db_labels"
		" --validation_labels=" + dataset_dir + "/val_db_labels"
		)

if 0: # Inference
	dataset_dir = "/Users/tzaman/jobs/20160615-215643-75fd" #MNIST
	network_dir = "/Users/tzaman/jobs/20161015-162923-2ce8" #MNIST
	inference_db = "/Users/tzaman/Desktop/zes.png"
	#inference_db = "/Users/tzaman/Desktop/list.txt"
	args = args + (
		" --inference_db=" + inference_db + ""
		" --batch_size=1"
		" --labels=" + dataset_dir + "/labels.txt"
		" --mean=" + dataset_dir + "/mean.binaryproto"
		" --subtractMean=pixel"
		" --network=network.py"
		" --networkDirectory=" + network_dir + ""
		" --weights=" + network_dir + "/snapshot_1.0.ckpt"
		#" --allPredictions=1"
		#" --visualization=False"
		#" --testMany=False"
		#" --testUntil=-1"
		)
	if 1: #with visualize
		args = args + (
			" --visualize_inf=1"
			)



# For some reason, the DYLD_LIBRARY_PATH is not copied with it, so supply in-line:

#cmd = "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib /opt/local/bin/python2.7 main.py" + args
cmd = "python main.py" + args

p = subprocess.Popen(cmd,
        close_fds=True,
        env=os.environ.copy(),
        shell=True
        ).wait()
