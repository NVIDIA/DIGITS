Torch code can also be run from command line as shown below:

To Train :
----------

th main.lua --train=/home/ubuntu/.digits/jobs/20150407-174547-f8f1/train_db --validation=/home/ubuntu/.digits/jobs/20150407-174547learningRateDecay-f8f1/val_db --network=lenet --networkDirectory=../../digits/standard-networks/torch/ --epoch=10 --save=/home/ubuntu/.digits/jobs/20150416-165008-4309 --snapshotPrefix=snapshot --snapshotInterval=1.000000 --subtractMean=yes --useMeanPixel=yes --mean=mean.jpg --labels=labels.txt --batchSize=32 --interval=1.000000 --learningRate=0.010000 --policy=step --gamma=0.100000 --stepvalues=33.000000

main.lua code uses data.lua module to load the images for training and validation, from lmdb.

--train=/home/ubuntu/.digits/jobs/20150407-174547-f8f1/train_db     => specifies train_db lmdb file contains all the images for training
--validation=/home/ubuntu/.digits/jobs/20150407-174547-f8f1/val_db  => specifies val_db lmdb file contains all the images for validation. This is an optional. Validations won't be done if not provided.
--network=lenet                                                     => specifies that the network file is "lenet.lua"
--networkDirectory=../../digits/standard-networks/torch/            => specifies that "lenet.lua" is present in "../../digits/standard-networks/torch/" directory
--epoch=10                                                          => specifies the total number of epochs
--save=/home/ubuntu/.digits/jobs/20150416-150654-37ca               => specifies the directory where weights file (named <SNAPSHOT_PREFIX>_<EPOCH_VALUE>_Weights.t7) and optimState (named optimState_<EPOCH_VALUE>.t7), are saved
--snapshotPrefix=snapshot                                           => specifies the snapshot prefix of weights file of trained model
--snapshotInterval=1.000000                                         => specifies after every 1 training epoch, weights and optimState will be saved.
                                                                       For instance, if this value is 1.21, then weights and optimState are saved for every 1.21 training epochs.
Note: Training will be done in batches, so some times saving weights (and optimState) for the given epoch value won't be possible. In this case the epoch value near to the given value will be considered.

--subtractMean=yes                                                  => subtract mean from the test image. Default is 'yes'
--useMeanPixel=yes                                                  => specifies to use mean pixel instead of mean full matrix during image preprocessing. Default is using mean full matrix.
                                                                       Below links contains more information about the same:
                                                                       https://groups.google.com/forum/#!topic/torch7/66LMB-F-0ME
                                                                       https://github.com/BVLC/caffe/issues/2069

--mean=mean.jpg                                                     => use mean.jpg file as mean file
--labels=labels.txt                                                 => specifies labels file. Each line in labels file specifies a distinct label name
--batchSize=32                                                      => specifies the batch size. Make sure that the batch size is multiple of 32, when ccn2 network is used.
--interval=1.000000                                                 => specifies that the model has to be validated against validation data for every one training epoch. 
                                                                       If this value is 0.5, then the model is validated for every half training epoch

Implemented Caffe kind of learning policies in Torch. Please refer to "LRPolicy.lua" for more details regarding Learning Policies.
Here, learningRate, policy, gamma and stepvalues are the parameters of learning policy. 
Note: if you want to use normal torch way of learning rate recalculation by SGD.lua, then use "torch_sgd" with policy parameter and also provide additional parameters like "learningRate", "learningRateDecay" as shown below,
--policy=torch_sgd
--learningRate=<some_value>
--learningRateDecay=<some_value>

To Test a single image : 
------------------------

th test.lua --image=/tmp/tmp4HHdV4.jpeg --network=lenet --networkDirectory=../../digits/standard-networks/torch/ --load=/home/ubuntu/.digits/jobs/20150416-150654-37ca --snapshotPrefix=snapshot --epoch=30 --subtractMean=yes --useMeanPixel=yes --mean=mean.jpg --labels=labels.txt

where,
--image=/tmp/tmp4HHdV4.jpeg                                         => specifies the image to be tested
--network=lenet                                                     => specifies that the network file is "lenet.lua"
--networkDirectory=../../digits/standard-networks/torch/            => specifies that "lenet.lua" is present in "../../digits/standard-networks/torch/" directory
--load=/home/ubuntu/.digits/jobs/20150416-150654-37ca               => specifies that trained network, named <SNAPSHOT_PREFIX>_<EPOCH_VALUE>_Weights.t7, exists in /home/ubuntu/.digits/jobs/20150416-150654-37ca
--snapshotPrefix=snapshot                                           => specifies the snapshot prefix of weights file
--epoch=30                                                          => specifies the weights file that was deployed during epoch 30 has to be loaded
--subtractMean=yes                                                  => subtract mean from the test image. Default is 'yes'
--useMeanPixel=yes                                                  => specifies to use mean pixel instead of mean full matrix during image preprocessing. Default is using mean full matrix.
                                                                       Below links contains more information about the same:
                                                                       https://groups.google.com/forum/#!topic/torch7/66LMB-F-0ME
                                                                       https://github.com/BVLC/caffe/issues/2069

--mean=mean.jpg                                                     => use mean.jpg file as mean file
--labels=labels.txt                                                 => specifies labels file. Each line in labels file specifies a distinct label name

