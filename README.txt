                       DiGiTS (Deep GPU Training System)

What is it?
-----------

DiGiTS is is a webapp for training deep learning models.


Installation
------------

0.  Build NVIDIA's fork of caffe
    git clone https://github.com/NVIDIA/caffe.git
    (follow the instructions here:
        http://caffe.berkeleyvision.org/installation.html)

1.  Install python dependencies
        sudo pip install -r requirements.txt

2.  Install graphviz (optional, used to visualize networks)
        sudo apt-get install graphviz

3.  Add the location of your caffe installation to your environment
        export CAFFE_HOME='/home/username/caffe'


Use
---

You can run DiGiTS in two ways:

1.  digits-devserver
        Starts a development server that listens on port 5000 (but you can
        change the port if you like - try running it with the --help flag).

2.  digits-server
        Starts a gunicorn app that listens on port 8080. If you have installed
        the nginx.site to your nginx sites-enabled/ directory, then you can
        view your app at http://localhost/.

