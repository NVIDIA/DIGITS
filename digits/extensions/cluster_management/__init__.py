import os
import cluster_factory

if os.environ.get('JENKINS_URL') is not None:
    cluster_factory.cluster_factory.set_system('slurm')
