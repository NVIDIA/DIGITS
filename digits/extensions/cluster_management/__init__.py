import os
import cluster_factory
import slurm

if os.environ.get('JENKINS_URL') is not None:
    cluster_factory.cluster_factory.set_system('slurm')
    slurm.get_digits_tmpdir()
