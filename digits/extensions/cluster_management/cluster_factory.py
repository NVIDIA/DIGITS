
import slurm
from cluster_manager import cluster_manager
from digits.config import option_list
class cluster_factory:
    selected_system = 'interactive'
    use_cluster = True

    def __init__(self):
        pass

    def get_cluster_manager(self):
        if cluster_factory.selected_system == 'slurm':
            return slurm.slurm_manager()
    @staticmethod
    def get_running_systems():
        running_systems = ['interactive']
        if slurm.test_if_slurm_system():
            running_systems.append('slurm')
        # add more systems here
        return running_systems
    @staticmethod
    def set_system(system):
        cluster_factory.selected_system = system
        option_list['system_type'] = cluster_factory.selected_system
        if cluster_factory.selected_system == 'slurm':
            slurm.get_digits_tmpdir()

    # enter this to set system_type
    # cluster_factory.cluster_factory.selected_system = "slurm"
    # cluster_factory.cluster_factory.set_system("slurm")

