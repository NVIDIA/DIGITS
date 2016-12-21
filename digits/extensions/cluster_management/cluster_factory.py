
import slurm
from digits.config import option_list
class cluster_factory:
    system_type = ''
    selected_system = 'interactive'
    use_cluster = True

    def __init__(self):
        pass

    def get_cluster_manager(self):
        if cluster_factory.system_type == 'slurm':
            return slurm.slurm_manager()

    def get_running_systems(self):
        running_systems = ['interactive']
        if slurm.test_if_slurm_system():
            running_systems.append('slurm')
        # add more systems here
        return running_systems
    @staticmethod
    def set_system(system):
        cluster_factory.selected_system = system
        option_list['system_type'] = cluster_factory.selected_system

    # enter this to set system_type
    # cluster_factory.cluster_factory.selected_system = "slurm"
    # cluster_factory.cluster_factory.set_system("slurm")

