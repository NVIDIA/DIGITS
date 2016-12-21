from digits.config import config_value
import slurm
class cluster_factory:
    system_type = config_value('system_type')
    use_cluster = True

    def __init__(self):
        pass
    def get_cluster_manager(self):
        if cluster_factory.system_type == 'slurm':
            return slurm.slurm_manager()

