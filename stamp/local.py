class LocalConfig:

    def __init__(self, base_dir):
        self._base_dir = base_dir

class MyLocalConfig(LocalConfig):
    def __init__(self):
        self.moment_models_dir = 'moment/MOMENT-1-large'
        self.tspulse_models_dir = '/path/to/tspulse_models'
        self.chronos_models_dir = '/path/to/chronos_models'
        self.datasets_dir = 'dataset/'
        self.tsfm_experiments_dir = 'experiments'
        self.tmp_dir = 'temporary'
        self.processed_data_dirs = {
            'bciciv2a': '/path/to/benchmark_data/bciciv2a/processed_inde_avg_03_50',
            'chbmit': '/path/to/benchmark_data/chbmit/processed_lmdb',
            'faced': '/path/to/benchmark_data/faced/processed',
            'isruc': '/path/to/benchmark_data/isruc/processed',
            'mumtaz': '/path/to/benchmark_data/mumtaz/processed_lmdb_75hz',
            'physio': '/dataset/processed_physionet/processed_average',
            'seedv': '/path/to/benchmark_data/seedv/processed',
            'seedvig': '/path/to/benchmark_data/seedvig/processed',
            'shu': '/path/to/benchmark_data/shu/processed',
            'speech': '/path/to/benchmark_data/speech/processed',
            'stress': '/path/to//benchmark_data/stress/processed',
            'tuab': '/path/to/benchmark_data/tuab/processed',
            'tuev': '/path/to/benchmark_data/tuev/processed',
            'sere': 'dataset/processed_sere/sere_framecomp_world_xyz_L64_S16.lmdb'

        }

def get_local_config():
    return MyLocalConfig()

local_config : LocalConfig = get_local_config()
