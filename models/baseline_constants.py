"""Configuration file for common models/experiments"""

MAIN_PARAMS = {
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'celeba': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'cifar100': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
    'cifar10': {
        'small': (1000, 100, 10),
        'medium': (10000, 100, 10),
        'large': (20000, 100, 10)
        },
    'inaturalist': {
        'small': (1000, 100, 10),
        'medium': (2500, 100, 10),
        'large': (5000, 100, 10)
    }
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'femnist.cnn': (0.0003, 62), # lr, num_classes
    'celeba.cnn': (0.1, 2), # lr, num_classes
    'celeba.mobilenet': (0.1, 2),
    'cifar100.cnn': (0.01, 100),
    'cifar10.cnn': (0.01, 10),
    'inaturalist.mobilenet': (0.01, 1203)
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
