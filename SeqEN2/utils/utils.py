from numpy.random import seed
from torch import cuda, manual_seed


def get_map_location():
    if cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"
    return map_location


def set_random_seed(num):
    seed(num)
    manual_seed(num)
