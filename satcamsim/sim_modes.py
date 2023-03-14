from enum import Enum


class Sim_modes(Enum):
    """
    Enumeration of simulation modes.

    Members
    ----------
    DEFAULT : 0
        sequential image acquisition, corresponds to Camera.default_swath()

    PROCESSES : 1
        multiprocessing, corresponds to Camera.multiprocess_swath()

    THREADS : 2
        multithreading, corresponds to Camera.multithread_swath()

    CHUNKS : 3
        save partial results in temp files to avoid filling CPU cache, corresponds to Camera.chunky_swath()

    """

    DEFAULT = 0
    PROCESSES = 1
    THREADS = 2
    CHUNKS = 3
