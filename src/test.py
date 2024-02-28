import sys

sys.path.append('../')

print('aaaa')

import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


for ff in range(2):
    # if ff == 50:
    #     # comm.barrier()

    c = comm.gather(ff, root=0)

    if rank == 0:
        d = np.mean(np.array(c))
    else:
        d = None

    data = comm.bcast(d, root=0)

    print(rank, data)
