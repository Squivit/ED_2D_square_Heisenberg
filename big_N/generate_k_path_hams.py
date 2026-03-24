from Betts_cluster import Cluster
from clusters_lattice_bigN_fragmented import SpinConfiguration
import numpy as np

if __name__ == "__main__":
    N = 32
    num = 'A'
    key = str(N)+num

    delta = 2.
    lamb = 0.

    c = Cluster(key=key)
    path = c.get_k_space(False)[1]

    #sc = SpinConfiguration(N=N, number=num, key=key, magnon_number=int(N/2), lowest_eignstates=5, delta = delta, lamb = lamb,
    #                       k=np.array([0., 0.]), print_data=True, force_ham_gen=False, save_ham=True)

    for k in path[7:]:
        print()
        print(f'Calculating k = {k} block')

        sc = SpinConfiguration(N=N, number=num, key=key, magnon_number=int(N/2)+1, lowest_eignstates=5,
                            delta = delta, lamb = lamb, k=k, print_data=True, force_ham_gen=False, save_ham=True)
